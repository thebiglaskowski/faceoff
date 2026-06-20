"""
Unit tests for model pool thread-safety and concurrency.

Tests cover:
- ModelPool singleton pattern
- Thread-safe model instance management
- Concurrent GPU access patterns
- Resource cleanup
"""

import numpy as np
import pytest
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_model_pool():
    """Reset model pool singleton between tests."""
    from core import model_pool as mp_module

    # Store original value
    original_instance = mp_module.ModelPool._instance

    # Reset singleton state
    mp_module.ModelPool._instance = None

    yield

    # Restore original value
    mp_module.ModelPool._instance = original_instance


@pytest.fixture
def mock_gpu_dependencies():
    """Mock all GPU-related dependencies for model pool testing."""
    with patch('core.model_pool.torch') as mock_torch, \
         patch('core.model_pool.ort') as mock_ort, \
         patch('core.model_pool.FaceAnalysis') as mock_face_analysis, \
         patch('core.model_pool.is_tensorrt_runtime_available', return_value=False), \
         patch('core.model_pool.build_face_analysis_providers', return_value=[('CUDAExecutionProvider', {}), 'CPUExecutionProvider']), \
         patch('core.model_pool.build_swapper_providers', return_value=[('CUDAExecutionProvider', {}), 'CPUExecutionProvider']), \
         patch('core.model_pool.config') as mock_config:

        # Configure mock torch.cuda
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.cuda.empty_cache = MagicMock()

        # Configure mock config
        mock_config.face_analysis_name = "buffalo_l"
        mock_config.face_analysis_det_size = [640, 640]
        mock_config.tensorrt_enabled = False
        mock_config.tensorrt_workspace_mb = 1024
        mock_config.tensorrt_fp16 = False
        mock_config.inswapper_model_path = "test_model.onnx"
        mock_config.get = MagicMock(return_value=2048)  # onnx_mem_limit_mb

        # Configure mock FaceAnalysis
        mock_face_app = MagicMock()
        mock_face_app.get.return_value = []
        mock_face_app.models = {}
        mock_face_analysis.return_value = mock_face_app

        # Configure mock ort.SessionOptions
        mock_session_options = MagicMock()
        mock_ort.SessionOptions.return_value = mock_session_options
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

        yield {
            'torch': mock_torch,
            'ort': mock_ort,
            'face_analysis': mock_face_analysis,
            'config': mock_config,
        }


# =============================================================================
# ModelPool Singleton Tests
# =============================================================================

class TestModelPoolSingleton:
    """Tests for ModelPool singleton pattern."""

    def test_singleton_returns_same_instance(self, mock_gpu_dependencies):
        """Test ModelPool returns the same instance."""
        from core.model_pool import ModelPool

        pool1 = ModelPool()
        pool2 = ModelPool()

        assert pool1 is pool2

    def test_get_model_pool_returns_singleton(self, mock_gpu_dependencies):
        """Test get_model_pool returns singleton."""
        from core.model_pool import get_model_pool

        pool1 = get_model_pool()
        pool2 = get_model_pool()

        assert pool1 is pool2

    def test_singleton_thread_safe(self, mock_gpu_dependencies):
        """Test singleton creation is thread-safe."""
        from core.model_pool import ModelPool

        pools = []
        creation_lock = threading.Lock()

        def create_pool():
            pool = ModelPool()
            with creation_lock:
                pools.append(pool)

        threads = [threading.Thread(target=create_pool) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All references should be to the same instance
        assert len(pools) == 10
        assert all(p is pools[0] for p in pools)


# =============================================================================
# ModelPool Instance Management Tests
# =============================================================================

class TestModelPoolInstanceManagement:
    """Tests for model instance management."""

    @patch('core.model_pool.GPUModelInstance')
    def test_get_instance_creates_new(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test get_instance creates instance on first access."""
        from core.model_pool import get_model_pool

        mock_instance = MagicMock()
        mock_gpu_instance.return_value = mock_instance

        pool = get_model_pool()
        pool.set_model_path("test.onnx")

        instance = pool.get_instance(0)

        assert instance is mock_instance
        mock_gpu_instance.assert_called_once()

    @patch('core.model_pool.GPUModelInstance')
    def test_get_instance_returns_cached(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test get_instance returns cached instance on subsequent calls."""
        from core.model_pool import get_model_pool

        mock_instance = MagicMock()
        mock_gpu_instance.return_value = mock_instance

        pool = get_model_pool()
        pool.set_model_path("test.onnx")

        instance1 = pool.get_instance(0)
        instance2 = pool.get_instance(0)

        assert instance1 is instance2
        # Should only create once
        assert mock_gpu_instance.call_count == 1

    @patch('core.model_pool.GPUModelInstance')
    def test_get_instance_recreates_after_vram_release(
        self, mock_gpu_instance, mock_gpu_dependencies
    ):
        """Stale cached instances without models are evicted and recreated."""
        from core.model_pool import get_model_pool

        stale = MagicMock()
        stale.models_ready.return_value = True
        fresh = MagicMock()
        fresh.models_ready.return_value = True
        mock_gpu_instance.side_effect = [stale, fresh]

        pool = get_model_pool()
        pool.set_model_path("test.onnx")

        first = pool.get_instance(0)
        assert first is stale

        stale.models_ready.return_value = False
        second = pool.get_instance(0)

        assert second is fresh
        stale.cleanup.assert_called_once()
        assert mock_gpu_instance.call_count == 2

    @patch('core.model_pool.GPUModelInstance')
    def test_get_instances_multiple_gpus(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test get_instances creates instances for multiple GPUs."""
        from core.model_pool import get_model_pool

        # Return different mock for each GPU
        instances = {0: MagicMock(), 1: MagicMock()}
        mock_gpu_instance.side_effect = lambda device_id, **kwargs: instances[device_id]

        pool = get_model_pool()
        pool.set_model_path("test.onnx")

        result = pool.get_instances([0, 1])

        assert len(result) == 2
        assert result[0] is instances[0]
        assert result[1] is instances[1]

    @patch('core.model_pool.GPUModelInstance')
    def test_get_instances_multi_gpu_tensorrt_primary_only(
        self, mock_gpu_instance, mock_gpu_dependencies
    ):
        """Multi-GPU uses TensorRT on primary GPU only to avoid duplicate engine builds."""
        from core.model_pool import get_model_pool

        mock_gpu_instance.side_effect = lambda device_id, **kwargs: MagicMock(device_id=device_id)
        mock_gpu_dependencies["config"].tensorrt_enabled = True

        with patch('core.model_pool.is_tensorrt_runtime_available', return_value=True):
            pool = get_model_pool()
            pool.set_model_path("test.onnx")
            pool.get_instances([0, 1])

        assert mock_gpu_instance.call_args_list[0].kwargs["use_tensorrt"] is True
        assert mock_gpu_instance.call_args_list[1].kwargs["use_tensorrt"] is False

    @patch('core.model_pool.GPUModelInstance')
    def test_is_gpu_initialized(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test is_gpu_initialized returns correct state."""
        from core.model_pool import get_model_pool

        mock_gpu_instance.return_value = MagicMock()

        pool = get_model_pool()
        pool.set_model_path("test.onnx")

        assert not pool.is_gpu_initialized(0)

        pool.get_instance(0)

        assert pool.is_gpu_initialized(0)
        assert not pool.is_gpu_initialized(1)


# =============================================================================
# Concurrent Access Tests
# =============================================================================

class TestModelPoolConcurrency:
    """Tests for concurrent access patterns."""

    @patch('core.model_pool.GPUModelInstance')
    def test_concurrent_get_instance_same_gpu(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test concurrent get_instance for same GPU returns same instance."""
        from core.model_pool import get_model_pool

        mock_instance = MagicMock()
        mock_gpu_instance.return_value = mock_instance

        pool = get_model_pool()
        pool.set_model_path("test.onnx")

        instances = []
        errors = []

        def get_instance():
            try:
                inst = pool.get_instance(0)
                instances.append(inst)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(instances) == 20
        # All should be the same instance
        assert all(i is instances[0] for i in instances)
        # Should only create once due to locking
        assert mock_gpu_instance.call_count == 1

    @patch('core.model_pool.GPUModelInstance')
    def test_concurrent_get_instance_different_gpus(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test concurrent get_instance for different GPUs."""
        from core.model_pool import get_model_pool

        instances = {}
        instance_lock = threading.Lock()

        def create_instance(device_id, **kwargs):
            # Simulate some initialization time
            time.sleep(0.01)
            inst = MagicMock()
            inst.device_id = device_id
            with instance_lock:
                instances[device_id] = inst
            return inst

        mock_gpu_instance.side_effect = create_instance

        pool = get_model_pool()
        pool.set_model_path("test.onnx")

        def get_gpu(device_id):
            return pool.get_instance(device_id)

        # Access GPU 0 and GPU 1 concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(get_gpu, i % 2) for i in range(10)]
            for f in as_completed(futures):
                f.result()

        # Should have created exactly 2 instances (one per GPU)
        assert len(instances) == 2
        assert 0 in instances
        assert 1 in instances


# =============================================================================
# Resource Cleanup Tests
# =============================================================================

class TestModelPoolCleanup:
    """Tests for resource cleanup."""

    @patch('core.model_pool.GPUModelInstance')
    def test_cleanup_single_gpu(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test cleanup releases single GPU instance."""
        from core.model_pool import get_model_pool

        mock_instance = MagicMock()
        mock_gpu_instance.return_value = mock_instance

        pool = get_model_pool()
        pool.set_model_path("test.onnx")
        pool.get_instance(0)

        assert pool.is_gpu_initialized(0)

        pool.cleanup(device_id=0)

        assert not pool.is_gpu_initialized(0)
        mock_instance.cleanup.assert_called_once()

    @patch('core.model_pool.GPUModelInstance')
    def test_cleanup_all_gpus(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test cleanup releases all GPU instances."""
        from core.model_pool import get_model_pool

        instances = {}

        def create_instance(device_id, **kwargs):
            inst = MagicMock()
            inst.device_id = device_id
            instances[device_id] = inst
            return inst

        mock_gpu_instance.side_effect = create_instance

        pool = get_model_pool()
        pool.set_model_path("test.onnx")
        pool.get_instance(0)
        pool.get_instance(1)

        assert pool.is_gpu_initialized(0)
        assert pool.is_gpu_initialized(1)

        pool.cleanup()

        assert not pool.is_gpu_initialized(0)
        assert not pool.is_gpu_initialized(1)
        instances[0].cleanup.assert_called_once()
        instances[1].cleanup.assert_called_once()

    @patch('core.model_pool.GPUModelInstance')
    def test_cleanup_model_pool_function(self, mock_gpu_instance, mock_gpu_dependencies):
        """Test cleanup_model_pool module function."""
        from core.model_pool import get_model_pool, cleanup_model_pool
        import core.model_pool as mp_module

        mock_instance = MagicMock()
        mock_gpu_instance.return_value = mock_instance

        pool = get_model_pool()
        pool.set_model_path("test.onnx")
        pool.get_instance(0)

        cleanup_model_pool()

        assert mp_module.ModelPool._instance is None


# =============================================================================
# GPUModelInstance Locking Tests
# =============================================================================

class TestGPUModelInstanceLocking:
    """Tests for GPUModelInstance thread-safety."""

    def test_acquire_context_manager(self, mock_gpu_dependencies):
        """Test acquire provides exclusive access."""
        from core.model_pool import GPUModelInstance

        with patch('insightface.model_zoo.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()

            instance = GPUModelInstance(
                device_id=0,
                model_path="test.onnx",
                use_tensorrt=False
            )

            acquired = threading.Event()
            released = threading.Event()
            second_acquired = threading.Event()

            def holder():
                with instance.acquire():
                    acquired.set()
                    # Hold the lock for a bit
                    time.sleep(0.1)
                released.set()

            def waiter():
                acquired.wait()  # Wait for first thread to acquire
                with instance.acquire():
                    second_acquired.set()

            t1 = threading.Thread(target=holder)
            t2 = threading.Thread(target=waiter)

            t1.start()
            t2.start()

            # Wait for first acquisition
            acquired.wait(timeout=1)
            assert acquired.is_set()

            # Second thread should not have acquired yet
            time.sleep(0.05)
            assert not second_acquired.is_set()

            # Wait for completion
            t1.join(timeout=2)
            t2.join(timeout=2)

            assert released.is_set()
            assert second_acquired.is_set()

    def test_swap_face_thread_safe(self, mock_gpu_dependencies):
        """Test swap_face method is thread-safe."""
        from core.model_pool import GPUModelInstance

        with patch('insightface.model_zoo.get_model') as mock_get_model:
            mock_swapper = MagicMock()
            mock_swapper.get.return_value = "swapped_frame"
            mock_get_model.return_value = mock_swapper

            instance = GPUModelInstance(
                device_id=0,
                model_path="test.onnx",
                use_tensorrt=False
            )

            results = []
            errors = []

            def do_swap():
                try:
                    result = instance.swap_face(
                        frame="frame",
                        target_face="target",
                        source_face="source"
                    )
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=do_swap) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert len(results) == 10
            assert all(r == "swapped_frame" for r in results)

    def test_get_faces_thread_safe(self, mock_gpu_dependencies):
        """Test get_faces method is thread-safe."""
        from core.model_pool import GPUModelInstance

        with patch('insightface.model_zoo.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()

            instance = GPUModelInstance(
                device_id=0,
                model_path="test.onnx",
                use_tensorrt=False
            )

            # Configure face_app.get to return mock faces
            instance.face_app.get.return_value = [MagicMock(), MagicMock()]

            results = []
            errors = []

            def detect_faces():
                try:
                    faces = instance.get_faces("test_image")
                    results.append(faces)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=detect_faces) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert len(results) == 10
            # Each result should have 2 faces
            assert all(len(r) == 2 for r in results)


class TestSwapFaceBatchGpuPaste:
    """Regression tests for GPU paste-back in swap_face_batch."""

    def test_numpy_pred_bgr_slice_copied_before_torch_from_numpy(self, mock_gpu):
        """RGB->BGR view has negative strides; must copy before torch.from_numpy."""
        import torch
        from core.model_pool import GPUModelInstance

        with patch("core.model_pool.torch") as mock_torch, patch(
            "core.model_pool.ort"
        ) as mock_ort, patch("core.model_pool.FaceAnalysis") as mock_face_analysis, patch(
            "core.model_pool.is_tensorrt_runtime_available", return_value=False
        ), patch(
            "core.model_pool.build_face_analysis_providers",
            return_value=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
        ), patch(
            "core.model_pool.build_swapper_providers",
            return_value=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
        ), patch("core.model_pool.config") as mock_config, patch(
            "insightface.model_zoo.get_model"
        ) as mock_get_model:
            mock_torch.Tensor = torch.Tensor
            mock_torch.from_numpy = torch.from_numpy
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.set_device = MagicMock()
            mock_config.face_analysis_name = "buffalo_l"
            mock_config.face_analysis_det_size = [640, 640]
            mock_config.tensorrt_fp16 = False
            mock_config.get = MagicMock(return_value=2048)
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
            mock_face_analysis.return_value = MagicMock()
            mock_swapper = MagicMock()
            mock_swapper.input_size = (128, 128)
            mock_swapper.input_std = 255.0
            mock_swapper.input_mean = 0.0
            mock_swapper.emap = np.eye(512, dtype=np.float32)
            mock_get_model.return_value = mock_swapper

            instance = GPUModelInstance(
                device_id=0,
                model_path="test.onnx",
                use_tensorrt=False,
            )
            instance.run_swapper_batch = MagicMock(
                return_value=np.ones((1, 3, 128, 128), dtype=np.float32) * 0.5
            )

            target_face = MagicMock()
            target_face.kps = np.zeros((5, 2), dtype=np.float32)
            source_face = MagicMock()
            source_face.normed_embedding = np.ones(512, dtype=np.float32)

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            aimg = np.zeros((128, 128, 3), dtype=np.uint8)
            M = np.eye(2, 3, dtype=np.float32)

            with patch(
                "insightface.utils.face_align.norm_crop2", return_value=(aimg, M)
            ), patch("core.face_paste_gpu.paste_swapped_face_gpu") as mock_paste, patch(
                "utils.config_manager.config"
            ) as cfg:
                cfg.gpu_paste_on_gpu = True
                mock_paste.return_value = torch.zeros(64, 64, 3)

                frame_gpu = torch.zeros(64, 64, 3, dtype=torch.uint8)
                instance.swap_face_batch(
                    image,
                    [target_face],
                    source_face,
                    paste_on_gpu=True,
                    frame_gpu=frame_gpu,
                )

            mock_paste.assert_called_once()
            bgr_fake = mock_paste.call_args[0][1]
            assert isinstance(bgr_fake, torch.Tensor)
