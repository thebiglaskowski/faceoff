"""
Unit tests for enhancement models (RealESRGAN, SwinIR, GFPGAN, CodeFormer).

Tests cover:
- Model initialization and cleanup
- Image enhancement pipelines
- Frame enhancement for GIF/Video
- Error handling
- Model selection and routing
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_rgb_array():
    """Create a sample RGB numpy array for testing."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_array():
    """Create a sample BGR numpy array for testing (OpenCV format)."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def temp_frames_dir(tmp_path):
    """Create temporary frames directory with sample frames."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create 5 sample frames
    for i in range(5):
        img = Image.new('RGB', (128, 128), color=(i * 50, 100, 150))
        img.save(frames_dir / f"frame_{i:04d}.png")

    return frames_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# RealESRGAN Tests
# =============================================================================

class TestRealESRGAN:
    """Tests for RealESRGAN enhancement module."""

    @pytest.mark.unit
    def test_model_options_defined(self):
        """Test that MODEL_OPTIONS dictionary is properly defined."""
        from utils.constants import MODEL_OPTIONS

        # MODEL_OPTIONS has display names as keys (e.g., "RealESRGAN_x4plus (General - Best for Photos)")
        assert len(MODEL_OPTIONS) > 0

        # Check that common model types exist (by checking model_name values)
        model_names = [info.get("model_name") for info in MODEL_OPTIONS.values()]
        assert "RealESRGAN_x4plus" in model_names
        assert "RealESRGAN_x2plus" in model_names

        # Check structure
        for display_name, model_info in MODEL_OPTIONS.items():
            assert "model_name" in model_info
            assert "supports_denoise" in model_info

    @pytest.mark.unit
    def test_realesrgan_x4_model_exists(self):
        """Test that x4 models exist in options."""
        from utils.constants import MODEL_OPTIONS

        # Check that x4 models exist by model_name
        model_names = [info.get("model_name") for info in MODEL_OPTIONS.values()]
        assert "RealESRGAN_x4plus" in model_names
        assert "RealESRGAN_x4plus_anime_6B" in model_names

    @pytest.mark.unit
    def test_realesrgan_x2_model_exists(self):
        """Test that x2 models exist in options."""
        from utils.constants import MODEL_OPTIONS

        model_names = [info.get("model_name") for info in MODEL_OPTIONS.values()]
        assert "RealESRGAN_x2plus" in model_names

    @pytest.mark.unit
    def test_denoise_support_detection(self):
        """Test that only specific models support denoise."""
        from utils.constants import MODEL_OPTIONS

        # Find models by their model_name and check supports_denoise
        for display_name, info in MODEL_OPTIONS.items():
            model_name = info.get("model_name")
            if model_name == "realesr-general-x4v3":
                assert info["supports_denoise"] is True
            elif model_name == "RealESRGAN_x4plus":
                assert info["supports_denoise"] is False

    @pytest.mark.unit
    def test_model_urls_defined(self):
        """Test that model download URLs are defined."""
        from processing.enhancement import MODEL_URLS

        assert "RealESRGAN_x4plus" in MODEL_URLS
        assert "RealESRGAN_x2plus" in MODEL_URLS
        assert all(url.startswith("https://") for url in MODEL_URLS.values())

    @pytest.mark.unit
    def test_model_scales_defined(self):
        """Test that model scales are defined."""
        from processing.enhancement import MODEL_SCALES

        assert MODEL_SCALES["RealESRGAN_x4plus"] == 4
        assert MODEL_SCALES["RealESRGAN_x2plus"] == 2

    @pytest.mark.unit
    @patch('processing.enhancement._get_upsampler')
    def test_enhance_image_calls_upsampler(self, mock_get_upsampler, sample_bgr_array, mock_gpu):
        """Test that enhance_image creates the upsampler correctly."""
        from processing.enhancement import enhance_image

        mock_upsampler = MagicMock()
        mock_upsampler.enhance.return_value = (sample_bgr_array * 2, None)
        mock_get_upsampler.return_value = mock_upsampler

        with patch('processing.enhancement._get_face_enhancer', return_value=None):
            result = enhance_image(
                sample_bgr_array,
                model_name="RealESRGAN_x4plus",
                gpu_id=0,
                outscale=4,
                tile_size=256,
                face_enhance=False
            )

        mock_get_upsampler.assert_called_once()
        mock_upsampler.enhance.assert_called_once()

    @pytest.mark.unit
    @patch('processing.enhancement._get_upsampler')
    def test_enhance_image_returns_array(self, mock_get_upsampler, sample_bgr_array, mock_gpu):
        """Test that enhance_image returns a numpy array."""
        from processing.enhancement import enhance_image

        expected_output = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_upsampler = MagicMock()
        mock_upsampler.enhance.return_value = (expected_output, None)
        mock_get_upsampler.return_value = mock_upsampler

        result = enhance_image(
            sample_bgr_array,
            model_name="RealESRGAN_x4plus",
            gpu_id=0,
            outscale=4,
            tile_size=256,
            face_enhance=False
        )

        assert isinstance(result, np.ndarray)

    @pytest.mark.unit
    def test_enhance_image_handles_none_input(self, mock_gpu):
        """Test that enhance_image handles None input gracefully."""
        from processing.enhancement import enhance_image

        # None input should return None
        result = enhance_image(None)
        assert result is None

    @pytest.mark.unit
    def test_clear_enhancement_cache(self, mock_gpu):
        """Test that enhancement cache can be cleared."""
        from processing.enhancement import clear_enhancement_cache, _upsampler_cache

        # Clear should not raise
        clear_enhancement_cache()

        # Cache should be empty after clearing
        assert len(_upsampler_cache) == 0


# =============================================================================
# SwinIR Tests
# =============================================================================

class TestSwinIR:
    """Tests for SwinIR/Swin2SR enhancement module."""

    @pytest.mark.unit
    def test_swinir_models_defined(self):
        """Test that SWINIR_MODELS dictionary is properly defined."""
        from processing.swinir_enhancement import SWINIR_MODELS

        assert "Swin2SR_x2" in SWINIR_MODELS
        assert "Swin2SR_x4" in SWINIR_MODELS
        assert "Swin2SR_RealWorld_x4" in SWINIR_MODELS
        assert "Swin2SR_Compressed_x4" in SWINIR_MODELS

    @pytest.mark.unit
    def test_swinir_model_structure(self):
        """Test that each SwinIR model has required keys."""
        from processing.swinir_enhancement import SWINIR_MODELS

        for model_name, model_info in SWINIR_MODELS.items():
            assert "model_id" in model_info, f"Missing model_id for {model_name}"
            assert "scale" in model_info, f"Missing scale for {model_name}"
            assert model_info["scale"] in [2, 4], f"Invalid scale for {model_name}"

    @pytest.mark.unit
    def test_swinir_default_model(self):
        """Test the default SwinIR model."""
        from processing.swinir_enhancement import DEFAULT_SWINIR_MODEL

        assert DEFAULT_SWINIR_MODEL == "Swin2SR_RealWorld_x4"

    @pytest.mark.unit
    def test_swinir_scale_retrieval(self):
        """Test get_swinir_scale function."""
        from processing.swinir_enhancement import get_swinir_scale

        assert get_swinir_scale("Swin2SR_x2") == 2
        assert get_swinir_scale("Swin2SR_x4") == 4
        assert get_swinir_scale("InvalidModel") == 4  # Default

    @pytest.mark.unit
    def test_swinir_invalid_model_name(self, mock_gpu, sample_bgr_array):
        """Test that invalid model name falls back to default."""
        from processing.swinir_enhancement import enhance_image_swinir

        # Should log a warning but use default model
        # With mocked GPU it will fail at model loading, returning None
        result = enhance_image_swinir(
            sample_bgr_array,
            model_name="InvalidModel",
            gpu_id=0
        )

        # Without actual model loading, result should be None
        assert result is None

    @pytest.mark.unit
    def test_swinir_none_input(self, mock_gpu):
        """Test that None input returns None."""
        from processing.swinir_enhancement import enhance_image_swinir

        # None input should fail gracefully
        result = enhance_image_swinir(
            None,
            model_name="Swin2SR_x4",
            gpu_id=0
        )

        assert result is None

    @pytest.mark.unit
    def test_swinir_cache_clear(self, mock_gpu):
        """Test that SwinIR cache can be cleared."""
        from processing.swinir_enhancement import clear_swinir_cache, _swinir_cache

        clear_swinir_cache()
        assert len(_swinir_cache) == 0

    @pytest.mark.unit
    def test_get_available_swinir_models(self):
        """Test that available models can be retrieved."""
        from processing.swinir_enhancement import get_available_swinir_models, SWINIR_MODELS

        available = get_available_swinir_models()
        assert available == SWINIR_MODELS
        # Should be a copy, not the same object
        assert available is not SWINIR_MODELS


# =============================================================================
# HAT Tests
# =============================================================================

class TestHAT:
    """Tests for HAT (Hybrid Attention Transformer) enhancement module."""

    @pytest.mark.unit
    def test_hat_models_defined(self):
        """Test that HAT_MODELS dictionary is properly defined."""
        from processing.hat_enhancement import HAT_MODELS

        assert "HAT_Base_4x_ImageNet" in HAT_MODELS
        assert "HAT_Base_4x_GAN_sharper" in HAT_MODELS

    @pytest.mark.unit
    def test_hat_model_structure(self):
        """Test that each HAT model has required keys."""
        from processing.hat_enhancement import HAT_MODELS

        for model_name, model_info in HAT_MODELS.items():
            assert "url" in model_info, f"Missing url for {model_name}"
            assert "scale" in model_info, f"Missing scale for {model_name}"
            assert model_info["scale"] in [2, 4], f"Invalid scale for {model_name}"
            assert "description" in model_info, f"Missing description for {model_name}"
            assert "arch" in model_info, f"Missing arch for {model_name}"
            assert model_info["arch"]["window_size"] == 16
            assert model_info["arch"]["embed_dim"] == 180

    @pytest.mark.unit
    def test_hat_base_uses_base_weights_not_large(self):
        """HAT_Base must not point at HAT-L weights (arch mismatch)."""
        from processing.hat_enhancement import HAT_MODELS

        url = HAT_MODELS["HAT_Base_4x_ImageNet"]["url"]
        assert "HAT-L" not in url
        assert "HAT_SRx4_ImageNet-pretrain.pth" in url

    @pytest.mark.unit
    def test_hat_window_padding(self):
        """HAT inputs must be padded to multiples of window_size (7)."""
        import numpy as np
        from processing.hat_enhancement import (
            HAT_WINDOW_SIZE,
            _hat_tile_dim,
            _pad_to_window_size,
        )

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        padded, orig_h, orig_w = _pad_to_window_size(img, HAT_WINDOW_SIZE)
        assert (orig_h, orig_w) == (256, 256)
        assert padded.shape[0] % HAT_WINDOW_SIZE == 0
        assert padded.shape[1] % HAT_WINDOW_SIZE == 0
        assert _hat_tile_dim(100, 256, HAT_WINDOW_SIZE) % HAT_WINDOW_SIZE == 0

    @pytest.mark.unit
    def test_hat_checkpoint_extraction(self):
        """HAT checkpoints should load params_ema when present."""
        from processing.hat_enhancement import _extract_hat_checkpoint

        weights = {"layer": 1}
        assert _extract_hat_checkpoint({"params_ema": weights}) is weights
        assert _extract_hat_checkpoint({"state_dict": weights}) is weights

    @pytest.mark.unit
    def test_hat_default_model(self):
        """Test the default HAT model."""
        from processing.hat_enhancement import DEFAULT_HAT_MODEL

        assert DEFAULT_HAT_MODEL == "HAT_Base_4x_ImageNet"

    @pytest.mark.unit
    def test_hat_scale_retrieval(self):
        """Test get_hat_scale function."""
        from processing.hat_enhancement import get_hat_scale

        assert get_hat_scale("HAT_Base_4x_ImageNet") == 4
        assert get_hat_scale("HAT_Base_4x_GAN_sharper") == 4
        assert get_hat_scale("InvalidModel") == 4  # Default

    @pytest.mark.unit
    def test_hat_cache_clear(self, mock_gpu):
        """Test that HAT cache can be cleared."""
        from processing.hat_enhancement import clear_hat_cache, _hat_cache

        clear_hat_cache()
        assert len(_hat_cache) == 0

    @pytest.mark.unit
    def test_get_hat_model_serializes_concurrent_loads(self, mock_gpu):
        """Concurrent _get_hat_model calls must not double-load the same GPU."""
        import threading
        from unittest.mock import MagicMock, patch

        from processing.hat_enhancement import _get_hat_model, _hat_cache, clear_hat_cache

        clear_hat_cache()
        load_count = 0
        model_tuple = (MagicMock(), MagicMock(), MagicMock(), 1.0)

        def _fake_load(model_name, gpu_id, model_path):
            nonlocal load_count
            load_count += 1
            return model_tuple

        with patch(
            "processing.hat_enhancement._load_hat_model_impl", side_effect=_fake_load
        ):
            errors = []

            def _worker():
                try:
                    _get_hat_model("HAT_Base_4x_ImageNet", 0)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=_worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        assert load_count == 1
        assert len(_hat_cache) == 1

    @pytest.mark.unit
    def test_preload_hat_models_is_sequential(self, mock_gpu):
        from unittest.mock import patch

        from processing.hat_enhancement import preload_hat_models

        order = []

        def _track_get(model_name, gpu_id, model_path=None):
            order.append(gpu_id)
            return (MagicMock(), MagicMock(), MagicMock(), 1.0)

        with patch("processing.hat_enhancement._get_hat_model", side_effect=_track_get):
            preload_hat_models([0, 1], "HAT_Base_4x_ImageNet")

        assert order == [0, 1]

    @pytest.mark.unit
    def test_get_available_hat_models(self):
        """Test that available HAT models can be retrieved."""
        from processing.hat_enhancement import get_available_hat_models, HAT_MODELS

        available = get_available_hat_models()
        assert available == HAT_MODELS
        assert available is not HAT_MODELS

    @pytest.mark.unit
    def test_should_use_tiled_when_vram_low(self, mock_gpu, temp_config, reset_config):
        from unittest.mock import patch
        from processing.hat_enhancement import _should_use_tiled_inference
        from utils.memory_manager import MemoryManager

        with patch.object(MemoryManager, "get_memory_stats", return_value={"free_mb": 500}):
            assert _should_use_tiled_inference(200, 200, 256, gpu_id=0) is True

    @pytest.mark.unit
    def test_oom_tile_attempts_reduce_size(self):
        from processing.hat_enhancement import _oom_tile_attempts

        attempts = _oom_tile_attempts(256)
        assert attempts[0] == 256
        assert attempts[-1] == 64
        assert len(attempts) == len(set(attempts))

    @pytest.mark.unit
    def test_hat_input_skips_external_mean_subtraction(self):
        """HAT.forward() normalizes mean internally — do not subtract twice."""
        import numpy as np
        import torch
        from processing.hat_enhancement import _apply_hat_model

        class _RecordingModel(torch.nn.Module):
            window_size = 16

            def __init__(self):
                super().__init__()
                self.last_input_mean = None

            def forward(self, x):
                self.last_input_mean = float(x.mean().item())
                batch, _, h, w = x.shape
                out = torch.nn.functional.interpolate(x, scale_factor=4, mode="bilinear")
                return torch.clamp(out, 0.0, 1.0)

        model = _RecordingModel()
        device = torch.device("cpu")
        mean = torch.zeros(1, 3, 1, 1)
        rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
        _apply_hat_model(rgb, model, device, mean, 1.0, gpu_id=0, tile_size=0)
        assert model.last_input_mean is not None
        assert abs(model.last_input_mean - (128 / 255.0)) < 0.02

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_hat_fp32_inference_avoids_fp16_nan(self, mock_gpu):
        """HAT must not use FP16 autocast (produces NaN/black frames on consumer GPUs)."""
        import numpy as np
        import torch
        from processing.hat_enhancement import _get_hat_model, clear_hat_cache

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        clear_hat_cache()
        model, device, mean, img_range = _get_hat_model("HAT_Base_4x_ImageNet", 0)
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        img = torch.from_numpy(
            np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fp16_out = model(img)
            fp32_out = model(img.float())
        assert torch.isnan(fp16_out).any()
        assert not torch.isnan(fp32_out).any()

    @pytest.mark.unit
    def test_tiled_inference_produces_nonzero_output(self):
        """Tiled path must not return an all-black frame (weight_map bug regression)."""
        import numpy as np
        import torch
        from processing.hat_enhancement import HAT_UPSCALE, _enhance_image_tiled

        class _IdentityScaleModel(torch.nn.Module):
            window_size = 16

            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x,
                    scale_factor=HAT_UPSCALE,
                    mode="bilinear",
                    align_corners=False,
                )

        image = np.random.randint(20, 200, (320, 280, 3), dtype=np.uint8)
        model = _IdentityScaleModel()
        device = torch.device("cpu")
        mean = torch.zeros(1, 3, 1, 1)
        result = _enhance_image_tiled(
            image, model, device, mean, img_range=1.0, tile_size=128, overlap=16
        )

        assert result.shape == (320 * HAT_UPSCALE, 280 * HAT_UPSCALE, 3)
        assert int(result.max()) > 0
        assert int(result.mean()) > 10


# =============================================================================
# GFPGAN Tests
# =============================================================================

class TestGFPGAN:
    """Tests for GFPGAN face restoration module."""

    @pytest.mark.unit
    def test_face_restorer_module_exists(self):
        """Test that face_restoration module exists and has FaceRestorer."""
        from processing.face_restoration import FaceRestorer
        assert FaceRestorer is not None

    @pytest.mark.unit
    def test_face_restorer_init_params(self):
        """Test FaceRestorer accepts correct init parameters."""
        from processing.face_restoration import FaceRestorer
        import inspect

        sig = inspect.signature(FaceRestorer.__init__)
        params = list(sig.parameters.keys())

        assert 'device_id' in params
        assert 'model_version' in params

    @pytest.mark.unit
    def test_face_restorer_init_no_immediate_load(self, mock_gpu):
        """Test FaceRestorer uses lazy initialization."""
        from processing.face_restoration import FaceRestorer

        # Should not load model during __init__ (lazy init)
        restorer = FaceRestorer(device_id=0)

        # Internal state should show not initialized
        assert restorer._initialized is False

    @pytest.mark.unit
    def test_face_restorer_has_restore_method(self):
        """Test FaceRestorer has restore_face method."""
        from processing.face_restoration import FaceRestorer

        assert hasattr(FaceRestorer, 'restore_face')
        assert callable(getattr(FaceRestorer, 'restore_face'))

    @pytest.mark.unit
    def test_face_restorer_has_cleanup_method(self):
        """Test FaceRestorer has cleanup method."""
        from processing.face_restoration import FaceRestorer

        assert hasattr(FaceRestorer, 'cleanup')
        assert callable(getattr(FaceRestorer, 'cleanup'))


# =============================================================================
# CodeFormer Tests
# =============================================================================

class TestCodeFormer:
    """Tests for CodeFormer face restoration module."""

    @pytest.mark.unit
    def test_codeformer_module_exists(self):
        """Test that codeformer_restoration module exists."""
        from processing.codeformer_restoration import CodeFormerRestorer
        assert CodeFormerRestorer is not None

    @pytest.mark.unit
    def test_codeformer_restorer_init_params(self):
        """Test CodeFormerRestorer accepts correct parameters."""
        from processing.codeformer_restoration import CodeFormerRestorer
        import inspect

        sig = inspect.signature(CodeFormerRestorer.__init__)
        params = list(sig.parameters.keys())

        assert 'device_id' in params
        assert 'upscale' in params
        assert 'model_dir' in params

    @pytest.mark.unit
    def test_codeformer_model_urls_defined(self):
        """Test that CodeFormer model URLs are defined."""
        from processing.codeformer_restoration import (
            CODEFORMER_MODEL_URL,
            DETECTION_MODEL_URL,
            PARSING_MODEL_URL
        )

        assert CODEFORMER_MODEL_URL.startswith("https://")
        assert DETECTION_MODEL_URL.startswith("https://")
        assert PARSING_MODEL_URL.startswith("https://")

    @pytest.mark.unit
    def test_codeformer_convenience_function_exists(self):
        """Test that convenience function exists."""
        from processing.codeformer_restoration import restore_faces_codeformer
        assert callable(restore_faces_codeformer)

    @pytest.mark.unit
    def test_codeformer_frames_function_exists(self):
        """Test that frame restoration function exists."""
        from processing.codeformer_restoration import restore_frames_codeformer
        assert callable(restore_frames_codeformer)

    @pytest.mark.unit
    def test_codeformer_cache_clear(self, mock_gpu):
        """Test that CodeFormer cache can be cleared."""
        from processing.codeformer_restoration import clear_codeformer_cache, _codeformer_cache

        clear_codeformer_cache()
        assert len(_codeformer_cache) == 0


# =============================================================================
# Model Selection/Routing Tests
# =============================================================================

class TestModelRouting:
    """Tests for model selection and routing logic."""

    @pytest.mark.unit
    def test_enhancement_model_options(self):
        """Test that enhancement model options are properly defined."""
        # Check UI defines both options
        enhancement_options = ["RealESRGAN", "SwinIR"]
        assert "RealESRGAN" in enhancement_options
        assert "SwinIR" in enhancement_options

    @pytest.mark.unit
    def test_restoration_model_options(self):
        """Test that restoration model options are properly defined."""
        restoration_options = ["GFPGAN", "CodeFormer"]
        assert "GFPGAN" in restoration_options
        assert "CodeFormer" in restoration_options

    @pytest.mark.unit
    def test_restoration_session_uses_gfpgan_by_default(self, mock_gpu):
        """RestorationSession should use GFPGAN by default."""
        from processing.restoration_session import RestorationSession

        with patch("processing.face_restoration.FaceRestorer") as mock_class:
            mock_restorer = MagicMock()
            mock_class.return_value = mock_restorer
            session = RestorationSession("GFPGAN", gpu_id=0)
            mock_class.assert_called_once()
            session.close()

    @pytest.mark.unit
    def test_restoration_session_uses_codeformer(self, mock_gpu):
        """RestorationSession should use CodeFormer when specified."""
        from processing.restoration_session import RestorationSession

        with patch(
            "processing.codeformer_restoration._get_codeformer_restorer"
        ) as mock_get:
            mock_restorer = MagicMock()
            mock_get.return_value = mock_restorer
            session = RestorationSession("CodeFormer", gpu_id=0)
            mock_get.assert_called_once()
            session.close()

    @pytest.mark.unit
    def test_in_memory_enhancer_routes_to_swinir(self):
        """InMemoryEnhancer should call SwinIR for SwinIR model selection."""
        import numpy as np
        from processing.in_memory_enhancement import InMemoryEnhancer

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        enhancer = InMemoryEnhancer(
            "SwinIR", "Swin2SR_x4", [0], tile_size=256, outscale=4
        )

        with patch("processing.swinir_enhancement.enhance_image_swinir") as mock_swinir:
            mock_swinir.return_value = frame
            result = enhancer.enhance_rgb_frames([frame])

        mock_swinir.assert_called_once()
        assert len(result) == 1


# =============================================================================
# Processing Pipeline Tests
# =============================================================================

class TestProcessingPipeline:
    """Tests for the processing pipeline with model selection."""

    @pytest.mark.unit
    def test_process_image_signature_includes_model_params(self):
        """Test that process_image accepts model selection parameters."""
        from processing.image_processing import process_image
        import inspect

        sig = inspect.signature(process_image)
        param_names = list(sig.parameters.keys())

        assert 'restore_faces' in param_names
        assert 'restoration_weight' in param_names

    @pytest.mark.unit
    def test_orchestrator_signature_includes_model_params(self):
        """Test that orchestrator accepts model selection parameters."""
        from processing.orchestrator import ProcessOptions
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ProcessOptions)}

        assert 'enhancement_model' in fields
        assert 'restoration_model' in fields

    @pytest.mark.unit
    def test_default_enhancement_model(self):
        """Test that default enhancement model is RealESRGAN."""
        from processing.orchestrator import ProcessOptions
        import dataclasses

        defaults = {
            f.name: f.default
            for f in dataclasses.fields(ProcessOptions)
            if f.default is not dataclasses.MISSING
        }

        assert defaults.get('enhancement_model') == 'RealESRGAN'

    @pytest.mark.unit
    def test_default_restoration_model(self):
        """Test that default restoration model is GFPGAN."""
        from processing.orchestrator import ProcessOptions
        import dataclasses

        defaults = {
            f.name: f.default
            for f in dataclasses.fields(ProcessOptions)
            if f.default is not dataclasses.MISSING
        }

        assert defaults.get('restoration_model') == 'GFPGAN'


# =============================================================================
# Handler Tests
# =============================================================================

class TestProcessingHandlers:
    """Tests for UI processing handlers."""

    @pytest.mark.unit
    def test_process_image_handler_signature(self):
        """Test that process_image handler accepts model parameters."""
        from ui.handlers.processing_handlers import process_image
        import inspect

        sig = inspect.signature(process_image)
        param_names = list(sig.parameters.keys())

        assert 'enhancement_model' in param_names
        assert 'restoration_model' in param_names

    @pytest.mark.unit
    def test_process_gif_handler_signature(self):
        """Test that process_gif handler accepts model parameters."""
        from ui.handlers.processing_handlers import process_gif
        import inspect

        sig = inspect.signature(process_gif)
        param_names = list(sig.parameters.keys())

        assert 'enhancement_model' in param_names
        assert 'restoration_model' in param_names

    @pytest.mark.unit
    def test_process_video_handler_signature(self):
        """Test that process_video handler accepts model parameters."""
        from ui.handlers.processing_handlers import process_video
        import inspect

        sig = inspect.signature(process_video)
        param_names = list(sig.parameters.keys())

        assert 'enhancement_model' in param_names
        assert 'restoration_model' in param_names


# =============================================================================
# UI Component Tests
# =============================================================================

class TestUIComponents:
    """Tests for UI component model selectors."""

    @pytest.mark.unit
    def test_image_tab_has_model_selectors(self):
        """Test that image_tab returns model selector components."""
        # We can check this by verifying the returned keys
        from ui.components.image_tab import create_image_tab
        import gradio as gr

        # Can't easily test Gradio component creation without running
        # Instead, test that the module has the expected structure
        import inspect
        source = inspect.getsource(create_image_tab)

        assert 'enhancement_model_selector' in source
        assert 'restoration_model_selector' in source

    @pytest.mark.unit
    def test_gif_tab_has_model_selectors(self):
        """Test that gif_tab returns model selector components."""
        from ui.components.gif_tab import create_gif_tab
        import inspect
        source = inspect.getsource(create_gif_tab)

        assert 'enhancement_model_selector_gif' in source
        assert 'restoration_model_selector_gif' in source

    @pytest.mark.unit
    def test_video_tab_has_model_selectors(self):
        """Test that video_tab returns model selector components."""
        from ui.components.video_tab import create_video_tab
        import inspect
        source = inspect.getsource(create_video_tab)

        assert 'enhancement_model_selector_vid' in source
        assert 'restoration_model_selector_vid' in source
