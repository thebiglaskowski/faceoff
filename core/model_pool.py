"""
Thread-safe model pool for multi-GPU face swapping.

This module provides isolated ONNX sessions per GPU with proper
CUDA context management to prevent model corruption during
concurrent multi-GPU processing.
"""

import logging
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import contextmanager

import torch
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

from utils.config_manager import config
from utils.tensorrt_utils import is_tensorrt_available

logger = logging.getLogger("FaceOff")


class GPUModelInstance:
    """
    Model instance isolated to a specific GPU.

    Contains all models needed for face swapping on a single GPU:
    - FaceAnalysis for face detection
    - Inswapper for face swapping
    """

    def __init__(self, device_id: int, model_path: str, use_tensorrt: bool = False):
        """
        Initialize models for a specific GPU.

        Args:
            device_id: CUDA device ID
            model_path: Path to inswapper model
            use_tensorrt: Whether to use TensorRT for face detection
        """
        self.device_id = device_id
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self._lock = threading.Lock()

        # Set CUDA context for this device
        torch.cuda.set_device(device_id)

        # Configure ONNX providers for this specific GPU
        providers = self._get_providers()

        logger.info(f"Initializing models for GPU {device_id}...")

        # Initialize face analysis
        self.face_app = FaceAnalysis(
            name=config.face_analysis_name,
            providers=providers
        )
        self.face_app.prepare(
            ctx_id=device_id,
            det_size=tuple(config.face_analysis_det_size)
        )

        # Initialize inswapper with GPU-specific session
        self.swapper = self._load_swapper(model_path, providers)

        logger.info(f"GPU {device_id} models initialized successfully")

    def _get_providers(self) -> List[tuple]:
        """Get ONNX providers configured for this GPU."""
        providers = []

        # TensorRT provider (if available and enabled)
        if self.use_tensorrt and is_tensorrt_available():
            providers.append(('TensorrtExecutionProvider', {
                'device_id': self.device_id,
                'trt_max_workspace_size': config.tensorrt_workspace_mb * 1024 * 1024,
                'trt_fp16_enable': config.tensorrt_fp16,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(Path("cache") / f"tensorrt_gpu{self.device_id}"),
            }))
            logger.debug(f"TensorRT enabled for GPU {self.device_id}")

        # CUDA provider (primary fallback)
        providers.append(('CUDAExecutionProvider', {
            'device_id': self.device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB limit per session
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
        }))

        # CPU fallback
        providers.append(('CPUExecutionProvider', {}))

        return providers

    def _load_swapper(self, model_path: str, providers: List[tuple]):
        """Load inswapper model with GPU-specific session."""
        from insightface.model_zoo import get_model

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        # Load the model with our specific providers
        swapper = get_model(model_path, providers=providers, session_options=sess_options)
        return swapper

    @contextmanager
    def acquire(self):
        """
        Acquire exclusive access to this GPU's models.

        Usage:
            with gpu_instance.acquire():
                result = gpu_instance.swapper.get(...)
        """
        with self._lock:
            # Set CUDA context before using
            torch.cuda.set_device(self.device_id)
            yield self

    def get_faces(self, image):
        """Detect faces using this GPU's face analysis."""
        with self._lock:
            torch.cuda.set_device(self.device_id)
            return self.face_app.get(image)

    def swap_face(self, frame, target_face, source_face, paste_back: bool = True):
        """Perform face swap using this GPU's swapper."""
        with self._lock:
            torch.cuda.set_device(self.device_id)
            return self.swapper.get(frame, target_face, source_face, paste_back=paste_back)

    def cleanup(self):
        """Release model resources."""
        try:
            del self.swapper
            del self.face_app
            torch.cuda.set_device(self.device_id)
            torch.cuda.empty_cache()
            logger.debug(f"Cleaned up GPU {self.device_id} models")
        except Exception as e:
            logger.warning(f"Error cleaning up GPU {self.device_id}: {e}")


class ModelPool:
    """
    Thread-safe pool of GPU model instances.

    Manages isolated model instances per GPU to enable safe
    concurrent multi-GPU processing.
    """

    _instance: Optional['ModelPool'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._gpu_instances: Dict[int, GPUModelInstance] = {}
        self._pool_lock = threading.Lock()
        self._model_path: Optional[str] = None
        self._use_tensorrt: bool = config.tensorrt_enabled
        self._initialized = True

        logger.info("ModelPool initialized")

    def set_model_path(self, model_path: str):
        """Set the inswapper model path."""
        self._model_path = model_path

    def get_instance(self, device_id: int) -> GPUModelInstance:
        """
        Get or create a model instance for a specific GPU.

        Args:
            device_id: CUDA device ID

        Returns:
            GPUModelInstance for the specified GPU
        """
        with self._pool_lock:
            if device_id not in self._gpu_instances:
                if self._model_path is None:
                    # Use default path from config
                    self._model_path = config.inswapper_model_path

                self._gpu_instances[device_id] = GPUModelInstance(
                    device_id=device_id,
                    model_path=self._model_path,
                    use_tensorrt=self._use_tensorrt
                )

            return self._gpu_instances[device_id]

    def get_instances(self, device_ids: List[int]) -> List[GPUModelInstance]:
        """
        Get model instances for multiple GPUs.

        Args:
            device_ids: List of CUDA device IDs

        Returns:
            List of GPUModelInstance objects
        """
        return [self.get_instance(dev_id) for dev_id in device_ids]

    @contextmanager
    def acquire_gpu(self, device_id: int):
        """
        Acquire exclusive access to a GPU's models.

        Args:
            device_id: CUDA device ID

        Usage:
            with pool.acquire_gpu(0) as gpu:
                result = gpu.swap_face(...)
        """
        instance = self.get_instance(device_id)
        with instance.acquire():
            yield instance

    def cleanup(self, device_id: Optional[int] = None):
        """
        Clean up model instances.

        Args:
            device_id: Specific GPU to clean up, or None for all
        """
        with self._pool_lock:
            if device_id is not None:
                if device_id in self._gpu_instances:
                    self._gpu_instances[device_id].cleanup()
                    del self._gpu_instances[device_id]
            else:
                for instance in self._gpu_instances.values():
                    instance.cleanup()
                self._gpu_instances.clear()

        logger.info("ModelPool cleaned up")

    def is_gpu_initialized(self, device_id: int) -> bool:
        """Check if a GPU instance is initialized."""
        return device_id in self._gpu_instances


# Module-level singleton accessor
_model_pool: Optional[ModelPool] = None


def get_model_pool() -> ModelPool:
    """Get the global ModelPool singleton."""
    global _model_pool
    if _model_pool is None:
        _model_pool = ModelPool()
    return _model_pool


def cleanup_model_pool():
    """Clean up the global model pool."""
    global _model_pool
    if _model_pool is not None:
        _model_pool.cleanup()
        _model_pool = None
