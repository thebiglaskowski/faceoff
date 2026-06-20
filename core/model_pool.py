"""
Thread-safe model pool for multi-GPU face swapping.

This module provides isolated ONNX sessions per GPU with proper
CUDA context management to prevent model corruption during
concurrent multi-GPU processing.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import onnxruntime as ort
from insightface.app import FaceAnalysis

from utils.config_manager import config
from utils.onnx_providers import (
    build_face_analysis_providers,
    build_swapper_providers,
    is_tensorrt_runtime_available,
    tensorrt_compile_guard,
)

logger = logging.getLogger("FaceOff")

InstanceKey = Tuple[int, bool]


def _trt_warmup_images() -> List[np.ndarray]:
    """
    Synthetic frames at common aspect ratios.

    A plain black 640x640 tensor does not always trigger det_10g TensorRT engine
    builds; real phone portrait/landscape shapes must be exercised at startup.
    """
    det_w, det_h = tuple(config.face_analysis_det_size)
    rng = np.random.default_rng(42)
    shapes = [
        (det_h, det_w),
        (1920, 1080),  # portrait phone — common GIF/photo source
        (1080, 1920),  # landscape
    ]
    images: List[np.ndarray] = []
    for height, width in shapes:
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        noise = rng.integers(0, 32, frame.shape, dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(frame)
    return images


class GPUModelInstance:
    """
    Model instance isolated to a specific GPU.

    Contains all models needed for face swapping on a single GPU:
    - FaceAnalysis for face detection
    - Inswapper for face swapping
    """

    def __init__(
        self,
        device_id: int,
        model_path: str,
        use_tensorrt: bool = False,
        tensorrt_fp16: Optional[bool] = None,
    ):
        self.device_id = device_id
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.tensorrt_fp16 = (
            config.tensorrt_fp16 if tensorrt_fp16 is None else tensorrt_fp16
        )
        self._lock = threading.Lock()

        torch.cuda.set_device(device_id)

        logger.info("Initializing models for GPU %d...", device_id)

        self._init_face_analysis()
        if self.use_tensorrt:
            self.warmup_face_detection()

        swapper_providers = build_swapper_providers(device_id)
        self.swapper = self._load_swapper(model_path, swapper_providers)

        logger.info("GPU %d models initialized successfully", device_id)

    def _init_face_analysis(self) -> None:
        def _create_face_app(use_trt: bool) -> FaceAnalysis:
            face_providers = build_face_analysis_providers(
                self.device_id,
                use_tensorrt=use_trt,
                tensorrt_fp16=self.tensorrt_fp16,
            )
            face_app = FaceAnalysis(
                name=config.face_analysis_name,
                providers=face_providers,
            )
            face_app.prepare(
                ctx_id=self.device_id,
                det_size=tuple(config.face_analysis_det_size),
            )
            return face_app

        if self.use_tensorrt:
            with tensorrt_compile_guard():
                self.face_app = _create_face_app(True)
                if self._det_providers() == ["CPUExecutionProvider"]:
                    logger.warning(
                        "TensorRT failed for face analysis on GPU %d, reinitializing with CUDA...",
                        self.device_id,
                    )
                    self.use_tensorrt = False
                    self.face_app = _create_face_app(False)
                    logger.info(
                        "Face analysis reinitialized with providers: %s",
                        self._det_providers(),
                    )
        else:
            self.face_app = _create_face_app(False)

    def warmup_face_detection(self) -> None:
        """Build/cache TensorRT detection engines for common input shapes."""
        images = _trt_warmup_images()
        logger.info(
            "Warming TensorRT face detection on GPU %d (%d input shapes)...",
            self.device_id,
            len(images),
        )
        with tensorrt_compile_guard():
            torch.cuda.set_device(self.device_id)
            for image in images:
                self.face_app.get(image)
        logger.info("TensorRT face-detection warmup complete on GPU %d", self.device_id)

    def _det_providers(self) -> List[str]:
        if not hasattr(self.face_app, "models"):
            return []
        for key in ("detection", "det_model"):
            det = self.face_app.models.get(key)
            if det is not None and hasattr(det, "session"):
                return det.session.get_providers()
        return []

    def _load_swapper(self, model_path: str, providers: List):
        from insightface.model_zoo import get_model

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        return get_model(model_path, providers=providers, session_options=sess_options)

    @contextmanager
    def acquire(self):
        with self._lock:
            torch.cuda.set_device(self.device_id)
            yield self

    def get_faces(self, image):
        with self._lock:
            if not self.models_ready():
                raise RuntimeError(
                    f"GPU {self.device_id} face models were released; "
                    "re-acquire from ModelPool before detection"
                )
            torch.cuda.set_device(self.device_id)
            return self.face_app.get(image)

    def swap_face(self, frame, target_face, source_face, paste_back: bool = True):
        with self._lock:
            torch.cuda.set_device(self.device_id)
            return self.swapper.get(
                frame, target_face, source_face, paste_back=paste_back
            )

    def run_swapper_batch(
        self,
        blobs: np.ndarray,
        src_stack: np.ndarray,
        *,
        use_iobinding: bool = False,
        return_gpu: bool = False,
    ):
        """
        Run batched inswapper ONNX inference.

        When ``use_iobinding`` is True, inputs are uploaded via DLPack and outputs
        stay on GPU until the final small D2H of swap predictions.
        """
        swapper = self.swapper
        blobs_f32 = blobs.astype(np.float32, copy=False)
        src_f32 = src_stack.astype(np.float32, copy=False)

        with self._lock:
            torch.cuda.set_device(self.device_id)
            if use_iobinding and torch.cuda.is_available():
                try:
                    from torch.utils.dlpack import from_dlpack, to_dlpack

                    blobs_t = torch.from_numpy(blobs_f32).to(
                        f"cuda:{self.device_id}", non_blocking=True
                    )
                    src_t = torch.from_numpy(src_f32).to(
                        f"cuda:{self.device_id}", non_blocking=True
                    )
                    io_binding = swapper.session.io_binding()
                    io_binding.bind_ortvalue_input(
                        swapper.input_names[0],
                        ort.OrtValue.from_dlpack(to_dlpack(blobs_t.contiguous())),
                    )
                    io_binding.bind_ortvalue_input(
                        swapper.input_names[1],
                        ort.OrtValue.from_dlpack(to_dlpack(src_t.contiguous())),
                    )
                    io_binding.bind_output(
                        swapper.output_names[0],
                        "cuda",
                        self.device_id,
                    )
                    swapper.session.run_with_iobinding(io_binding)
                    ort_out = io_binding.get_outputs()[0]
                    out_t = from_dlpack(ort_out.to_dlpack())
                    if return_gpu:
                        return out_t
                    return out_t.detach().cpu().numpy()
                except Exception as exc:
                    logger.debug(
                        "Swapper IoBinding failed on GPU %d (%s) — numpy fallback",
                        self.device_id,
                        exc,
                    )

            return swapper.session.run(
                swapper.output_names,
                {
                    swapper.input_names[0]: blobs_f32,
                    swapper.input_names[1]: src_f32,
                },
            )[0]

    def swap_face_batch(
        self,
        image: np.ndarray,
        target_faces: list,
        source_face,
        *,
        use_iobinding: bool = False,
        frame_gpu: Optional[torch.Tensor] = None,
        paste_on_gpu: bool = False,
    ):
        """Batched multi-face swap with optional GPU IoBinding inference and paste."""
        from insightface.utils import face_align
        import cv2
        from core.face_paste import ensure_rgb_image, paste_swapped_face
        from utils.config_manager import config

        if not target_faces:
            if paste_on_gpu and frame_gpu is not None:
                return frame_gpu
            return image.copy()

        gpu_paste = (
            paste_on_gpu
            and config.gpu_paste_on_gpu
            and frame_gpu is not None
            and torch.cuda.is_available()
        )

        swapper = self.swapper
        crop_size = swapper.input_size[0]
        blobs, maps = [], []

        for face in target_faces:
            aimg, M = face_align.norm_crop2(image, face.kps, crop_size)
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / swapper.input_std,
                swapper.input_size,
                (swapper.input_mean,) * 3,
                swapRB=True,
            )
            blobs.append(blob)
            maps.append((M, aimg))

        blobs_stack = np.concatenate(blobs, axis=0)
        src_emb = source_face.normed_embedding.reshape((1, -1))
        src_emb = np.dot(src_emb, swapper.emap)
        src_emb /= np.linalg.norm(src_emb)
        src_stack = np.repeat(src_emb, len(target_faces), axis=0)

        try:
            pred = self.run_swapper_batch(
                blobs_stack,
                src_stack,
                use_iobinding=use_iobinding and gpu_paste,
                return_gpu=gpu_paste,
            )
        except Exception:
            swapped = ensure_rgb_image(image)
            for face in target_faces:
                swapped = self.swap_face(swapped, face, source_face)
            if gpu_paste:
                frame_gpu.copy_(torch.from_numpy(swapped).to(frame_gpu.device))
                return frame_gpu
            return swapped

        if gpu_paste:
            from core.face_paste_gpu import paste_swapped_face_gpu

            swapped_gpu = frame_gpu
            aimg_t = torch.from_numpy
            for i, (M, aimg) in enumerate(maps):
                if isinstance(pred, torch.Tensor):
                    img_fake = pred[i].permute(1, 2, 0).clamp(0, 1)
                    bgr_fake = (img_fake * 255.0).flip(-1)
                else:
                    img_fake = pred[i].transpose((1, 2, 0))
                    bgr_fake = torch.from_numpy(
                        np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1].copy()
                    ).to(swapped_gpu.device).float()
                aimg_tensor = aimg_t(aimg).to(swapped_gpu.device).float()
                swapped_gpu = paste_swapped_face_gpu(swapped_gpu, bgr_fake, aimg_tensor, M)
            return swapped_gpu

        swapped = ensure_rgb_image(image)
        for i, (M, aimg) in enumerate(maps):
            img_fake = pred[i].transpose((1, 2, 0))
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
            swapped = paste_swapped_face(swapped, bgr_fake, aimg, M)
        return swapped

    def models_ready(self) -> bool:
        """True when detection and swap models are loaded on this GPU."""
        return (
            getattr(self, "face_app", None) is not None
            and getattr(self, "swapper", None) is not None
        )

    def cleanup(self):
        try:
            if hasattr(self, "swapper"):
                del self.swapper
            if hasattr(self, "face_app"):
                del self.face_app
            torch.cuda.set_device(self.device_id)
            torch.cuda.empty_cache()
            logger.debug("Cleaned up GPU %d models", self.device_id)
        except Exception as e:
            logger.warning("Error cleaning up GPU %d: %s", self.device_id, e)


class ModelPool:
    """Thread-safe pool of GPU model instances."""

    _instance: Optional["ModelPool"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    _get_instance_lock = threading.Lock()
    _cleanup_lock = threading.Lock()

    def __init__(self):
        if self._initialized:
            return

        self._gpu_instances: Dict[InstanceKey, GPUModelInstance] = {}
        self._pool_lock = threading.Lock()
        self._model_path: Optional[str] = None
        self._initialized = True

        logger.info("ModelPool initialized")

    @classmethod
    def _get_instance(cls) -> "ModelPool":
        with cls._get_instance_lock:
            if cls._instance is None:
                cls._instance = cls.__new__(cls)
                cls._instance._initialized = False
                cls._instance.__init__()
            return cls._instance

    @classmethod
    def _cleanup(cls) -> None:
        with cls._cleanup_lock:
            if cls._instance is not None:
                cls._instance.cleanup()
                cls._instance._initialized = False
                cls._instance = None

    def set_model_path(self, model_path: str):
        self._model_path = model_path

    def get_instance(
        self,
        device_id: int,
        use_tensorrt: Optional[bool] = None,
        tensorrt_fp16: Optional[bool] = None,
    ) -> GPUModelInstance:
        if use_tensorrt is None:
            use_tensorrt = config.tensorrt_enabled
        use_tensorrt = bool(use_tensorrt and is_tensorrt_runtime_available())
        key: InstanceKey = (device_id, use_tensorrt)

        with self._pool_lock:
            cached = self._gpu_instances.get(key)
            if cached is not None and cached.models_ready():
                return cached
            if cached is not None:
                logger.debug(
                    "Evicting stale model instance for GPU %d (VRAM release)",
                    device_id,
                )
                try:
                    cached.cleanup()
                except Exception as e:
                    logger.warning(
                        "Error cleaning stale GPU %d instance: %s", device_id, e
                    )
                del self._gpu_instances[key]
            model_path = self._model_path or config.inswapper_model_path

        # Heavy ONNX/TRT init must not run under pool_lock (blocks other GPUs).
        instance = GPUModelInstance(
            device_id=device_id,
            model_path=model_path,
            use_tensorrt=use_tensorrt,
            tensorrt_fp16=tensorrt_fp16,
        )

        with self._pool_lock:
            if self._model_path is None:
                self._model_path = model_path
            existing = self._gpu_instances.get(key)
            if existing is not None and existing.models_ready():
                instance.cleanup()
                return existing
            if existing is not None:
                try:
                    existing.cleanup()
                except Exception as e:
                    logger.warning(
                        "Error replacing stale GPU %d instance: %s", device_id, e
                    )
            self._gpu_instances[key] = instance
            return instance

    def get_instances(
        self,
        device_ids: List[int],
        use_tensorrt: Optional[bool] = None,
        tensorrt_fp16: Optional[bool] = None,
    ) -> List[GPUModelInstance]:
        if use_tensorrt is None:
            use_tensorrt = config.tensorrt_enabled
        trt_requested = bool(use_tensorrt and is_tensorrt_runtime_available())

        if len(device_ids) > 1:
            if trt_requested:
                logger.info(
                    "Multi-GPU mode: TensorRT face detection on GPU %d only; "
                    "secondary GPU(s) use CUDA to avoid duplicate engine builds",
                    device_ids[0],
                )
            else:
                logger.info(
                    "Multi-GPU mode: CUDA face detection on %d GPU(s)",
                    len(device_ids),
                )

        instances: List[GPUModelInstance] = []
        for idx, dev_id in enumerate(device_ids):
            trt_for_gpu = trt_requested and (len(device_ids) == 1 or idx == 0)
            instances.append(
                self.get_instance(
                    dev_id,
                    use_tensorrt=trt_for_gpu,
                    tensorrt_fp16=tensorrt_fp16,
                )
            )
        return instances

    @contextmanager
    def acquire_gpu(self, device_id: int, use_tensorrt: Optional[bool] = None):
        instance = self.get_instance(device_id, use_tensorrt=use_tensorrt)
        with instance.acquire():
            yield instance

    def cleanup(self, device_id: Optional[int] = None):
        with self._pool_lock:
            if device_id is not None:
                for key in list(self._gpu_instances.keys()):
                    if key[0] == device_id:
                        self._gpu_instances[key].cleanup()
                        del self._gpu_instances[key]
            else:
                for instance in self._gpu_instances.values():
                    instance.cleanup()
                self._gpu_instances.clear()

        logger.info("ModelPool cleaned up")

    def is_gpu_initialized(self, device_id: int) -> bool:
        return any(key[0] == device_id for key in self._gpu_instances)


get_model_pool = ModelPool._get_instance


def cleanup_model_pool():
    ModelPool._cleanup()