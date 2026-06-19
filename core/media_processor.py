"""
Core media processing class for face detection and swapping.
"""

import cv2
from insightface.utils import face_align
import io
import logging
import numpy as np
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from core.face_paste import ensure_rgb_image, paste_swapped_face
from core.model_pool import get_model_pool
from utils.config_manager import config
from utils.model_optimizer import optimize_onnx_model
from utils.error_handler import ErrorHandler, FriendlyError
from utils.onnx_providers import is_tensorrt_runtime_available

logger = logging.getLogger("FaceOff")

_model_init_lock = threading.Lock()


@contextmanager
def suppress_insightface_output():
    """Context manager to suppress InsightFace/ONNX verbose output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class ModelInitializationError(Exception):
    """Raised when model initialization fails."""

    pass


class FileProcessingError(Exception):
    """Raised when file processing fails."""

    pass


class MediaProcessor:
    """
    Facade over ModelPool for face detection, swapping, and I/O.

    All ONNX sessions are owned by the shared model pool to avoid duplicate
    initialization across UI preview, orchestrator, and multi-GPU paths.
    """

    def __init__(
        self,
        device_id: int = 0,
        use_tensorrt: bool = False,
        optimize_models: bool = False,
        tensorrt_fp16: Optional[bool] = None,
    ) -> None:
        self.device_id = device_id
        self.use_tensorrt = use_tensorrt and is_tensorrt_runtime_available()
        self.tensorrt_fp16 = tensorrt_fp16
        self.optimize_models = optimize_models
        self._validate_environment()
        if optimize_models:
            self._optimize_models()
        self._bind_pool_models()

    def _validate_environment(self) -> None:
        required_paths = {
            "inswapper_model": Path(config.inswapper_model_path),
            "buffalo_model": Path(config.buffalo_model_path),
        }
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing critical resources: {', '.join(missing)}")

    def _optimize_models(self) -> None:
        try:
            inswapper_path = config.inswapper_model_path
            optimized_inswapper = optimize_onnx_model(
                inswapper_path, optimization_level="basic"
            )
            if optimized_inswapper != inswapper_path:
                logger.info("Using optimized inswapper model: %s", optimized_inswapper)

            buffalo_dir = Path(config.buffalo_model_path)
            if buffalo_dir.is_dir():
                for model_file in buffalo_dir.glob("*.onnx"):
                    if "_optimized" not in model_file.stem:
                        optimize_onnx_model(str(model_file), optimization_level="basic")
        except Exception as e:
            logger.warning("Model optimization encountered errors: %s", e)

    def release_gpu_models(self) -> None:
        """Release pooled ONNX models for this device to free VRAM."""
        self.face_app = None
        self.swapper = None
        self._gpu = None
        get_model_pool().cleanup(self.device_id)
        logger.debug("Released pooled models for device %d", self.device_id)

    def _models_bound(self) -> bool:
        """True when pool models are loaded and usable."""
        return self._gpu is not None and self._gpu.models_ready()

    def _ensure_bound(self) -> None:
        """Re-acquire pool models after VRAM release (e.g. post-enhancement cleanup)."""
        if not self._models_bound():
            logger.debug(
                "Re-binding model pool for device %d (prior session released models)",
                self.device_id,
            )
            self._bind_pool_models()

    def _bind_pool_models(self) -> None:
        with _model_init_lock:
            logger.info(
                "Acquiring model initialization lock for device %d", self.device_id
            )
            try:
                pool = get_model_pool()
                pool.set_model_path(config.inswapper_model_path)
                self._gpu = pool.get_instance(
                    self.device_id,
                    use_tensorrt=self.use_tensorrt,
                    tensorrt_fp16=self.tensorrt_fp16,
                )
                self.face_app = self._gpu.face_app
                self.swapper = self._gpu.swapper
                if self.use_tensorrt:
                    logger.info(
                        "TensorRT optimization enabled on device %d", self.device_id
                    )
                else:
                    logger.info(
                        "Face swapper using CUDA only (TensorRT not compatible with inswapper model)"
                    )
                logger.info("MediaProcessor initialized on device %d", self.device_id)
            except FriendlyError:
                raise
            except Exception as e:
                context = {
                    "device_id": self.device_id,
                    "use_tensorrt": self.use_tensorrt,
                    "optimize_models": self.optimize_models,
                }
                raise ErrorHandler.handle_error(e, context) from e
            finally:
                logger.info(
                    "Released model initialization lock for device %d", self.device_id
                )

    def get_faces(self, image: np.ndarray) -> List:
        self._ensure_bound()
        return self._gpu.get_faces(image)

    def get_faces_batch(self, images: List[np.ndarray]) -> List[List]:
        self._ensure_bound()
        return [self._gpu.get_faces(img) for img in images]

    def swap_face(self, image: np.ndarray, target_face, source_face) -> np.ndarray:
        self._ensure_bound()
        return self._gpu.swap_face(image, target_face, source_face, paste_back=True)

    def swap_face_batch(
        self,
        image: np.ndarray,
        target_faces: List,
        source_face,
        *,
        frame_gpu=None,
        paste_on_gpu: bool = False,
    ) -> np.ndarray:
        self._ensure_bound()
        return self._gpu.swap_face_batch(
            image,
            target_faces,
            source_face,
            use_iobinding=config.gpu_frame_retention_enabled,
            frame_gpu=frame_gpu,
            paste_on_gpu=paste_on_gpu,
        )

    def _swap_face_sequential(
        self, image: np.ndarray, target_faces: List, source_face, maps: List
    ) -> np.ndarray:
        swapped = ensure_rgb_image(image)
        src_emb = source_face.normed_embedding.reshape((1, -1))
        src_emb = np.dot(src_emb, self.swapper.emap)
        src_emb /= np.linalg.norm(src_emb)

        for i, (M, aimg) in enumerate(maps):
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / self.swapper.input_std,
                self.swapper.input_size,
                (self.swapper.input_mean,) * 3,
                swapRB=True,
            )
            pred = self.swapper.session.run(
                self.swapper.output_names,
                {
                    self.swapper.input_names[0]: blob.astype(np.float32),
                    self.swapper.input_names[1]: src_emb.astype(np.float32),
                },
            )[0]
            img_fake = pred[0].transpose((1, 2, 0))
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
            swapped = paste_swapped_face(swapped, bgr_fake, aimg, M)

        return swapped

    def swap_faces_batch(
        self,
        images: List[np.ndarray],
        target_faces_list: List[List],
        source_faces_list: List[List],
    ) -> List[np.ndarray]:
        self._ensure_bound()
        results = []
        for img, target_faces, source_faces in zip(
            images, target_faces_list, source_faces_list
        ):
            swapped = img.copy()
            for target_face, source_face in zip(target_faces, source_faces):
                swapped = self._gpu.swap_face(
                    swapped, target_face, source_face, paste_back=True
                )
            results.append(swapped)
        return results

    def read_image(self, file_path: str) -> np.ndarray:
        logger.debug("Reading image: %s", file_path)
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        try:
            image = cv2.imread(str(Path(file_path).resolve()), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image unreadable: {file_path}")
            return image
        except Exception as e:
            raise FileProcessingError("Failed to read image") from e

    def write_image(self, file_path: str, image: np.ndarray) -> str:
        path_obj = Path(file_path)
        unique_filename = f"{path_obj.stem}_{int(time.time())}{path_obj.suffix}"
        resolved_path = path_obj.resolve().parent / unique_filename
        try:
            success = cv2.imwrite(str(resolved_path), image)
            if not success:
                raise FileProcessingError(f"Failed to write image: {resolved_path}")
            return str(resolved_path)
        except Exception as e:
            raise FileProcessingError("Failed to write image") from e

