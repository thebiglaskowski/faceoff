"""
Core media processing class for face detection and swapping.
"""

import cv2
from insightface.utils import face_align
import insightface
import io
import logging
import numpy as np
import os
import sys
import threading
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

from utils.config_manager import config
from processing.model_optimizer import optimize_onnx_model
from utils.error_handler import ErrorHandler, FriendlyError
from utils.tensorrt_utils import is_tensorrt_available
from utils import video_io

logger = logging.getLogger("FaceOff")

# Global lock to prevent concurrent model initialization
# ONNX Runtime model loading is not thread-safe when multiple threads
# load the same model file simultaneously
_model_init_lock = threading.Lock()


@contextmanager
def suppress_insightface_output():
    """Context manager to suppress InsightFace/ONNX verbose output."""
    # Redirect to StringIO instead of closing file descriptors
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        yield
    finally:
        # Restore original streams
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
    Handles media processing with CUDA acceleration.

    This class manages:
    - Face detection and analysis
    - Face swapping model (inswapper)
    - Image reading/writing
    - GIF frame extraction
    """

    def __init__(
        self,
        device_id: int = 0,
        use_tensorrt: bool = False,
        optimize_models: bool = True,
    ) -> None:
        """
        Initialize MediaProcessor on specified GPU.

        Args:
            device_id: CUDA device ID to use
            use_tensorrt: Whether to use TensorRT optimization (faster inference)
            optimize_models: Whether to optimize ONNX models before loading
        """
        self.device_id = device_id
        self.use_tensorrt = use_tensorrt
        self.optimize_models = optimize_models
        self._validate_environment()
        self._optimize_models()
        self._initialize_models()

    def _validate_environment(self) -> None:
        """Validate required model files exist."""
        required_paths = {
            "inswapper_model": Path(config.inswapper_model_path),
            "buffalo_model": Path(config.buffalo_model_path),
        }

        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing critical resources: {', '.join(missing)}")

    def _optimize_models(self) -> None:
        """Optimize ONNX models before loading."""
        if not self.optimize_models:
            logger.debug("Model optimization disabled")
            return

        try:
            # Optimize inswapper model (use basic optimization for safety)
            inswapper_path = config.inswapper_model_path
            optimized_inswapper = optimize_onnx_model(
                inswapper_path, optimization_level="basic"
            )

            # Note: We don't update the config path since we're using the global config now
            if optimized_inswapper != inswapper_path:
                logger.info("Using optimized inswapper model: %s", optimized_inswapper)

            # Optimize buffalo models (use basic optimization for safety)
            buffalo_dir = Path(config.buffalo_model_path)
            if buffalo_dir.is_dir():
                for model_file in buffalo_dir.glob("*.onnx"):
                    if "_optimized" not in model_file.stem:
                        optimize_onnx_model(str(model_file), optimization_level="basic")

        except Exception as e:
            logger.warning("Model optimization encountered errors: %s", e)
            logger.info("Continuing with original models")

    def _initialize_models(self) -> None:
        """Initialize face analysis and swapper models with CUDA."""
        # Use global lock to prevent concurrent model initialization
        # ONNX Runtime is not thread-safe during model loading
        with _model_init_lock:
            logger.info(
                "Acquiring model initialization lock for device %d", self.device_id
            )
            try:
                # Configure CUDA provider for specific device
                # Use 'kSameAsRequested' to reduce memory fragmentation
                # (kNextPowerOfTwo can cause BFC arena allocation failures)
                gpu_mem_limit_mb = config.get("gpu", "onnx_mem_limit_mb", default=2048)
                cuda_provider_options = {
                    "device_id": self.device_id,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": gpu_mem_limit_mb * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }

                # Build provider list with optional TensorRT
                if self.use_tensorrt and is_tensorrt_available():
                    # Use GPU-specific cache path to prevent race conditions
                    # when multiple GPUs compile TensorRT engines simultaneously
                    gpu_cache_path = (
                        Path(config.tensorrt_cache_dir) / f"gpu_{self.device_id}"
                    )
                    gpu_cache_path.mkdir(parents=True, exist_ok=True)
                    tensorrt_provider_options = {
                        "device_id": self.device_id,
                        "trt_max_workspace_size": config.tensorrt_workspace_mb
                        * 1024
                        * 1024,
                        "trt_fp16_enable": config.tensorrt_fp16,  # FP16 for ~30% speedup
                        "trt_engine_cache_enable": True,  # Cache TensorRT engines
                        "trt_engine_cache_path": str(gpu_cache_path),
                    }
                    providers = [
                        ("TensorrtExecutionProvider", tensorrt_provider_options),
                        ("CUDAExecutionProvider", cuda_provider_options),
                        "CPUExecutionProvider",
                    ]
                    fp_mode = "FP16" if config.tensorrt_fp16 else "FP32"
                    logger.info(
                        "TensorRT optimization enabled on device %d (%s mode)",
                        self.device_id,
                        fp_mode,
                    )
                else:
                    if self.use_tensorrt:
                        logger.debug("TensorRT requested but not available, using CUDA")
                        self.use_tensorrt = False
                    providers = [
                        ("CUDAExecutionProvider", cuda_provider_options),
                        "CPUExecutionProvider",
                    ]

                # Initialize face analysis - suppress verbose ONNX output
                # Note: TensorRT can fail for some models (Myelin autotuning errors)
                # If it fails and falls back to CPU, we reinitialize with CUDA only
                logger.debug("Loading face analysis models (InsightFace)...")
                cuda_only_providers = [
                    ("CUDAExecutionProvider", cuda_provider_options),
                    "CPUExecutionProvider",
                ]

                with suppress_insightface_output():
                    self.face_app = insightface.app.FaceAnalysis(
                        name=config.face_analysis_name,
                        root=".",
                        providers=providers,
                    )
                    self.face_app.prepare(
                        ctx_id=self.device_id,
                        det_size=tuple(config.face_analysis_det_size),
                    )

                # Check actual providers - if TensorRT failed and fell back to CPU, reinitialize with CUDA
                actual_providers = []
                if (
                    hasattr(self.face_app, "models")
                    and "det_model" in self.face_app.models
                ):
                    det_model = self.face_app.models["det_model"]
                    if hasattr(det_model, "session"):
                        actual_providers = det_model.session.get_providers()
                        logger.debug(
                            "Face analysis using providers: %s", actual_providers
                        )

                # If we wanted TensorRT but ended up with only CPU, reinitialize with CUDA
                if self.use_tensorrt and actual_providers == ["CPUExecutionProvider"]:
                    logger.warning(
                        "TensorRT failed for face analysis, reinitializing with CUDA..."
                    )
                    with suppress_insightface_output():
                        self.face_app = insightface.app.FaceAnalysis(
                            name=config.face_analysis_name,
                            root=".",
                            providers=cuda_only_providers,
                        )
                        self.face_app.prepare(
                            ctx_id=self.device_id,
                            det_size=tuple(config.face_analysis_det_size),
                        )
                    # Log the new providers
                    if (
                        hasattr(self.face_app, "models")
                        and "det_model" in self.face_app.models
                    ):
                        det_model = self.face_app.models["det_model"]
                        if hasattr(det_model, "session"):
                            actual_providers = det_model.session.get_providers()
                            logger.info(
                                "Face analysis reinitialized with providers: %s",
                                actual_providers,
                            )

                # Initialize face swapper (CUDA only - TensorRT has compilation issues with this model)
                logger.debug("Loading face swapper model (inswapper)...")
                swapper_providers = [
                    ("CUDAExecutionProvider", cuda_provider_options),
                    "CPUExecutionProvider",
                ]
                with suppress_insightface_output():
                    self.swapper = insightface.model_zoo.get_model(
                        config.inswapper_model_path,
                        download=False,
                        download_zip=False,
                        providers=swapper_providers,
                    )
                logger.info(
                    "Face swapper using CUDA only (TensorRT not compatible with inswapper model)"
                )

                logger.info("MediaProcessor initialized on device %d", self.device_id)
                logger.info(
                    "Released model initialization lock for device %d", self.device_id
                )

            except FriendlyError:
                # Re-raise friendly errors
                raise
            except Exception as e:
                logger.error("Model initialization failed: %s", e)
                # Convert to friendly error
                context = {
                    "device_id": self.device_id,
                    "use_tensorrt": self.use_tensorrt,
                    "optimize_models": self.optimize_models,
                }
                friendly_error = ErrorHandler.handle_error(e, context)
                raise friendly_error

    def get_faces(self, image: np.ndarray) -> List:
        """
        Detect and analyze faces in image.

        Args:
            image: Input image as numpy array

        Returns:
            List of detected faces with metadata
        """
        return self.face_app.get(image)

    def get_faces_batch(self, images: List[np.ndarray]) -> List[List]:
        """
        Detect and analyze faces in multiple images (batched).

        Args:
            images: List of input images as numpy arrays

        Returns:
            List of face lists (one per image)
        """
        return [self.face_app.get(img) for img in images]

    def swap_face(self, image: np.ndarray, target_face, source_face) -> np.ndarray:
        """
        Swap a single face in an image.

        Args:
            image: Input image
            target_face: Target face to replace
            source_face: Source face to use

        Returns:
            Image with swapped face
        """
        return self.swapper.get(image, target_face, source_face, paste_back=True)

    def swap_face_batch(
        self, image: np.ndarray, target_faces: List, source_face
    ) -> np.ndarray:
        """
        Batch ONNX inference for all target faces sharing the same source face,
        then paste_back sequentially. Reduces per-frame GPU overhead when a frame
        has multiple faces — instead of N separate ONNX calls (N = face count),
        we do one batched call with batch_size=N and N paste_back ops on CPU.

        The inswapper model accepts:
          target: [B, 3, 128, 128]  (cropped face crops)
          source: [B, 512]         (source embeddings)
        Output:  [B, 3, 128, 128] (face crops with swapped content)

        paste_back is CPU-bound (warpAffine + mask blend) and can't be batched,
        but ONNX inference is GPU-bound — batching gives a 2-5x speedup when
        there are 3+ detected faces per frame.
        """
        if not target_faces:
            return image.copy()

        # Preprocess all target faces: crop + blob
        crop_size = self.swapper.input_size[0]
        blobs = []
        maps = []  # (M, aimg) per face for paste_back

        for face in target_faces:
            aimg, M = face_align.norm_crop2(image, face.kps, crop_size)
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / self.swapper.input_std,
                self.swapper.input_size,
                (self.swapper.input_mean,) * 3,
                swapRB=True,
            )
            blobs.append(blob)
            maps.append((M, aimg))

        blobs_stack = np.concatenate(blobs, axis=0)  # [N, 3, 128, 128]
        src_emb = source_face.normed_embedding.reshape((1, -1))  # [1, 512]
        src_emb = np.dot(src_emb, self.swapper.emap)
        src_emb /= np.linalg.norm(src_emb)
        src_stack = np.repeat(src_emb, len(target_faces), axis=0)  # [N, 512]

        # Single batched ONNX inference
        pred = self.swapper.session.run(
            self.swapper.output_names,
            {
                self.swapper.input_names[0]: blobs_stack.astype(np.float32),
                self.swapper.input_names[1]: src_stack.astype(np.float32),
            },
        )[0]  # [N, 3, 128, 128]

        # Paste each result back sequentially (CPU-bound warpAffine).
        # We modify `image` in-place to preserve previous swaps in the frame (e.g. multi-face).
        swapped = image.copy()
        # Ensure 3D image for arithmetic - unconditionally force 3 channels
        if swapped.ndim == 2:
            swapped = np.stack([swapped] * 3, axis=-1)
        elif swapped.ndim == 3 and swapped.shape[2] == 1:
            swapped = np.squeeze(swapped, axis=-1)
            swapped = np.stack([swapped] * 3, axis=-1)
        assert swapped.ndim == 3 and swapped.shape[2] == 3, (
            f"swap_face_batch requires 3D (H,W,3) image, got {swapped.shape}"
        )
        for i, (M, aimg) in enumerate(maps):
            img_fake = pred[i].transpose((1, 2, 0))  # 128x128x3
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

            # Paste back (copied from inswapper.py)
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2, :] = 0
            fake_diff[-2:, :] = 0
            fake_diff[:, :2] = 0
            fake_diff[:, -2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake_full = bgr_fake.copy()
            bgr_fake = cv2.warpAffine(
                bgr_fake, IM, (swapped.shape[1], swapped.shape[0]), borderValue=0.0
            )
            img_white = cv2.warpAffine(
                img_white, IM, (swapped.shape[1], swapped.shape[0]), borderValue=0.0
            )
            # Ensure 3D so arithmetic with 3D arrays works (prevents broadcasting error)
            if img_white.ndim == 2:
                img_white = img_white[..., np.newaxis]
            fake_diff = cv2.warpAffine(
                fake_diff, IM, (swapped.shape[1], swapped.shape[0]), borderValue=0.0
            )
            img_white[img_white > 20] = 255
            fthresh = 10
            fake_diff[fake_diff < fthresh] = 0
            fake_diff[fake_diff >= fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask == 255)
            if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
                mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
                mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
                mask_size = int(np.sqrt(mask_h * mask_w))
            else:
                mask_size = 10
            k = max(mask_size // 10, 10)
            kernel = np.ones((k, k), np.uint8)
            img_mask = cv2.erode(img_mask, kernel, iterations=1)
            kernel = np.ones((2, 2), np.uint8)
            fake_diff = cv2.warpAffine(
                np.clip(
                    (bgr_fake - swapped.astype(np.float32)).astype(np.uint8)
                    + img_white,
                    0,
                    255,
                ),
                IM,
                (swapped.shape[1], swapped.shape[0]),
                borderValue=0.0,
            )
            # Blend difference to fill seams at edges
            k2 = max(mask_size // 20, 5)
            diff_mask_mat = cv2.warpAffine(
                np.clip(
                    (bgr_fake - swapped.astype(np.float32)).astype(np.uint8)
                    + img_white,
                    0,
                    255,
                ),
                IM,
                (swapped.shape[1], swapped.shape[0]),
                borderValue=0.0,
            )
            blur_size = tuple(2 * i + 1 for i in (k2, k2))
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            img_mask /= 255
            img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            fake_merged = img_mask * bgr_fake + (1 - img_mask) * swapped.astype(
                np.float32
            )
            swapped = fake_merged.astype(np.uint8)

        return swapped

    def swap_faces_batch(
        self,
        images: List[np.ndarray],
        target_faces_list: List[List],
        source_faces_list: List[List],
    ) -> List[np.ndarray]:
        """
        Swap faces in multiple images (batched for better GPU utilization).

        Args:
            images: List of input images
            target_faces_list: List of target face lists (one per image)
            source_faces_list: List of source face lists (one per image)

        Returns:
            List of images with swapped faces
        """
        results = []
        for img, target_faces, source_faces in zip(
            images, target_faces_list, source_faces_list
        ):
            swapped = img.copy()
            # Swap each target face with corresponding source face
            for target_face, source_face in zip(target_faces, source_faces):
                swapped = self.swapper.get(
                    swapped, target_face, source_face, paste_back=True
                )
            results.append(swapped)
        return results

    def read_image(self, file_path: str) -> np.ndarray:
        """
        Read image from disk with validation.

        Args:
            file_path: Path to image file

        Returns:
            Image as numpy array (BGR format)
        """
        logger.debug("Reading image: %s", file_path)

        if not Path(file_path).exists():
            logger.error("Image file not found: %s", file_path)
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            image = cv2.imread(str(Path(file_path).resolve()), cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to read image: %s", file_path)
                raise FileNotFoundError(f"Image unreadable: {file_path}")
            logger.debug("Image read successfully: %s", file_path)
            return image
        except Exception as e:
            logger.error("Image read error: %s", e)
            raise FileProcessingError("Failed to read image") from e

    def write_image(self, file_path: str, image: np.ndarray) -> str:
        """
        Save image to disk with timestamp-based naming.

        Args:
            file_path: Destination path
            image: Image as numpy array

        Returns:
            Actual saved file path
        """
        logger.debug("Writing image: %s", file_path)

        path_obj = Path(file_path)
        unique_filename = f"{path_obj.stem}_{int(time.time())}{path_obj.suffix}"
        resolved_path = path_obj.resolve().parent / unique_filename

        try:
            success = cv2.imwrite(str(resolved_path), image)
            if not success:
                logger.error("Failed to write image: %s", resolved_path)
                raise FileProcessingError(f"Failed to write image: {resolved_path}")
            logger.debug("Image written successfully: %s", resolved_path)
            return str(resolved_path)
        except Exception as e:
            logger.error("Image write error: %s", e)
            raise FileProcessingError("Failed to write image") from e

    def process_gif(
        self, gif_path: str, output_dir: str
    ) -> Tuple[List[str], List[float]]:
        """
        Extract GIF frames with durations.

        Args:
            gif_path: Path to GIF file
            output_dir: Directory to save extracted frames

        Returns:
            Tuple of (frame_paths, frame_durations)
        """
        logger.debug("Processing GIF: %s", gif_path)

        if not Path(gif_path).exists():
            logger.error("GIF file not found: %s", gif_path)
            raise FileNotFoundError(f"GIF file not found: {gif_path}")

        result_frames = []
        frame_durations = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def process_frame(i, frame):
            frame_path = str(Path(output_dir) / f"frame_{i:04d}.png")
            try:
                success = cv2.imwrite(
                    frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )
                if not success:
                    logger.error("Failed to write frame: %s", frame_path)
                    raise FileProcessingError(f"Failed to write frame: {frame_path}")
                logger.debug("Frame written: %s", frame_path)
                return frame_path
            except Exception as e:
                logger.error("Error processing frame %d: %s", i, e)
                raise FileProcessingError("Failed to process frame") from e

        # Extract frames using video_io (FFmpeg)
        frame_results = video_io.extract_video_frames(
            gif_path,
            output_dir,
            fps=10.0,  # Default GIF FPS
        )

        result_frames = []
        for fr in frame_results:
            result_frames.append(fr.path)

        frame_durations = [int(1000 / 10.0)] * len(result_frames)
        logger.info(
            "Extracted %d frames from GIF (10 fps, 100 ms per frame)",
            len(result_frames),
        )

        logger.debug("GIF processing completed: %d frames", len(result_frames))
        return result_frames, frame_durations
