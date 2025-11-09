"""
Core media processing class for face detection and swapping.
"""
import cv2
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
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import List, Tuple

from utils.config_manager import config
from processing.model_optimizer import optimize_onnx_model
from utils.error_handler import ErrorHandler, FriendlyError

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
    
    def __init__(self, device_id: int = 0, use_tensorrt: bool = False, optimize_models: bool = True) -> None:
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
            optimized_inswapper = optimize_onnx_model(inswapper_path, optimization_level="basic")
            
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
            logger.info("Acquiring model initialization lock for device %d", self.device_id)
            try:
                # Configure CUDA provider for specific device
                cuda_provider_options = {
                    'device_id': self.device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                
                # Build provider list with optional TensorRT
                if self.use_tensorrt:
                    # Check if TensorRT is available
                    try:
                        import onnxruntime as ort
                        available_providers = ort.get_available_providers()
                        
                        if 'TensorrtExecutionProvider' in available_providers:
                            tensorrt_provider_options = {
                                'device_id': self.device_id,
                                'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
                                'trt_fp16_enable': False,  # Disable FP16 - can cause quality issues
                                'trt_engine_cache_enable': True,  # Cache TensorRT engines
                                'trt_engine_cache_path': config.tensorrt_cache_dir,
                            }
                            providers = [
                                ('TensorrtExecutionProvider', tensorrt_provider_options),
                                ('CUDAExecutionProvider', cuda_provider_options),
                                'CPUExecutionProvider'
                            ]
                            logger.info("TensorRT optimization enabled on device %d (FP32 mode)", self.device_id)
                        else:
                            logger.warning("TensorRT not available. Install TensorRT libraries for acceleration. Falling back to CUDA.")
                            providers = [
                                ('CUDAExecutionProvider', cuda_provider_options),
                                'CPUExecutionProvider'
                            ]
                            self.use_tensorrt = False  # Disable flag since it's not available
                    except Exception as e:
                        logger.warning("TensorRT check failed: %s. Falling back to CUDA.", e)
                        providers = [
                            ('CUDAExecutionProvider', cuda_provider_options),
                            'CPUExecutionProvider'
                        ]
                        self.use_tensorrt = False
                else:
                    providers = [
                        ('CUDAExecutionProvider', cuda_provider_options),
                        'CPUExecutionProvider'
                    ]
                
                # Initialize face analysis (can use TensorRT) - suppress verbose ONNX output
                logger.debug("Loading face analysis models (InsightFace)...")
                with suppress_insightface_output():
                    self.face_app = insightface.app.FaceAnalysis(
                        name=config.face_analysis_name,
                        root=".",
                        providers=providers,
                    )
                    self.face_app.prepare(
                        ctx_id=self.device_id,
                        det_size=tuple(config.face_analysis_det_size)
                    )
                
                # Log actual providers being used
                if hasattr(self.face_app.models, 'det_model') and hasattr(self.face_app.models['det_model'], 'session'):
                    actual_providers = self.face_app.models['det_model'].session.get_providers()
                    logger.debug("Face analysis using providers: %s", actual_providers)
                
                # Initialize face swapper (CUDA only - TensorRT has compilation issues with this model)
                logger.debug("Loading face swapper model (inswapper)...")
                swapper_providers = [
                    ('CUDAExecutionProvider', cuda_provider_options),
                    'CPUExecutionProvider'
                ]
                with suppress_insightface_output():
                    self.swapper = insightface.model_zoo.get_model(
                        config.inswapper_model_path,
                        download=False,
                        download_zip=False,
                        providers=swapper_providers
                    )
                logger.info("Face swapper using CUDA only (TensorRT not compatible with inswapper model)")
                
                logger.info("MediaProcessor initialized on device %d", self.device_id)
                logger.info("Released model initialization lock for device %d", self.device_id)
                
            except FriendlyError:
                # Re-raise friendly errors
                raise
            except Exception as e:
                logger.error("Model initialization failed: %s", e)
                # Convert to friendly error
                context = {
                    'device_id': self.device_id,
                    'use_tensorrt': self.use_tensorrt,
                    'optimize_models': self.optimize_models
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
    
    def swap_faces_batch(self, images: List[np.ndarray], target_faces_list: List[List], 
                         source_faces_list: List[List]) -> List[np.ndarray]:
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
        for img, target_faces, source_faces in zip(images, target_faces_list, source_faces_list):
            swapped = img.copy()
            # Swap each target face with corresponding source face
            for target_face, source_face in zip(target_faces, source_faces):
                swapped = self.swapper.get(swapped, target_face, source_face, paste_back=True)
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
    
    def process_gif(self, gif_path: str, output_dir: str) -> Tuple[List[str], List[float]]:
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
        
        with VideoFileClip(gif_path) as clip:
            # Get frame duration (time per frame)
            fps = clip.fps if clip.fps else 10  # Default to 10 FPS if not specified
            frame_duration_ms = int(1000 / fps)  # Convert to milliseconds
            
            # Extract frames in parallel
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_frame, i, frame): i
                    for i, frame in enumerate(clip.iter_frames())
                }
                for future in futures:
                    try:
                        result_frames.append(future.result())
                    except Exception as e:
                        logger.error("Error processing frame %d: %s", futures[future], e)
                        raise FileProcessingError("Failed to process GIF frames") from e
            
            # Use consistent frame duration for all frames based on FPS
            frame_durations = [frame_duration_ms] * len(result_frames)
            logger.info("Extracted %d frames from GIF (FPS: %.2f, duration: %dms per frame)", 
                       len(result_frames), fps, frame_duration_ms)
        
        logger.debug("GIF processing completed: %d frames", len(result_frames))
        return result_frames, frame_durations
