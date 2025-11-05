"""
Core media processing class for face detection and swapping.
"""
import cv2
import insightface
import logging
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import List, Tuple

from config import Config

logger = logging.getLogger("FaceOff")


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
    
    def __init__(self, device_id: int = 0) -> None:
        """
        Initialize MediaProcessor on specified GPU.
        
        Args:
            device_id: CUDA device ID to use
        """
        self.device_id = device_id
        self.config = Config()
        self._validate_environment()
        self._initialize_models()
    
    def _validate_environment(self) -> None:
        """Validate required model files exist."""
        required_paths = {
            "inswapper_model": Path(self.config.get("inswapper_model_path")),
            "buffalo_model": Path(self.config.get("buffalo_model_path")),
        }
        
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing critical resources: {', '.join(missing)}")
    
    def _initialize_models(self) -> None:
        """Initialize face analysis and swapper models with CUDA."""
        try:
            # Configure CUDA provider for specific device
            cuda_provider_options = {
                'device_id': self.device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            
            providers = [
                ('CUDAExecutionProvider', cuda_provider_options),
                'CPUExecutionProvider'
            ]
            
            # Initialize face analysis
            self.face_app = insightface.app.FaceAnalysis(
                name=self.config.get("face_analysis_name"),
                root=".",
                providers=providers,
            )
            self.face_app.prepare(
                ctx_id=self.device_id,
                det_size=tuple(self.config.get("face_analysis_det_size"))
            )
            
            # Initialize face swapper
            self.swapper = insightface.model_zoo.get_model(
                self.config.get("inswapper_model_path"),
                download=False,
                download_zip=False,
                providers=providers
            )
            
            logger.info("MediaProcessor initialized on device %d", self.device_id)
            
        except Exception as e:
            logger.error("Model initialization failed: %s", e)
            raise ModelInitializationError("Failed to initialize models") from e
    
    def get_faces(self, image: np.ndarray):
        """
        Detect and analyze faces in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with metadata
        """
        return self.face_app.get(image)
    
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
            
            # Extract frame durations
            for frame in clip.iter_frames(with_times=True):
                frame_durations.append(frame[1])
        
        logger.debug("GIF processing completed: %d frames", len(result_frames))
        return result_frames, frame_durations
