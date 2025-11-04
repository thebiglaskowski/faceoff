import time
import cv2
import logging
import numpy as np
import insightface
from pathlib import Path
from typing import List, Tuple
from moviepy.editor import VideoFileClip
from config import Config
from concurrent.futures import ThreadPoolExecutor

# Initialize logger
logger = logging.getLogger("FaceOff")

config = Config()


class ModelInitializationError(Exception):
    """Raised when model initialization fails"""

    pass


class FileProcessingError(Exception):
    """Raised when file processing fails"""

    pass


class MediaProcessor:
    """Handles media processing with CUDA acceleration and enhanced error handling"""

    def __init__(self, device_id=0) -> None:
        self.device_id = device_id
        self._validate_environment()
        self._initialize_models()

    def _validate_environment(self) -> None:
        """Validate required models and CUDA availability"""
        required_paths = {
            "inswapper_model": Path(config.get("inswapper_model_path")),
            "buffalo_model": Path(config.get("buffalo_model_path")),
        }

        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing critical resources: {', '.join(missing)}")

    def _initialize_models(self) -> None:
        """Initialize models with CUDA context management"""
        try:
            # Configure CUDA provider with specific device
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
            
            self.face_app = insightface.app.FaceAnalysis(
                name=config.get("face_analysis_name"),
                root=".",
                providers=providers,
            )
            self.face_app.prepare(
                ctx_id=self.device_id, det_size=tuple(config.get("face_analysis_det_size"))
            )
            self.swapper = insightface.model_zoo.get_model(
                config.get("inswapper_model_path"), 
                download=False, 
                download_zip=False,
                providers=providers
            )
        except Exception as e:
            logging.error("Model initialization failed: %s", e)
            raise ModelInitializationError("Failed to initialize models") from e

    def get_faces(self, image: np.ndarray):
        """Extract faces using the internal face analysis model"""
        return self.face_app.get(image)

    def read_image(self, file_path: str) -> np.ndarray:
        """Read image with validation and automatic orientation handling"""
        logger.debug("Attempting to read image from path: %s", file_path)

        if not Path(file_path).exists():
            logger.error("Image file not found: %s", file_path)
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            image = cv2.imread(str(Path(file_path).resolve()), cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to read image: %s", file_path)
                raise FileNotFoundError(f"Image file not found or unreadable: {file_path}")
            logger.debug("Image read successfully: %s", file_path)
            return image
        except Exception as e:
            logger.error("Image read error: %s", e)
            raise FileProcessingError("Failed to read image") from e

    def write_image(self, file_path: str, image: np.ndarray) -> str:
        """Save an image to disk using timestamp-based naming"""
        logger.debug("Attempting to write image to path: %s", file_path)

        path_obj = Path(file_path)  # Convert to Path object
        unique_filename = f"{path_obj.stem}_{int(time.time())}{path_obj.suffix}"
        resolved_path = path_obj.resolve().parent / unique_filename

        try:
            success = cv2.imwrite(str(resolved_path), image)
            if not success:
                logger.error("Failed to write image to %s", resolved_path)
                raise FileProcessingError(f"Failed to write image: {resolved_path}")
            logger.debug("Image written successfully to: %s", resolved_path)
            return str(resolved_path)
        except Exception as e:
            logger.error("Image write error: %s", e)
            raise FileProcessingError("Failed to write image") from e

    def process_gif(self, gif_path: str, output_dir: str) -> Tuple[List[str], List[float]]:
        """Process GIF frames with parallelized resource cleanup and preserve frame durations."""
        logger.debug("Attempting to process GIF from path: %s", gif_path)

        if not Path(gif_path).exists():
            logger.error("GIF file not found: %s", gif_path)
            raise FileNotFoundError(f"GIF file not found: {gif_path}")

        result_frames = []
        frame_durations = []  # Store frame durations
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
                logger.debug("Frame written successfully: %s", frame_path)
                return frame_path
            except Exception as e:
                logger.error("Error processing frame %d: %s", i, e)
                raise FileProcessingError("Failed to process frame") from e

        with VideoFileClip(gif_path) as clip:
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

            # Extract frame durations from the original GIF
            for frame in clip.iter_frames(with_times=True):
                frame_durations.append(frame[1])

        logger.debug("GIF processing completed successfully")
        return result_frames, frame_durations

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance the quality of an image using Real-ESRGAN."""
        try:
            from realesrgan import RealESRGANer

            # Get the model path from the config
            model_path = config.get("realesrgan_model_path")
            logger.info("Using Real-ESRGAN model path: %s", model_path)  # Debugging log

            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError(f"Real-ESRGAN model path is invalid: {model_path}")

            # Fix: Ensure Real-ESRGAN model path is correctly resolved
            model_path = Path(config.get("realesrgan_model_path")).resolve()
            logger.info("Resolved Real-ESRGAN model path: %s", model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Real-ESRGAN model path does not exist: {model_path}")

            # Initialize the Real-ESRGAN model
            model = RealESRGANer(
                scale=4,
                model_path=model_path,
                dni_weight=0.5,
                device="cuda",
            )

            # Convert the image to RGB if it is not already
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # Enhance the image
            enhanced_image, _ = model.enhance(image, outscale=4)
            return enhanced_image
        except ImportError as e:
            logging.error("Real-ESRGAN module not found: %s", e)
            raise ImportError("Real-ESRGAN module is required but not installed.") from e
        except Exception as e:
            logging.error("Error during image enhancement: %s", e)
            raise RuntimeError("Failed to enhance image using Real-ESRGAN.") from e

    # Remaining methods (swap_faces_image, etc)
    # follow similar patterns with enhanced error handling and resource management
