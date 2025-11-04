import cv2
import logging
import numpy as np
from pathlib import Path
from config import Config
import torch

# Import RealESRGANer from the cloned Real-ESRGAN repository.
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

class EnhancementProcessor:
    """
    Processor to enhance images using Real-ESRGAN with integrated face enhancement.
    
    When the "Real-ESRGAN" method is selected, the model upscales the image.
    Face enhancement is enabled by setting attributes on the RealESRGANer instance.
    """
    def __init__(self) -> None:
        self.config = Config()
        self.realesrganer = None
        self.gfpganer = None
        self._init_real_esrgan()
        self._init_gfpgan()
        # Set integrated face enhancement attributes on the RealESRGANer instance.
        if self.realesrganer and self.gfpganer:
            self.realesrganer.face_enhancer = self.gfpganer

    def _init_real_esrgan(self) -> None:
        """
        Initialize the Real-ESRGAN model.
        Uses the RealESRGAN_x4plus model.
        Ensure that the weight file is at:
          'models/realesrgan/weights/RealESRGAN_x4plus.pth'
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Initializing Real-ESRGAN on device: {device}")
            weight_path = Path(
                self.config.get(
                    "real_esrgan_weight_path",
                    "models/realesrgan/weights/RealESRGAN_x4plus.pth",
                )
            )
            if not weight_path.exists():
                raise FileNotFoundError(
                    f"RealESRGAN weight file not found at {weight_path}"
                )
            logging.info(f"Loading RealESRGAN weights from {weight_path}")
            half_precision = device == "cuda"
            self.realesrganer = RealESRGANer(
                scale=4,
                model_path=str(weight_path),
                dni_weight=None,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=half_precision,
                device=device,
            )
        except Exception as e:
            logging.error(f"Failed to initialize Real-ESRGAN: {e}")
            self.realesrganer = None

    def _init_gfpgan(self) -> None:
        """
        Initialize the GFPGAN model for face restoration.
        Ensure that the GFPGAN weight file is at:
          'models/gfpgan/weights/GFPGANv1.4.pth'
        """
        try:
            weight_path = Path(
                self.config.get(
                    "gfpgan_weight_path",
                    "models/gfpgan/weights/GFPGANv1.4.pth",
                )
            )
            if not weight_path.exists():
                raise FileNotFoundError(
                    f"GFPGAN weight file not found at {weight_path}"
                )
            logging.info(f"Loading GFPGAN weights from {weight_path}")
            self.gfpganer = GFPGANer(
                model_path=str(weight_path),
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
        except Exception as e:
            logging.error(f"Failed to initialize GFPGAN: {e}")
            self.gfpganer = None

    def apply_enhancement(self, image: np.ndarray, method: str) -> np.ndarray:
        """
        Enhance the image using the specified method.
        Available methods:
          - "Real-ESRGAN": Upscales the image and applies integrated face enhancement.
          - "Basic": Returns the original image.
        """
        if method.lower() == "real-esrgan":
            if not self.realesrganer:
                raise RuntimeError("Real-ESRGAN model not initialized.")
            scale = self.config.get_real_esrgan_scaling_factor()
            try:
                # Ensure contiguous uint8 data before conversion
                image = np.ascontiguousarray(image, dtype=np.uint8)

                # Real-ESRGAN expects images in BGR order
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                enhanced_image, _ = self.realesrganer.enhance(
                    image,
                    outscale=scale,
                    face_enhance=self.gfpganer is not None,
                )
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
                return enhanced_image
            except Exception as e:
                logging.error(f"Real-ESRGAN enhance failed: {e}")
                raise
        else:
            return image
