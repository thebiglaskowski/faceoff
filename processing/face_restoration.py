"""
Face restoration using GFPGAN.

This module provides face enhancement/restoration capabilities to improve
face quality after swapping or for general face enhancement.
"""
import logging
import cv2
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Union
import torch

logger = logging.getLogger("FaceOff")


class FaceRestorer:
    """Wrapper for GFPGAN face restoration."""
    
    def __init__(self, device_id: int = 0, model_version: str = "1.4"):
        """
        Initialize GFPGAN face restorer.
        
        Args:
            device_id: GPU device ID to use
            model_version: GFPGAN model version ("1.3" or "1.4")
        """
        self.device_id = device_id
        self.model_version = model_version
        self._restorer = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization of GFPGAN model."""
        if self._initialized:
            return
            
        try:
            # Suppress torchvision deprecation warnings from GFPGAN
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                from gfpgan import GFPGANer
            
            # Determine model path
            model_name = f"GFPGANv{self.model_version}.pth"
            model_dir = Path(__file__).parent.parent / "models" / "gfpgan"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_name
            
            # Download model if it doesn't exist
            if not model_path.exists():
                logger.info("Downloading GFPGAN v%s model...", self.model_version)
                import urllib.request
                url = f"https://github.com/TencentARC/GFPGAN/releases/download/v{self.model_version}.0/{model_name}"
                try:
                    urllib.request.urlretrieve(url, model_path)
                    logger.info("GFPGAN model downloaded to %s", model_path)
                except Exception as download_error:
                    logger.error("Failed to download GFPGAN model from %s: %s", url, download_error)
                    raise RuntimeError(f"Could not download GFPGAN model. Please check your internet connection or manually download from {url} to {model_path}")
            
            # Initialize GFPGANer
            # Suppress torchvision deprecation warnings during model initialization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
                self._restorer = GFPGANer(
                    model_path=str(model_path),
                    upscale=1,  # We don't want upscaling, just restoration
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,  # No background upsampling
                    device=f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu'
                )
            
            self._initialized = True
            logger.info("GFPGAN v%s initialized on device %d", self.model_version, self.device_id)
            
        except Exception as e:
            logger.error("Failed to initialize GFPGAN: %s", e)
            raise
    
    def restore_face(
        self,
        image: np.ndarray,
        has_aligned: bool = False,
        weight: float = 0.5
    ) -> np.ndarray:
        """
        Restore/enhance a single face image.
        
        Args:
            image: Input image (BGR format, numpy array)
            has_aligned: Whether input is already aligned face
            weight: Balance between original and restored (0=original, 1=fully restored)
            
        Returns:
            Restored image (BGR format)
        """
        self._lazy_init()
        
        try:
            # GFPGAN expects BGR format
            if image.shape[2] == 3:
                # Process with GFPGAN
                _, _, restored_img = self._restorer.enhance(
                    image,
                    has_aligned=has_aligned,
                    only_center_face=False,
                    paste_back=True,
                    weight=weight
                )
                
                return restored_img
            else:
                logger.warning("Unexpected image format, returning original")
                return image
                
        except Exception as e:
            logger.error("Face restoration failed: %s", e)
            return image  # Return original on failure
    
    def restore_faces_in_frame(
        self,
        frame: np.ndarray,
        weight: float = 0.5
    ) -> np.ndarray:
        """
        Restore all faces in a frame.
        
        Args:
            frame: Input frame (BGR format)
            weight: Balance between original and restored
            
        Returns:
            Frame with restored faces
        """
        return self.restore_face(frame, has_aligned=False, weight=weight)
    
    def cleanup(self):
        """Release GPU resources."""
        if self._restorer is not None:
            del self._restorer
            self._restorer = None
            self._initialized = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GFPGAN resources released")


def restore_face_batch(
    images: list,
    device_id: int = 0,
    weight: float = 0.5,
    model_version: str = "1.4"
) -> list:
    """
    Restore a batch of face images.
    
    Args:
        images: List of images (BGR format)
        device_id: GPU device ID
        weight: Restoration strength (0-1)
        model_version: GFPGAN model version
        
    Returns:
        List of restored images
    """
    restorer = FaceRestorer(device_id=device_id, model_version=model_version)
    
    try:
        restored = []
        for img in images:
            restored_img = restorer.restore_face(img, weight=weight)
            restored.append(restored_img)
        return restored
    finally:
        restorer.cleanup()
