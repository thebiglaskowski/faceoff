"""
Model preloading functionality for FaceOff.

This module handles preloading models at startup to reduce first-run delay.
Lives in processing layer since it needs to import from core.

Architecture (from CLAUDE.md):
- processing/ may import from core/, utils/
"""

import logging
import numpy as np

from utils.config_manager import config
from core.media_processor import MediaProcessor

logger = logging.getLogger("FaceOff")


def preload_models(device_id: int = 0) -> None:
    """
    Preload models at startup to reduce first-run delay.

    This function can be called during application startup to compile
    TensorRT engines in the background before the user makes a request.

    Args:
        device_id: GPU device ID to use for preloading
    """
    if not config.preload_on_startup:
        logger.info("Model preloading disabled in config")
        return

    logger.info("Starting model preloading on device %d...", device_id)

    try:
        # Create a dummy processor to trigger TensorRT compilation
        logger.info("Preloading face detection model (buffalo_l)...")
        processor = MediaProcessor(device_id=device_id, use_tensorrt=True, optimize_models=False)

        # Create a small dummy image to trigger model initialization
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Trigger face detection to compile TensorRT engine
        _ = processor.get_faces(dummy_image)

        logger.info("Model preloading complete (TensorRT engines compiled and cached)")

    except Exception as e:
        logger.warning("Model preloading failed: %s (will compile on first use)", e)
