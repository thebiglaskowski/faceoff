"""
GIF processing with face swapping and enhancement.

Uses the same chunked streaming pipeline as video (decode → swap → optional enhance → encode).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.media_processor import MediaProcessor
from processing.streaming_media import build_gif_context, process_streaming

logger = logging.getLogger("FaceOff")


def extract_duration(duration) -> int:
    """Extract duration value from various formats (ms)."""
    try:
        if isinstance(duration, (list, np.ndarray)):
            return int(duration[0]) if len(duration) > 0 else 100
        return int(duration)
    except (TypeError, ValueError):
        logger.warning("Invalid duration value: %s. Defaulting to 100ms.", duration)
        return 100


def process_gif(
    processor: MediaProcessor,
    source_image: np.ndarray,
    dest_path: str,
    output_dir: Path,
    enhance: bool = False,
    tile_size: int = 256,
    outscale: int = 4,
    face_confidence: float = 0.5,
    device_ids: Optional[List[int]] = None,
    face_mappings: Optional[List[Tuple[int, int]]] = None,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0,
    restore_faces: bool = False,
    restoration_weight: float = 0.5,
    adaptive_detection: bool = None,
    detection_scale: float = None,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN",
) -> Tuple[None, Optional[str]]:
    """Process GIF files with face swapping via the streaming pipeline."""
    if not device_ids:
        device_ids = [0]

    logger.info("process_gif received face_mappings: %s", face_mappings)
    ctx = build_gif_context(dest_path, output_dir)
    logger.info(
        "Streaming GIF: %s (%dx%d, %d duration entries)",
        Path(dest_path).name,
        ctx.width,
        ctx.height,
        len(ctx.frame_durations or []),
    )

    return process_streaming(
        processor,
        source_image,
        ctx,
        face_confidence=face_confidence,
        device_ids=device_ids,
        face_mappings=face_mappings,
        enhance=enhance,
        tile_size=tile_size,
        outscale=outscale,
        model_name=model_name,
        denoise_strength=denoise_strength,
        use_fp32=use_fp32,
        pre_pad=pre_pad,
        restore_faces=restore_faces,
        restoration_weight=restoration_weight,
        adaptive_detection=adaptive_detection,
        detection_scale=detection_scale,
        enhancement_model=enhancement_model,
        restoration_model=restoration_model,
    )