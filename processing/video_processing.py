"""
Video processing with face swapping and enhancement.

Uses the chunked streaming pipeline (decode → swap → optional enhance → encode)
to avoid full-video RAM residency and PNG round-trips.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.media_processor import MediaProcessor
from processing.streaming_media import build_video_context, process_streaming
from utils.config_manager import config

logger = logging.getLogger("FaceOff")


def process_video(
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
    adaptive_detection: Optional[bool] = None,
    detection_scale: Optional[float] = None,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN",
) -> Tuple[None, Optional[str]]:
    """Process video files with face swapping via the streaming pipeline."""
    if not device_ids:
        device_ids = [0]

    if not config.streaming_enabled:
        raise RuntimeError(
            "Legacy full-memory video processing was removed. "
            "Enable streaming.enabled in config.yaml."
        )

    ctx = build_video_context(dest_path, output_dir)
    logger.info(
        "Streaming video: %s (%dx%d @ %.2f fps)",
        Path(dest_path).name,
        ctx.width,
        ctx.height,
        ctx.fps,
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