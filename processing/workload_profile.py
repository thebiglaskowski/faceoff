"""Automatic GPU pipeline profiles per workload (Wave 4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from utils.config_manager import config

logger = logging.getLogger("FaceOff")

_PIXELS_1080P = 1920 * 1080
_PIXELS_4K = 3840 * 2160


@dataclass(frozen=True)
class WorkloadProfile:
    """Resolved runtime flags for a single processing job."""

    name: str
    frame_retention: bool
    paste_on_gpu: bool
    detection_on_gpu: bool
    enhancement_chain: bool
    zero_copy: bool
    pinned_encode: bool
    defer_download: bool
    chunk_size: Optional[int] = None
    use_nvcodec_decode: bool = False

    def summary(self) -> str:
        parts = [
            f"retention={self.frame_retention}",
            f"paste={self.paste_on_gpu}",
            f"det_gpu={self.detection_on_gpu}",
            f"enh_chain={self.enhancement_chain}",
            f"zero_copy={self.zero_copy}",
            f"pinned_enc={self.pinned_encode}",
            f"defer_d2h={self.defer_download}",
        ]
        if self.chunk_size is not None:
            parts.append(f"chunk={self.chunk_size}")
        if self.use_nvcodec_decode:
            parts.append("nvcodec=1")
        return ", ".join(parts)


def _chunk_for_pixels(
    pixels: int,
    *,
    enhance: bool,
    outscale: int,
    base_swap: int = 32,
) -> int:
    """VRAM-aware chunk size from frame resolution."""
    if pixels >= _PIXELS_4K:
        size = 8 if enhance else 12
    elif pixels >= _PIXELS_1080P:
        size = 16 if enhance else 24
    elif pixels >= 1280 * 720:
        size = 24 if enhance else 32
    else:
        size = base_swap

    if enhance and outscale >= 4:
        size = max(config.min_batch_size, size // 2)
    return max(config.min_batch_size, min(size, config.streaming_chunk_size))


def _cuda_ready() -> bool:
    return torch.cuda.is_available()


def resolve_workload_profile(
    *,
    media_type: str,
    enhance: bool,
    enhancement_model: str,
    restore_faces: bool,
    face_mappings: Optional[List[Tuple[int, int]]],
    width: int = 0,
    height: int = 0,
    outscale: int = 4,
    face_enhance_in_esrgan: bool = False,
) -> Optional[WorkloadProfile]:
    """
    Pick an automatic pipeline profile for this job.

    Returns None when ``gpu.auto_workload_tune`` is disabled (use config.yaml).
    """
    if not config.gpu_auto_workload_tune:
        return None

    has_mappings = bool(face_mappings)
    pixels = max(0, width) * max(0, height)
    chunk = _chunk_for_pixels(pixels, enhance=enhance, outscale=outscale)

    if not _cuda_ready():
        return WorkloadProfile(
            name="cpu_fallback",
            frame_retention=False,
            paste_on_gpu=False,
            detection_on_gpu=False,
            enhancement_chain=False,
            zero_copy=False,
            pinned_encode=False,
            defer_download=False,
            chunk_size=chunk,
        )

    from utils.gpu_decode import nvcodec_decode_available

    nvcodec = nvcodec_decode_available() and media_type == "video"

    if restore_faces:
        return WorkloadProfile(
            name="face_restore",
            frame_retention=True,
            paste_on_gpu=not has_mappings,
            detection_on_gpu=True,
            enhancement_chain=False,
            zero_copy=media_type == "video",
            pinned_encode=False,
            defer_download=False,
            chunk_size=chunk,
            use_nvcodec_decode=nvcodec,
        )

    if media_type == "image":
        if enhance:
            return WorkloadProfile(
                name="image_enhance",
                frame_retention=False,
                paste_on_gpu=False,
                detection_on_gpu=True,
                enhancement_chain=False,
                zero_copy=False,
                pinned_encode=False,
                defer_download=False,
            )
        return WorkloadProfile(
            name="image_swap",
            frame_retention=False,
            paste_on_gpu=False,
            detection_on_gpu=True,
            enhancement_chain=False,
            zero_copy=False,
            pinned_encode=False,
            defer_download=False,
        )

    # Streaming video / GIF
    if not enhance:
        return WorkloadProfile(
            name="stream_swap_only",
            frame_retention=True,
            paste_on_gpu=not has_mappings,
            detection_on_gpu=True,
            enhancement_chain=False,
            zero_copy=media_type == "video",
            pinned_encode=media_type == "video",
            defer_download=False,
            chunk_size=chunk,
            use_nvcodec_decode=nvcodec,
        )

    if face_enhance_in_esrgan:
        return WorkloadProfile(
            name="stream_esrgan_face_enhance",
            frame_retention=True,
            paste_on_gpu=not has_mappings,
            detection_on_gpu=True,
            enhancement_chain=False,
            zero_copy=media_type == "video",
            pinned_encode=False,
            defer_download=False,
            chunk_size=max(config.min_batch_size, chunk // 2),
            use_nvcodec_decode=nvcodec,
        )

    chain_model = enhancement_model.upper()
    if chain_model in ("HAT", "REALESRGAN", "SWINIR"):
        return WorkloadProfile(
            name=f"stream_{chain_model.lower()}_chain",
            frame_retention=True,
            paste_on_gpu=not has_mappings,
            detection_on_gpu=True,
            enhancement_chain=True,
            zero_copy=media_type == "video",
            pinned_encode=media_type == "video",
            defer_download=True,
            chunk_size=chunk,
            use_nvcodec_decode=nvcodec,
        )

    return WorkloadProfile(
        name="stream_enhance_default",
        frame_retention=True,
        paste_on_gpu=not has_mappings,
        detection_on_gpu=True,
        enhancement_chain=False,
        zero_copy=media_type == "video",
        pinned_encode=False,
        defer_download=False,
        chunk_size=chunk,
        use_nvcodec_decode=nvcodec,
    )


def log_workload_profile(profile: Optional[WorkloadProfile]) -> None:
    if profile is None:
        logger.info("Workload profile: config.yaml defaults (auto_workload_tune=false)")
        return
    logger.info("Workload profile: %s — %s", profile.name, profile.summary())


def flag(
    profile: Optional[WorkloadProfile],
    field: str,
    config_value: bool,
) -> bool:
    """Resolve a boolean pipeline flag from profile or config."""
    if profile is not None:
        return bool(getattr(profile, field))
    return config_value