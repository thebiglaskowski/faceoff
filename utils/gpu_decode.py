"""Optional PyNvCodec GPU decode probe (Wave 4)."""

from __future__ import annotations

import logging

logger = logging.getLogger("FaceOff")

_NVcodec_AVAILABLE: bool | None = None


def nvcodec_decode_available() -> bool:
    """True when PyNvCodec is importable (optional NVIDIA video decode SDK)."""
    global _NVcodec_AVAILABLE
    if _NVcodec_AVAILABLE is not None:
        return _NVcodec_AVAILABLE
    try:
        import PyNvCodec  # noqa: F401

        _NVcodec_AVAILABLE = True
        logger.debug("PyNvCodec available for GPU decode")
    except ImportError:
        _NVcodec_AVAILABLE = False
    return _NVcodec_AVAILABLE