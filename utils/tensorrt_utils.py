"""
TensorRT availability detection utilities.

This module provides functions to detect if TensorRT is properly installed
and functional, to avoid noisy error messages when it's not available.
"""

import logging
from functools import lru_cache
from typing import List

from utils.onnx_providers import (
    build_onnx_providers,
    is_tensorrt_runtime_available,
)

logger = logging.getLogger("FaceOff")


@lru_cache(maxsize=1)
def is_tensorrt_available() -> bool:
    """Return True only when TensorRT shared libraries are loadable."""
    return is_tensorrt_runtime_available()


def get_onnx_providers(
    device_id: int = 0,
    use_tensorrt: bool = True,
    use_fp16: bool = True,
    workspace_mb: int = 2048,
) -> List:
    """Get ONNX execution providers for a device."""
    return build_onnx_providers(
        device_id=device_id,
        use_tensorrt=use_tensorrt,
        use_fp16=use_fp16,
        workspace_mb=workspace_mb,
    )