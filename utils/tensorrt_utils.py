"""
TensorRT availability detection utilities.

This module provides functions to detect if TensorRT is properly installed
and functional, to avoid noisy error messages when it's not available.
"""
import logging
import os
import sys
from functools import lru_cache
from typing import List

logger = logging.getLogger("FaceOff")

# Suppress ONNX Runtime TensorRT warnings during detection
_original_stderr = None


def _suppress_ort_warnings():
    """Temporarily suppress stderr to catch ONNX Runtime warnings."""
    global _original_stderr
    _original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')


def _restore_stderr():
    """Restore stderr after suppression."""
    global _original_stderr
    if _original_stderr is not None:
        sys.stderr.close()
        sys.stderr = _original_stderr
        _original_stderr = None


@lru_cache(maxsize=1)
def is_tensorrt_available() -> bool:
    """
    Check if TensorRT is properly installed and functional.

    This function tests actual TensorRT availability by attempting to
    query the provider, catching any errors silently.

    Returns:
        True if TensorRT is available and functional, False otherwise.
    """
    try:
        _suppress_ort_warnings()
        import onnxruntime as ort

        providers = ort.get_available_providers()

        # Check if TensorRT provider is listed
        if 'TensorrtExecutionProvider' not in providers:
            logger.debug("TensorRT provider not in available providers list")
            return False

        # Try to actually use it - this will fail if libs are missing
        try:
            # Create a minimal test to see if TensorRT actually works
            # We don't need a real model, just check if the provider initializes
            import numpy as np

            # Check if the TensorRT libraries are loadable
            # by checking if we can get provider options
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 4  # Suppress logs

            # If we get here without the DLL error, TensorRT might work
            # But we saw errors, so let's be more thorough

            # The real test is whether nvinfer DLL is loadable
            import ctypes
            try:
                # Try to load the TensorRT library directly
                if sys.platform == 'win32':
                    ctypes.CDLL('nvinfer.dll')
                else:
                    ctypes.CDLL('libnvinfer.so')
                logger.info("TensorRT libraries found and loadable")
                return True
            except OSError:
                logger.debug("TensorRT DLLs not found or not loadable")
                return False

        except Exception as e:
            logger.debug("TensorRT provider test failed: %s", e)
            return False

    except Exception as e:
        logger.debug("TensorRT availability check failed: %s", e)
        return False
    finally:
        _restore_stderr()


def get_onnx_providers(device_id: int = 0, use_tensorrt: bool = True) -> List:
    """
    Get the list of ONNX execution providers to use.

    Args:
        device_id: CUDA device ID
        use_tensorrt: Whether to attempt using TensorRT (if available)

    Returns:
        List of provider tuples for ONNX Runtime session
    """
    providers = []

    # Only try TensorRT if requested and available
    if use_tensorrt and is_tensorrt_available():
        providers.append(('TensorrtExecutionProvider', {
            'device_id': device_id,
            'trt_max_workspace_size': 2 << 30,  # 2GB
            'trt_fp16_enable': True,
        }))
        logger.debug("TensorRT provider enabled for device %d", device_id)

    # Always include CUDA as fallback
    providers.append(('CUDAExecutionProvider', {
        'device_id': device_id,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB limit
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }))

    # CPU as final fallback
    providers.append('CPUExecutionProvider')

    return providers
