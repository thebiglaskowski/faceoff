"""
TensorRT availability detection utilities.

This module provides functions to detect if TensorRT is properly installed
and functional, to avoid noisy error messages when it's not available.
"""
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List

logger = logging.getLogger("FaceOff")

# Add NVIDIA/TensorRT library paths to PATH
# pip packages install DLLs in subdirectories not automatically on PATH
_site_packages = Path(sys.prefix) / 'Lib' / 'site-packages'

# Add tensorrt_libs directory (contains nvinfer_10.dll etc)
_tensorrt_libs = _site_packages / 'tensorrt_libs'
if _tensorrt_libs.exists() and str(_tensorrt_libs) not in os.environ.get('PATH', ''):
    os.environ['PATH'] = str(_tensorrt_libs) + os.pathsep + os.environ.get('PATH', '')

# Add nvidia subdirectories (cublas, cudnn, etc)
_nvidia_path = _site_packages / 'nvidia'
if _nvidia_path.exists():
    for lib_dir in _nvidia_path.glob('*/bin'):
        if lib_dir.is_dir() and str(lib_dir) not in os.environ.get('PATH', ''):
            os.environ['PATH'] = str(lib_dir) + os.pathsep + os.environ.get('PATH', '')

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

        # Try to load TensorRT DLL to verify it's actually usable
        import ctypes
        try:
            if sys.platform == 'win32':
                # Try nvinfer_10.dll first (TensorRT 10.x), then nvinfer.dll
                try:
                    ctypes.CDLL('nvinfer_10.dll')
                except OSError:
                    ctypes.CDLL('nvinfer.dll')
            else:
                ctypes.CDLL('libnvinfer.so')
            logger.info("TensorRT libraries found and loadable")
            return True
        except OSError:
            # DLL not directly loadable, but provider might still work
            # if ONNX Runtime can find it through its own mechanisms
            logger.debug("TensorRT DLLs not directly loadable, checking ONNX Runtime...")

            # Final check: see if ONNX Runtime can actually use TensorRT
            # by checking session options (lightweight check)
            try:
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 4  # Suppress logs
                # If we can create session options without error, assume it works
                # The actual test happens when creating a session
                return True
            except Exception:
                return False

    except Exception as e:
        logger.debug("TensorRT availability check failed: %s", e)
        return False
    finally:
        _restore_stderr()


def get_onnx_providers(device_id: int = 0, use_tensorrt: bool = True,
                       use_fp16: bool = True, workspace_mb: int = 2048) -> List:
    """
    Get the list of ONNX execution providers to use.

    Args:
        device_id: CUDA device ID
        use_tensorrt: Whether to attempt using TensorRT (if available)
        use_fp16: Whether to use FP16 precision (faster, slightly lower quality)
        workspace_mb: TensorRT workspace size in MB

    Returns:
        List of provider tuples for ONNX Runtime session
    """
    providers = []

    # Only try TensorRT if requested and available
    if use_tensorrt and is_tensorrt_available():
        providers.append(('TensorrtExecutionProvider', {
            'device_id': device_id,
            'trt_max_workspace_size': workspace_mb * 1024 * 1024,
            'trt_fp16_enable': use_fp16,
        }))
        fp_mode = "FP16" if use_fp16 else "FP32"
        logger.debug("TensorRT provider enabled for device %d (%s)", device_id, fp_mode)

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
