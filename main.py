"""
FaceOff - AI Face Swapper
Main entry point for the application.
"""
import atexit
import os
import signal
import sys
import threading
from pathlib import Path

# Add NVIDIA/TensorRT library paths so onnxruntime can find them at runtime.
# pip installs these libraries inside site-packages, not in system library paths.
import site as _site
_site_packages_dirs = _site.getsitepackages() or []
# Also include user site-packages and the venv's own site-packages
_sp = Path(sys.prefix)
for _candidate in [
    _sp / 'Lib' / 'site-packages',   # Windows venv
    _sp / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages',  # Linux venv
]:
    if _candidate.exists() and str(_candidate) not in _site_packages_dirs:
        _site_packages_dirs.append(str(_candidate))

import ctypes as _ctypes

def _preload_so(path: Path) -> None:
    """Load a shared library into the process so later dlopen calls find it."""
    try:
        _ctypes.CDLL(str(path))
    except OSError:
        pass

for _sp_dir in _site_packages_dirs:
    _sp_path = Path(_sp_dir)

    # tensorrt_libs: contains libnvinfer.so.10 (Linux) / nvinfer_10.dll (Windows)
    _tensorrt_libs = _sp_path / 'tensorrt_libs'
    if _tensorrt_libs.exists():
        if sys.platform == 'win32':
            os.environ['PATH'] = str(_tensorrt_libs) + os.pathsep + os.environ.get('PATH', '')
        else:
            # Setting LD_LIBRARY_PATH after process start doesn't affect dlopen in the
            # same process (glibc caches it at startup). Preload the key libraries with
            # ctypes so they're already in the process namespace when onnxruntime loads
            # the TensorRT provider via dlopen.
            for _so in sorted(_tensorrt_libs.glob('*.so.*')):
                _preload_so(_so)
            os.environ['LD_LIBRARY_PATH'] = str(_tensorrt_libs) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

    # nvidia sub-packages: cublas, cudnn, etc.
    _nvidia_path = _sp_path / 'nvidia'
    if _nvidia_path.exists():
        # Windows puts DLLs in bin/, Linux puts .so files in lib/
        _lib_subdir = 'bin' if sys.platform == 'win32' else 'lib'
        _path_var = 'PATH' if sys.platform == 'win32' else 'LD_LIBRARY_PATH'
        for _lib_dir in _nvidia_path.glob(f'*/{_lib_subdir}'):
            if _lib_dir.is_dir():
                if sys.platform != 'win32':
                    for _so in _lib_dir.glob('*.so.*'):
                        _preload_so(_so)
                os.environ[_path_var] = str(_lib_dir) + os.pathsep + os.environ.get(_path_var, '')

# Suppress ONNX Runtime verbose logging (especially TensorRT provider errors)
# Level 3 = WARNING, Level 4 = ERROR only
# This must be set BEFORE importing onnxruntime
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')

# Check TensorRT availability early and cache the result
from utils.tensorrt_utils import is_tensorrt_available
_tensorrt_ok = is_tensorrt_available()

import logging
from ui.app import create_app
from utils.config_manager import config
from utils.model_cache import get_cache_info
from processing.model_preloader import preload_models

logger = logging.getLogger("FaceOff")

# Track if shutdown is in progress to prevent double cleanup
_shutdown_in_progress = False
_shutdown_lock = threading.Lock()


def cleanup_resources():
    """Clean up all resources before exit."""
    global _shutdown_in_progress

    with _shutdown_lock:
        if _shutdown_in_progress:
            return
        _shutdown_in_progress = True

    logger.info("Cleaning up resources...")

    try:
        # Clean up model pool (releases ONNX sessions)
        from core.model_pool import cleanup_model_pool
        cleanup_model_pool()
    except Exception as e:
        logger.debug("Model pool cleanup: %s", e)

    try:
        # Clear enhancement model caches
        from processing.swinir_enhancement import clear_swinir_cache
        clear_swinir_cache()
    except Exception as e:
        logger.debug("SwinIR cache cleanup: %s", e)

    try:
        from processing.codeformer_restoration import clear_codeformer_cache
        clear_codeformer_cache()
    except Exception as e:
        logger.debug("CodeFormer cache cleanup: %s", e)

    try:
        # Clear CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.debug("CUDA cleanup: %s", e)

    logger.info("Cleanup complete")


def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info("Received signal %s, shutting down...", sig_name)
    cleanup_resources()
    # Force exit after cleanup - this ensures the process terminates
    os._exit(0)


# Register cleanup handlers
atexit.register(cleanup_resources)

# Register signal handlers (Windows uses SIGINT, SIGTERM; SIGBREAK for Ctrl+Break)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGBREAK'):  # Windows-specific
    signal.signal(signal.SIGBREAK, signal_handler)


if __name__ == "__main__":
    # Display cache info at startup
    cache_info = get_cache_info()
    logger.info("Model cache: %d engine(s) cached (%.2f MB total)", 
               cache_info['num_files'], cache_info['total_size_mb'])
    
    # Optional: Preload models in background to reduce first-run delay
    if config.preload_on_startup:
        logger.info("Model preloading enabled - compiling TensorRT engines...")
        preload_models(device_id=0)
    
    # Launch Gradio app
    demo = create_app()

    # Configure Gradio queue for handling concurrent requests
    # Per Context7 best practices: bound queue size and concurrency
    queue_max_size = config.get('ui', 'queue_max_size', default=20)
    queue_concurrency = config.get('ui', 'queue_concurrency_limit', default=4)
    demo.queue(max_size=queue_max_size, default_concurrency_limit=queue_concurrency)
    logger.info("Gradio queue configured: max_size=%d, concurrency=%d", queue_max_size, queue_concurrency)

    # Try the configured port first, then auto-find if busy
    try:
        demo.launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=config.share
        )
    except OSError as e:
        if "Cannot find empty port" in str(e):
            logger.info(f"Port {config.server_port} is busy, trying alternative ports...")
            # Try ports 7861-7870
            for port in range(7861, 7871):
                try:
                    demo.launch(
                        server_name=config.server_name,
                        server_port=port,
                        share=config.share
                    )
                    logger.info(f"✅ Successfully started on port {port}")
                    break
                except OSError:
                    continue
            else:
                logger.error("Could not find any available port in range 7861-7870")
                raise
        else:
            raise
