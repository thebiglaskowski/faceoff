"""
FaceOff - AI Face Swapper
Main entry point for the application.
"""
import atexit
import os
import signal
import sys
import ctypes as _ctypes
import logging
from pathlib import Path
from contextlib import suppress

# Compat shim: torchvision >=0.18 removed torchvision.transforms.functional_tensor.
# Older basicsr (1.4.2), still pulled by gfpgan/realesrgan, imports rgb_to_grayscale
# from that module.  Patch sys.modules before anyone else can import it.
from types import ModuleType
from torchvision.transforms.functional import rgb_to_grayscale
_fts = ModuleType("torchvision.transforms.functional_tensor")
_fts.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _fts

# ---------------------------------------------------------------------------
# NVIDIA / TensorRT library discovery (Linux-native).
#
# pip installs these libraries inside site-packages so later dlopen calls
# can find them. On Linux we need to preload the shared objects so they stay
# in the process namespace when onnxruntime loads the TensorRT provider via
# dlopen.
# ---------------------------------------------------------------------------
from utils.tensorrt_utils import is_tensorrt_available
from utils.config_manager import config
from utils.model_cache import get_cache_info
from processing.model_preloader import preload_models
from ui.app import create_app

logger = logging.getLogger("FaceOff")

# Suppress ONNX Runtime verbose logging (especially TensorRT provider errors).
# Level 3 = WARNING, Level 4 = ERROR only. This must be set BEFORE importing onnxruntime.
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

# Check TensorRT availability early and cache the result.
_tensorrt_ok = is_tensorrt_available()

# Track if shutdown is in progress to prevent double cleanup.
_shutdown_in_progress = False
_shutdown_lock = _Lock = None
try:
    import threading
    _shutdown_lock = threading.Lock()
except ImportError:
    _shutdown_lock = None


def _preload_lib(path: Path) -> None:
    """Load a shared library into the process so later dlopen calls find it."""
    with suppress(OSError):
        _ctypes.CDLL(str(path))


_PRELOAD_DIRS: list[Path] = []

_sp = Path(sys.prefix)
for _sub in (
    _sp / "Lib" / "site-packages",   # Windows venv (keep for cross-platform compat)
    _sp / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
):
    if not _sub.exists():
        continue

    # -- tensorrt_libs: contains libnvinfer.so* --
    _trt = _sub / "tensorrt_libs"
    if _trt.exists():
        _PRELOAD_DIRS.append(_trt)
        os.environ["LD_LIBRARY_PATH"] = str(_trt) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    # -- nvidia sub-packages: cublas, cudnn, etc. --
    _nvidia = _sub / "nvidia"
    if _nvidia.exists():
        for _lib_dir in _nvidia.glob("*/lib"):
            if _lib_dir.is_dir():
                _PRELOAD_DIRS.append(_lib_dir)
                os.environ["LD_LIBRARY_PATH"] = str(_lib_dir) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def cleanup_resources() -> None:
    """Clean up all resources before exit."""
    global _shutdown_in_progress

    if _shutdown_lock:
        with _shutdown_lock:
            if _shutdown_in_progress:
                return
            _shutdown_in_progress = True
    else:
        # Fallback: no threading — still guard against double-cleanup
        if _shutdown_in_progress:
            return
        _shutdown_in_progress = True

    logger.info("Cleaning up resources...")

    try:
        from core.model_pool import cleanup_model_pool
        cleanup_model_pool()
    except Exception as exc:
        logger.debug("Model pool cleanup: %s", exc)

    try:
        from processing.swinir_enhancement import clear_swinir_cache
        clear_swinir_cache()
    except Exception as exc:
        logger.debug("SwinIR cache cleanup: %s", exc)

    try:
        from processing.codeformer_restoration import clear_codeformer_cache
        clear_codeformer_cache()
    except Exception as exc:
        logger.debug("CodeFormer cache cleanup: %s", exc)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("CUDA cleanup: %s", exc)

    logger.info("Cleanup complete")


def _signal_handler(signum: int, _frame: object) -> None:
    """Handle termination signals for graceful shutdown — sys.exit(0)."""
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    if hasattr(signal, "Signals"):
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            pass
    logger.info("Received signal %s, shutting down...", sig_name)
    cleanup_resources()
    # Use sys.exit instead of os._exit so Python's normal cleanup runs.
    sys.exit(0)


# Register cleanup handlers.
atexit.register(cleanup_resources)

# Register signal handlers.
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
if hasattr(signal, "SIGBREAK"):  # Windows-specific
    signal.signal(signal.SIGBREAK, _signal_handler)


if __name__ == "__main__":
    # Display cache info at startup.
    cache_info = get_cache_info()
    logger.info(
        "Model cache: %d engine(s) cached (%.2f MB total)",
        cache_info["num_files"],
        cache_info["total_size_mb"],
    )

    # Optional: Preload models in background to reduce first-run delay.
    if config.preload_on_startup:
        logger.info("Model preloading enabled - compiling TensorRT engines...")
        preload_models(device_id=0)

    # Launch Gradio app.
    demo = create_app()

    # Configure Gradio queue for handling concurrent requests.
    queue_max_size = config.get("ui", "queue_max_size", default=20)
    queue_concurrency = config.get("ui", "queue_concurrency_limit", default=4)
    demo.queue(max_size=queue_max_size, default_concurrency_limit=queue_concurrency)
    logger.info(
        "Gradio queue configured: max_size=%d, concurrency=%d",
        queue_max_size,
        queue_concurrency,
    )

    # Try the configured port first, then auto-find if busy.
    try:
        demo.launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=config.share,
        )
    except OSError as exc:
        if "Cannot find empty port" in str(exc):
            logger.info("Port %s is busy, trying alternative ports...", config.server_port)
            for port in range(7861, 7871):
                try:
                    demo.launch(
                        server_name=config.server_name,
                        server_port=port,
                        share=config.share,
                    )
                    logger.info("Successfully started on port %d", port)
                    break
                except OSError:
                    continue
            else:
                logger.error("Could not find any available port in range 7861-7870")
                raise
        else:
            raise
