"""
FaceOff - AI Face Swapper
Main entry point for the application.
"""

import atexit
import logging
import os
import signal
import sys

# Headless-safe matplotlib backend before any dependency imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
from contextlib import suppress
from pathlib import Path

# Compatibility shim: torchvision >=0.18 removed torchvision.transforms.functional_tensor.
# Older basicsr (1.4.2), still pulled by gfpgan/realesrgan, imports rgb_to_grayscale
# from that module.  Create a fake module so the old import path still works.
import torch
from types import ModuleType
from torchvision.transforms.functional import rgb_to_grayscale

_mod = ModuleType("torchvision.transforms.functional_tensor")
_mod.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _mod

# Patch torch.load for GFPGAN dependency chain.
# WARNING: PyTorch 2.6+ defaults weights_only=True to prevent trojaned .pth files.
# These packages omit the parameter, so we pass weights_only=False as a
# compatibility shim. This is a conscious security relaxation for a
# local-only desktop tool — all model files are user-supplied.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

# ---------------------------------------------------------------------------
# Logging + NVIDIA libs — MUST run before any onnxruntime import.
# LD_LIBRARY_PATH changes do not affect an already-running process; we dlopen
# cuDNN/TensorRT from wheel paths with RTLD_GLOBAL instead.
# ---------------------------------------------------------------------------
from utils.config_manager import config
from utils.logging_setup import setup_logging
from utils.onnx_providers import preload_nvidia_libraries, setup_nvidia_library_path

# Suppress ONNX Runtime verbose logging (especially TensorRT provider errors).
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

setup_logging()
setup_nvidia_library_path()
preload_nvidia_libraries()

from utils.tensorrt_utils import is_tensorrt_available
from utils.model_cache import get_cache_info
from processing.model_preloader import preload_models
from ui.app import create_app
from ui.faceoff_theme import CUSTOM_CSS, GRADIO_THEME

logger = logging.getLogger("FaceOff")

_tensorrt_ok = is_tensorrt_available()

# Track if shutdown is in progress to prevent double cleanup.
_shutdown_in_progress = False
_shutdown_lock = _Lock = None
try:
    import threading

    _shutdown_lock = threading.Lock()
except ImportError:
    _shutdown_lock = None


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
        preload_models()

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
            theme=GRADIO_THEME,
            css=CUSTOM_CSS,
            max_file_size=f"{config.max_file_size_mb}mb",
        )
    except OSError as exc:
        if "Cannot find empty port" in str(exc):
            logger.info(
                "Port %s is busy, trying alternative ports...", config.server_port
            )
            for port in range(7861, 7871):
                try:
                    demo.launch(
                        server_name=config.server_name,
                        server_port=port,
                        share=config.share,
                        theme=GRADIO_THEME,
                        css=CUSTOM_CSS,
                        max_file_size=f"{config.max_file_size_mb}mb",
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
