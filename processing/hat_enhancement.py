"""
HAT (Hybrid Attention Transformer) super-resolution enhancement module.

Provides HAT integration as an alternative enhancement backend alongside
Real-ESRGAN and SwinIR, following the same API pattern.

Uses the BasicSR-based HAT architecture with .pth weight files.
Download URLs (HAT-Base arch — 96 embed_dim, 20.8M params):
  HAT-Base (4x): https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth
  HAT (4x) GAN:  https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth
"""
import gc
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils.config_manager import config
from utils.lru_cache import LRUModelCache
from utils.memory_manager import MemoryManager, refresh_gpu_memory

logger = logging.getLogger("FaceOff")


# =============================================================================
# HAT Model Definitions (auto-registered via basicsr ARCH_REGISTRY)
# =============================================================================

# Pre-configure the arch scan so that hat_arch.py registers HAT with
# basicsr before any model-loading code runs.
def _register_hat_archs():
    """Import the arches module to trigger auto-registration."""
    # This is already imported transitively via processing.hat_models
    # but import explicitly here to ensure registration order.
    import processing.hat_models.arches  # noqa: F401


_register_hat_archs()


# =============================================================================
# Model Configuration
# =============================================================================

# Official HAT-SRx4 / Real_HAT_GAN pretrained arch (see XPixelGroup/HAT options/test)
_OFFICIAL_HAT_ARCH = {
    "embed_dim": 180,
    "num_heads": (6, 6, 6, 6, 6, 6),
    "depths": (6, 6, 6, 6, 6, 6),
    "window_size": 16,
    "mlp_ratio": 2.0,
    "compress_ratio": 3,
    "squeeze_factor": 30,
    "conv_scale": 0.01,
    "overlap_ratio": 0.5,
    "img_range": 1.0,
    "upscale": 4,
    "resi_connection": "1conv",
    "upsampler": "pixelshuffle",
}

HAT_MODELS = {
    "HAT_Base_4x_ImageNet": {
        "url": "https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth",
        "scale": 4,
        "description": "HAT 4x pre-trained on ImageNet — strong general SR",
        "arch": dict(_OFFICIAL_HAT_ARCH),
        "param_key": "params_ema",
    },
    "HAT_Base_4x_GAN_sharper": {
        "url": "https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth",
        "scale": 4,
        "description": "HAT GAN 4x — sharper perceptual quality for real-world SR",
        "arch": dict(_OFFICIAL_HAT_ARCH),
        "param_key": "params_ema",
    },
}

DEFAULT_HAT_MODEL = "HAT_Base_4x_ImageNet"


def _hat_cache_cleanup(model_tuple):
    """Cleanup function for HAT model cache entries."""
    try:
        model = model_tuple[0]
        device = model_tuple[1] if len(model_tuple) > 1 else None
        if hasattr(model, "cpu"):
            model.cpu()
        if (
            torch.cuda.is_available()
            and isinstance(device, torch.device)
            and device.type == "cuda"
        ):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


_hat_cache = LRUModelCache("HAT", cleanup_fn=_hat_cache_cleanup)
_hat_load_lock = threading.Lock()
_hat_gpu_locks: Dict[int, threading.Lock] = {}
_hat_gpu_locks_guard = threading.Lock()


def _hat_gpu_lock(gpu_id: int) -> threading.Lock:
    """Per-GPU lock so concurrent workers don't race the CUDA allocator."""
    with _hat_gpu_locks_guard:
        if gpu_id not in _hat_gpu_locks:
            _hat_gpu_locks[gpu_id] = threading.Lock()
        return _hat_gpu_locks[gpu_id]


def _set_cuda_device(gpu_id: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

# Back-compat aliases used by padding helpers/tests
HAT_UPSCALE = _OFFICIAL_HAT_ARCH["upscale"]
HAT_IMG_RANGE = _OFFICIAL_HAT_ARCH["img_range"]
HAT_WINDOW_SIZE = _OFFICIAL_HAT_ARCH["window_size"]
HAT_EMBED_DIM = _OFFICIAL_HAT_ARCH["embed_dim"]
HAT_NUM_HEADS = _OFFICIAL_HAT_ARCH["num_heads"]
HAT_DEPTH = _OFFICIAL_HAT_ARCH["depths"]
HAT_COMPRESS_RATIO = _OFFICIAL_HAT_ARCH["compress_ratio"]
HAT_SQUEEZE_FACTOR = _OFFICIAL_HAT_ARCH["squeeze_factor"]
HAT_CONVS_SCALE = _OFFICIAL_HAT_ARCH["conv_scale"]
HAT_OVERLAP_RATIO = _OFFICIAL_HAT_ARCH["overlap_ratio"]
HAT_MLP_RATIO = _OFFICIAL_HAT_ARCH["mlp_ratio"]
HAT_RESI_CONNECTION = _OFFICIAL_HAT_ARCH["resi_connection"]
HAT_UPSAMPLER = _OFFICIAL_HAT_ARCH["upsampler"]


# =============================================================================
# HAT Model Loading
# =============================================================================

def _hat_arch_for_model(model_name: str) -> dict:
    """Return architecture kwargs for a registered HAT model variant."""
    if model_name not in HAT_MODELS:
        logger.warning("Unknown HAT model '%s', using default arch", model_name)
        model_name = DEFAULT_HAT_MODEL
    return dict(HAT_MODELS[model_name].get("arch", _OFFICIAL_HAT_ARCH))


def _extract_hat_checkpoint(raw_checkpoint: dict) -> dict:
    """Normalize BasicSR/HuggingFace HAT checkpoints to a flat state dict."""
    if "params_ema" in raw_checkpoint:
        return raw_checkpoint["params_ema"]
    if "params" in raw_checkpoint:
        return raw_checkpoint["params"]
    if "state_dict" in raw_checkpoint:
        return raw_checkpoint["state_dict"]
    return raw_checkpoint


def _build_hat_model(arch: dict):
    """Instantiate HAT with architecture settings that match pretrained weights."""
    from processing.hat_models import HAT

    return HAT(
        upscale=arch["upscale"],
        img_range=arch["img_range"],
        embed_dim=arch["embed_dim"],
        num_heads=arch["num_heads"],
        depths=arch["depths"],
        window_size=arch["window_size"],
        compress_ratio=arch["compress_ratio"],
        squeeze_factor=arch["squeeze_factor"],
        conv_scale=arch["conv_scale"],
        overlap_ratio=arch["overlap_ratio"],
        mlp_ratio=arch["mlp_ratio"],
        resi_connection=arch["resi_connection"],
        upsampler=arch["upsampler"],
    )


def _load_hat_weights(model, raw_checkpoint: dict, model_name: str) -> None:
    """Load checkpoint weights and fail loudly on architecture mismatch."""
    state_dict = _extract_hat_checkpoint(raw_checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"HAT checkpoint incompatible with model architecture for {model_name}"
        ) from exc


def _load_hat_model_impl(
    model_name: str,
    gpu_id: int,
    model_path: Optional[str],
) -> Tuple:
    """Load HAT weights onto a single GPU (caller must hold _hat_load_lock)."""
    _set_cuda_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model_info = HAT_MODELS.get(model_name, HAT_MODELS[DEFAULT_HAT_MODEL])
    arch = _hat_arch_for_model(model_name)

    resolved_name = model_name
    resolved_path = model_path
    if resolved_path is None:
        if resolved_name not in HAT_MODELS:
            logger.warning("Unknown HAT model '%s', using default", resolved_name)
            resolved_name = DEFAULT_HAT_MODEL
            model_info = HAT_MODELS[resolved_name]
        resolved_path = model_info.get("url", "")

    logger.info(
        "Loading HAT model: %s (GPU %d, path=%s)",
        resolved_name,
        gpu_id,
        resolved_path,
    )

    model = _build_hat_model(arch)
    checkpoint_path: Optional[Path] = None

    if resolved_path and Path(resolved_path).exists():
        checkpoint_path = Path(resolved_path)
        logger.info("Loading HAT from local path: %s", checkpoint_path)
    elif resolved_path and resolved_path.startswith("http"):
        import urllib.request
        import urllib.error

        cache_dir = Path.home() / ".cache" / "faceoff_models" / "hat"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / Path(resolved_path).name

        if not checkpoint_path.exists():
            logger.info("Downloading HAT model from %s ...", resolved_path)
            try:
                with urllib.request.urlopen(resolved_path, timeout=120) as resp:
                    checkpoint_path.write_bytes(resp.read())
            except (urllib.error.URLError, OSError) as e:
                logger.error("Failed to download HAT model: %s", e)
                raise
        logger.info("Loading HAT from cached path: %s", checkpoint_path)
    else:
        raise ValueError("No valid HAT model path. Set a valid .pth path or URL.")

    raw_checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True, mmap=True
    )
    _load_hat_weights(model, raw_checkpoint, resolved_name)
    model.eval().to(device)
    mean = model.mean.cpu()
    img_range_val = model.img_range

    logger.info("HAT model loaded on %s", device)
    return (model.to(device), device, mean, img_range_val)


def _get_hat_model(
    model_name: str = DEFAULT_HAT_MODEL,
    gpu_id: int = 0,
    model_path: Optional[str] = None,
) -> Tuple:
    """
    Get or create a HAT model instance (cached per GPU).

    Model loading is serialized globally so multi-GPU workers do not race
    the CUDA driver during concurrent weight loads (common on WSL2).

    Args:
        model_name: Name of the HAT model variant
        gpu_id: GPU device ID
        model_path: Optional local .pth file path (bypasses download)

    Returns:
        Tuple of (model, device, mean, img_range)
    """
    cache_key = (model_name, gpu_id, model_path)
    cached = _hat_cache.get(cache_key)
    if cached is not None:
        return cached

    with _hat_load_lock:
        cached = _hat_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            model_tuple = _load_hat_model_impl(model_name, gpu_id, model_path)
            _hat_cache.put(cache_key, model_tuple)
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            return model_tuple
        except Exception as e:
            logger.error("Failed to load HAT model: %s", e, exc_info=True)
            raise


def preload_hat_models(
    device_ids: List[int],
    model_name: str = DEFAULT_HAT_MODEL,
    model_path: Optional[str] = None,
) -> None:
    """Warm HAT on each GPU sequentially before parallel frame inference."""
    if not device_ids:
        return
    logger.info(
        "Preloading HAT model %s on GPU(s) %s (sequential)",
        model_name,
        device_ids,
    )
    for gpu_id in device_ids:
        _get_hat_model(model_name, gpu_id, model_path)


def clear_hat_cache() -> None:
    """Clear the HAT model cache to free GPU memory."""
    global _hat_cache
    with _hat_load_lock:
        _hat_cache.clear()
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        gc.collect()
    logger.info("HAT model cache cleared")


# =============================================================================
# Core Inference
# =============================================================================

def _pad_to_window_size(
    image: np.ndarray,
    window_size: int = HAT_WINDOW_SIZE,
) -> Tuple[np.ndarray, int, int]:
    """Pad an HWC image so height and width are multiples of window_size."""
    h, w = image.shape[:2]
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return image, h, w


def _hat_tile_dim(size: int, tile_size: int, window_size: int = HAT_WINDOW_SIZE) -> int:
    """Minimum tile dimension that satisfies tile_size and window_size alignment."""
    target = max(size, tile_size)
    remainder = target % window_size
    if remainder:
        target += window_size - remainder
    return target


def _should_use_tiled_inference(
    height: int,
    width: int,
    tile_size: int,
    gpu_id: int = 0,
    *,
    force_tiled: bool = False,
) -> bool:
    """Decide whether HAT should use tiled inference to avoid OOM."""
    if tile_size <= 0:
        return False
    if force_tiled:
        return True
    if height * width > tile_size * tile_size * 4:
        return True
    stats = MemoryManager(gpu_id).get_memory_stats(use_cache=False)
    return stats["free_mb"] < config.hat_force_tiled_below_free_mb


def _oom_tile_attempts(tile_size: int) -> List[int]:
    """Tile sizes to try after CUDA OOM, largest first."""
    min_tile = max(64, config.hat_oom_min_tile_size)
    candidates = [tile_size]
    for size in (192, 128, 96, min_tile):
        if size < tile_size and size >= min_tile:
            candidates.append(size)
    if min_tile not in candidates:
        candidates.append(min_tile)
    return list(dict.fromkeys(candidates))


def _apply_hat_model(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    img_range: float,
    gpu_id: int = 0,
    tile_size: int = 256,
    *,
    force_tiled: bool = False,
) -> np.ndarray:
    """
    Run HAT inference on a single RGB image with optional tile-based processing.

    Args:
        image_rgb: Input image (H, W, C) uint8
        model: HAT nn.Module on GPU
        device: GPU device
        mean: Mean tensor from model (1, C, 1, 1)
        img_range: Image range from model (usually 1.0)
        gpu_id: GPU device ID (for VRAM profiling)
        tile_size: Tile size for overlapping patch processing

    Returns:
        Enhanced image (H, W, C) uint8
    """
    _set_cuda_device(gpu_id)
    h, w = image_rgb.shape[:2]
    window_size = int(getattr(model, "window_size", HAT_WINDOW_SIZE))

    if _should_use_tiled_inference(h, w, tile_size, gpu_id, force_tiled=force_tiled):
        return _enhance_image_tiled(
            image_rgb, model, device, mean, img_range, tile_size, window_size=window_size
        )

    # Direct inference (small images or tile_size=0)
    padded_rgb, orig_h, orig_w = _pad_to_window_size(image_rgb, window_size)

    # [0, 1] float RGB — HAT.forward() applies mean/img_range internally.
    img = padded_rgb.astype(np.float32) / 255.0

    # HWC -> CHW
    img = np.moveaxis(img, -1, 0)
    img = np.ascontiguousarray(img).astype(np.float32)

    # Add batch dim and move to GPU
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # Inference — FP32 only; HAT attention overflows to NaN under FP16 autocast.
    with torch.no_grad():
        output = model(img_tensor.float())

    # Model output is already denormalized to ~[0, 1]
    output = output.squeeze(0).float().cpu().numpy()
    output = np.clip(output, 0.0, 1.0)
    output = (output * 255.0).astype(np.uint8)

    # CHW -> HWC, then crop back to original spatial extent
    output = np.moveaxis(output, 0, -1)
    scale = HAT_UPSCALE
    output = output[: orig_h * scale, : orig_w * scale]
    return output


def _run_hat_tile(
    tile_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    img_range: float,
    tile_size: int,
    window_size: int,
) -> np.ndarray:
    """Run HAT on a single input tile; returns HWC uint8 RGB at 4x scale."""
    target_h = _hat_tile_dim(tile_rgb.shape[0], tile_size, window_size)
    target_w = _hat_tile_dim(tile_rgb.shape[1], tile_size, window_size)
    tile_in = tile_rgb.astype(np.float32)
    if tile_in.shape[0] < target_h or tile_in.shape[1] < target_w:
        pad_h = max(0, target_h - tile_in.shape[0])
        pad_w = max(0, target_w - tile_in.shape[1])
        tile_in = np.pad(tile_in, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    tile = tile_in / 255.0
    tile = np.moveaxis(tile, -1, 0)
    tile = np.ascontiguousarray(tile).astype(np.float32)
    tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tile = model(tile_tensor.float())

    out_tile = out_tile.squeeze(0).float().cpu().numpy()
    out_tile = np.clip(out_tile, 0.0, 1.0) * 255.0
    out_tile = out_tile.astype(np.float32)
    out_tile = np.moveaxis(out_tile, 0, -1)

    orig_th, orig_tw = tile_rgb.shape[0], tile_rgb.shape[1]
    return out_tile[: orig_th * HAT_UPSCALE, : orig_tw * HAT_UPSCALE]


def _enhance_image_tiled(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    img_range: float,
    tile_size: int,
    overlap: int = 16,
    window_size: int = HAT_WINDOW_SIZE,
) -> np.ndarray:
    """
    Tile-based HAT inference for large images to manage VRAM.

    Iterates in input space, blends overlapping 4x outputs into the full frame.
    """
    h, w = image_rgb.shape[:2]
    scale = HAT_UPSCALE
    out_h, out_w = h * scale, w * scale

    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)

    stride = max(1, tile_size - overlap)

    y = 0
    while y < h:
        x = 0
        while x < w:
            y0, x0 = y, x
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile_in = image_rgb[y0:y1, x0:x1].astype(np.float32)

            out_tile = _run_hat_tile(
                tile_in, model, device, mean, img_range, tile_size, window_size
            )

            oy0, ox0 = y0 * scale, x0 * scale
            oh, ow = out_tile.shape[0], out_tile.shape[1]
            oy1, ox1 = min(oy0 + oh, out_h), min(ox0 + ow, out_w)
            th, tw = oy1 - oy0, ox1 - ox0

            output[oy0:oy1, ox0:ox1] += out_tile[:th, :tw]
            weight_map[oy0:oy1, ox0:ox1] += 1.0

            if x1 >= w:
                break
            x += stride
            if x + tile_size > w:
                x = max(0, w - tile_size)

        if y1 >= h:
            break
        y += stride
        if y + tile_size > h:
            y = max(0, h - tile_size)

    weight_map = np.maximum(weight_map, 1e-6)
    output = output / weight_map
    return np.clip(output, 0, 255).astype(np.uint8)


# =============================================================================
# Single Image Enhancement
# =============================================================================

def enhance_image_hat(
    img: np.ndarray,
    model_name: str = DEFAULT_HAT_MODEL,
    gpu_id: int = 0,
    tile_size: int = 256,
    model_path: Optional[str] = None,
    clear_cache: bool = True,
) -> Optional[np.ndarray]:
    """
    Enhance image using HAT.

    Args:
        img: Input image (BGR format, numpy array)
        model_name: HAT model variant name
        gpu_id: GPU device ID
        tile_size: Tile size for processing (0 = auto)
        model_path: Optional local .pth file path

    Returns:
        Enhanced image (BGR format) or None if failed
    """
    if clear_cache:
        refresh_gpu_memory(gpu_id, force=True)
    model, device, mean, img_range = _get_hat_model(model_name, gpu_id, model_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    last_error: Optional[Exception] = None
    for attempt_tile in _oom_tile_attempts(tile_size):
        try:
            force_tiled = attempt_tile < tile_size
            enhanced_rgb = _apply_hat_model(
                img_rgb,
                model,
                device,
                mean,
                img_range,
                gpu_id=gpu_id,
                tile_size=attempt_tile,
                force_tiled=force_tiled,
            )
            result = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
            if attempt_tile != tile_size:
                logger.info(
                    "HAT succeeded with reduced tile size %d (requested %d)",
                    attempt_tile,
                    tile_size,
                )
            logger.debug("HAT enhancement complete: %s -> %s", img.shape, result.shape)
            return result
        except torch.cuda.OutOfMemoryError as oom_error:
            last_error = oom_error
            logger.warning(
                "HAT OOM at tile=%d on GPU %d — refreshing memory and retrying smaller",
                attempt_tile,
                gpu_id,
            )
            refresh_gpu_memory(gpu_id, force=True)
        except Exception as e:
            last_error = e
            logger.error("HAT enhancement failed: %s", e, exc_info=True)
            break

    if last_error is not None:
        logger.error("HAT enhancement failed after retries: %s", last_error)
    return None


def enhance_image_batch_hat(
    images: List[np.ndarray],
    model_name: str = DEFAULT_HAT_MODEL,
    gpu_id: int = 0,
    tile_size: int = 256,
    batch_size: int = 4,
    model_path: Optional[str] = None,
    clear_cache: bool = True,
) -> List[np.ndarray]:
    """
    Enhance a batch of images using HAT with sequential processing.

    Args:
        images: List of BGR numpy arrays
        model_name: HAT model variant
        gpu_id: GPU device ID
        tile_size: Tile size for processing
        batch_size: Number of images to process in sequence (for VRAM management)
        model_path: Optional local .pth file path

    Returns:
        List of enhanced BGR numpy arrays (None for failed frames)
    """
    results = []
    _set_cuda_device(gpu_id)
    if clear_cache:
        refresh_gpu_memory(gpu_id, force=True)
    model, device, mean, img_range = _get_hat_model(model_name, gpu_id, model_path)

    with _hat_gpu_lock(gpu_id):
        for img in images:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                enhanced_rgb = _apply_hat_model(
                    img_rgb, model, device, mean, img_range, gpu_id, tile_size
                )
                results.append(cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logger.warning("HAT failed for frame: %s", e)
                results.append(None)
        if torch.cuda.is_available():
            torch.cuda.synchronize(gpu_id)

    return results


def get_hat_scale(model_name: str) -> int:
    """Get the upscaling factor for a HAT model."""
    if model_name in HAT_MODELS:
        return HAT_MODELS[model_name]["scale"]
    return 4


def get_available_hat_models() -> dict:
    """Get dictionary of available HAT models."""
    return HAT_MODELS.copy()
