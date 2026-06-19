"""
HAT (Hybrid Attention Transformer) super-resolution enhancement module.

Provides HAT integration as an alternative enhancement backend alongside
Real-ESRGAN and SwinIR, following the same API pattern.

Uses the BasicSR-based HAT architecture with .pth weight files.
Download URLs:
  HAT-Base (4x): https://huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain/resolve/main/HAT-L_SRx4_ImageNet-pretrain.pth
  HAT (4x) GAN:  https://huggingface.co/anchuang/Real_HAT_GAN_SRx4_sharper/resolve/main/Real_HAT_GAN_sharper.pth
"""
import gc
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from utils.lru_cache import LRUModelCache

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

HAT_MODELS = {
    "HAT_Base_4x_ImageNet": {
        "url": "https://huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain/resolve/main/HAT-L_SRx4_ImageNet-pretrain.pth",
        "scale": 4,
        "description": "HAT Base 4x pre-trained on ImageNet — strong general SR",
    },
    "HAT_Base_4x_GAN_sharper": {
        "url": "https://huggingface.co/anchuang/Real_HAT_GAN_SRx4_sharper/resolve/main/Real_HAT_GAN_sharper.pth",
        "scale": 4,
        "description": "HAT GAN 4x — sharper perceptual quality for real-world SR",
    },
}

DEFAULT_HAT_MODEL = "HAT_Base_4x_ImageNet"

# HAT architecture defaults — Base uses 96 embed_dim, 6 layers
HAT_EMBED_DIM = 96
HAT_NUM_HEADS = (6, 6, 6, 6)
HAT_DEPTH = (6, 6, 6, 6)
HAT_UPSCALE = 4
HAT_IMG_RANGE = 1.0
HAT_WINDOW_SIZE = 7
HAT_COMPRESS_RATIO = 3
HAT_SQUEEZE_FACTOR = 30
HAT_CONVS_SCALE = 0.01
HAT_OVERLAP_RATIO = 0.5
HAT_MLP_RATIO = 4.0
HAT_RESI_CONNECTION = "1conv"
HAT_UPSAMPLER = "pixelshuffle"


# =============================================================================
# HAT Model Loading
# =============================================================================

def _get_hat_model(
    model_name: str = DEFAULT_HAT_MODEL,
    gpu_id: int = 0,
    model_path: Optional[str] = None,
) -> Tuple:
    """
    Get or create a HAT model instance (cached per GPU).

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

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Import the HAT architecture — registers with basicsr
    from processing.hat_models import HAT

    # Determine model path
    if model_path is None:
        if model_name not in HAT_MODELS:
            logger.warning("Unknown HAT model '%s', using default", model_name)
            model_name = DEFAULT_HAT_MODEL
        model_path = HAT_MODELS[model_name].get("url", "")

    logger.info("Loading HAT model: %s (GPU %d, path=%s)", model_name, gpu_id, model_path)

    try:
        # If a local path exists, load from it
        if model_path and Path(model_path).exists():
            logger.info("Loading HAT from local path: %s", model_path)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model = HAT(
                upscale=HAT_UPSCALE,
                img_range=HAT_IMG_RANGE,
                embed_dim=HAT_EMBED_DIM,
                num_heads=HAT_NUM_HEADS,
                depth=HAT_DEPTH,
                window_size=HAT_WINDOW_SIZE,
                compress_ratio=HAT_COMPRESS_RATIO,
                squeeze_factor=HAT_SQUEEZE_FACTOR,
                conv_scale=HAT_CONVS_SCALE,
                overlap_ratio=HAT_OVERLAP_RATIO,
                mlp_ratio=HAT_MLP_RATIO,
                resi_connection=HAT_RESI_CONNECTION,
                upsampler=HAT_UPSAMPLER,
            )
            model.load_state_dict(state_dict, strict=False)
            model.eval().to(device)
            mean = model.mean.cpu()
            img_range_val = model.img_range
        else:
            # Build a default HAT model (expects state_dict keys on disk)
            model = HAT(
                upscale=HAT_UPSCALE,
                img_range=HAT_IMG_RANGE,
                embed_dim=HAT_EMBED_DIM,
                num_heads=HAT_NUM_HEADS,
                depth=HAT_DEPTH,
                window_size=HAT_WINDOW_SIZE,
                compress_ratio=HAT_COMPRESS_RATIO,
                squeeze_factor=HAT_SQUEEZE_FACTOR,
                conv_scale=HAT_CONVS_SCALE,
                overlap_ratio=HAT_OVERLAP_RATIO,
                mlp_ratio=HAT_MLP_RATIO,
                resi_connection=HAT_RESI_CONNECTION,
                upsampler=HAT_UPSAMPLER,
            )

            # Download and load from URL
            if model_path and model_path.startswith("http"):
                try:
                    import urllib.request
                    import urllib.error
                    cache_dir = Path.home() / ".cache" / "faceoff_models" / "hat"
                    cache_dir.mkdir(parents=True, exist_ok=True)

                    fname = Path(model_path).name
                    cache_path = cache_dir / fname

                    if not cache_path.exists():
                        logger.info("Downloading HAT model from %s ...", model_path)
                        try:
                            with urllib.request.urlopen(model_path, timeout=30) as resp:
                                cache_path.write_bytes(resp.read())
                        except (urllib.error.URLError, OSError) as e:
                            logger.error("Failed to download HAT model: %s", e)
                            raise

                    logger.info("Loading HAT from cached path: %s", cache_path)
                    state_dict = torch.load(str(cache_path), map_location=device, weights_only=True, mmap=True)
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    model.load_state_dict(state_dict, strict=False)
                    model.eval().to(device)

                except Exception as e:
                    logger.error("Failed to download/load HAT model: %s", e)
                    raise
            else:
                logger.error("No valid HAT model path provided. Cannot enhance.")
                raise ValueError("No valid HAT model path. Set a valid .pth path or URL.")

            mean = model.mean.cpu()
            img_range_val = model.img_range

        model.eval()
        model.to(device)

        logger.info("HAT model loaded on %s", device)
        model_tuple = (model.to(device), device, mean, img_range_val)
        _hat_cache.put(cache_key, model_tuple)
        return model_tuple

    except Exception as e:
        logger.error("Failed to load HAT model: %s", e, exc_info=True)
        raise


def clear_hat_cache() -> None:
    """Clear the HAT model cache to free GPU memory."""
    global _hat_cache
    _hat_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("HAT model cache cleared")


# =============================================================================
# Core Inference
# =============================================================================

def _apply_hat_model(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    img_range: float,
    gpu_id: int = 0,
    tile_size: int = 256,
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
    h, w = image_rgb.shape[:2]

    # For large images, use tile-based processing
    if h * w > tile_size * tile_size * 4 and tile_size > 0:
        return _enhance_image_tiled(image_rgb, model, device, mean, img_range, tile_size)

    # Direct inference (small images or tile_size=0)
    # Normalize to [0, 1]
    img = image_rgb.astype(np.float32) / 255.0 * img_range

    # Mean normalization
    img = img - mean.cpu().numpy().squeeze()

    # HWC -> CHW
    img = np.moveaxis(img, -1, 0)
    img = np.ascontiguousarray(img).astype(np.float32)

    # Add batch dim and move to GPU
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(img_tensor)
        else:
            output = model(img_tensor)

    # Denormalize
    output = output.squeeze(0).cpu().numpy()
    output = np.clip(output, 0, img_range)
    output = (output / img_range) * 255.0
    output = np.clip(output, 0, 255).astype(np.uint8)

    # CHW -> HWC
    output = np.moveaxis(output, 0, -1)
    return output


def _enhance_image_tiled(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    img_range: float,
    tile_size: int,
    overlap: int = 16,
) -> np.ndarray:
    """
    Tile-based HAT inference for large images to manage VRAM.

    Args:
        image_rgb: Input image (H, W, C) uint8
        model: HAT nn.Module on GPU
        device: GPU device
        mean: Mean tensor from model (1, C, 1, 1)
        img_range: Image range from model (usually 1.0)
        tile_size: Tile size
        overlap: Overlap between tiles in pixels

    Returns:
        Enhanced image (H, W, C) uint8
    """
    h, w = image_rgb.shape[:2]
    scale = HAT_UPSCALE
    out_h, out_w = h * scale, w * scale

    # Initialize output buffer
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w), dtype=np.float32)

    # Normalize
    img_norm = image_rgb.astype(np.float32) / 255.0 * img_range
    img_norm = img_norm - mean.cpu().numpy().squeeze()

    stride = tile_size * scale - overlap * 2

    for y in range(0, h * scale, stride):
        for x in range(0, w * scale, stride):
            # Define tile region in output space
            y0, y1 = max(0, y), min(out_h, y + tile_size * scale)
            x0, x1 = max(0, x), min(out_w, x + tile_size * scale)

            # Source region in input space
            sy0, sy1 = int(y0 / scale), int(y1 / scale)
            sx0, sx1 = int(x0 / scale), int(x1 / scale)
            tile_in = image_rgb[sy0:sy1].astype(np.float32) if sy1 > sy0 else image_rgb.astype(np.float32)

            # Pad tile to tile_size if needed
            if tile_in.shape[0] < tile_size or tile_in.shape[1] < tile_size:
                pad_h = max(0, tile_size - tile_in.shape[0])
                pad_w = max(0, tile_size - tile_in.shape[1])
                tile_in = np.pad(tile_in, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

            # Normalize tile
            tile = tile_in.astype(np.float32) / 255.0 * img_range
            tile = tile - mean.cpu().numpy().squeeze()

            # HWC -> CHW
            tile = np.moveaxis(tile, -1, 0)
            tile = np.ascontiguousarray(tile).astype(np.float32)

            tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)

            with torch.no_grad():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out_tile = model(tile_tensor)
                else:
                    out_tile = model(tile_tensor)

            # Denormalize tile output
            out_tile = out_tile.squeeze(0).cpu().numpy()
            out_tile = np.clip(out_tile, 0, img_range)
            out_tile = (out_tile / img_range) * 255.0
            out_tile = np.clip(out_tile, 0, 255).astype(np.float32)
            out_tile = np.moveaxis(out_tile, 0, -1)

            # Place tile output in output buffer (accounting for source offset)
            oy0, oy1 = y0 - (sy0 * scale), y1 - (sy1 * scale)
            ox0, ox1 = x0 - (sx0 * scale), x1 - (sx1 * scale)

            # Only place the inner (non-overlap) region
            inner_oy0, inner_oy1 = oy0 + overlap, oy1 - overlap
            inner_ox0, inner_ox1 = ox0 + overlap, ox1 - overlap

            if inner_oy1 > inner_oy0 and inner_ox1 > inner_ox0:
                output[inner_oy0:inner_oy1, inner_ox0:inner_ox1] += out_tile[
                    overlap:out_tile.shape[0] - overlap, overlap:out_tile.shape[1] - overlap
                ]
                weight_map[inner_oy0:inner_oy1, inner_ox0:inner_ox1] += 1.0

    # Normalize by weight map
    np.divide(output, weight_map, out=output, where=weight_map > 0)
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
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, device, mean, img_range = _get_hat_model(model_name, gpu_id, model_path)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run HAT inference
        enhanced_rgb = _apply_hat_model(
            img_rgb, model, device, mean, img_range,
            gpu_id=gpu_id, tile_size=tile_size,
        )

        # RGB back to BGR
        result = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

        logger.debug("HAT enhancement complete: %s -> %s", img.shape, result.shape)
        return result

    except Exception as e:
        logger.error("HAT enhancement failed: %s", e, exc_info=True)
        return None


def enhance_image_hat_file(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = DEFAULT_HAT_MODEL,
    gpu_id: int = 0,
    tile_size: int = 256,
    model_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Enhance an image file using HAT.

    This function replaces the original file with the enhanced version,
    matching the behavior of Real-ESRGAN's enhance_image_single_gpu.

    Args:
        input_path: Path to input image
        output_dir: Output directory
        model_name: HAT model variant
        gpu_id: GPU device ID
        tile_size: Tile size for processing
        model_path: Optional local .pth file path

    Returns:
        Path to enhanced image (same as input_path) or None if failed
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to read image: %s", input_path)
        return None

    # Enhance
    enhanced = enhance_image_hat(img, model_name, gpu_id, tile_size, model_path)
    if enhanced is None:
        return None

    # Save to temporary file first
    temp_output_path = output_dir / f"{input_path.stem}_hat_temp{input_path.suffix}"
    cv2.imwrite(str(temp_output_path), enhanced)

    # Replace original with enhanced version (with retry for file locking)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if input_path.exists():
                input_path.unlink()
            time.sleep(0.1)
            shutil.move(str(temp_output_path), str(input_path))
            logger.info("HAT enhanced image saved: %s", input_path)
            return input_path
        except (PermissionError, FileExistsError) as e:
            if attempt < max_retries - 1:
                logger.warning("File locked, retrying... (attempt %d/%d)", attempt + 1, max_retries)
                time.sleep(0.5)
            else:
                logger.error("Failed to move enhanced file after %d attempts: %s", max_retries, e)
                if temp_output_path.exists():
                    temp_output_path.unlink()
                return None

    return None


# =============================================================================
# Frame Batching
# =============================================================================

def enhance_image_batch_hat(
    images: List[np.ndarray],
    model_name: str = DEFAULT_HAT_MODEL,
    gpu_id: int = 0,
    tile_size: int = 256,
    batch_size: int = 4,
    model_path: Optional[str] = None,
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
    model, device, mean, img_range = _get_hat_model(model_name, gpu_id, model_path)

    for img in images:
        try:
            enhanced = _apply_hat_model(img, model, device, mean, img_range, gpu_id, tile_size)
            results.append(enhanced)
        except Exception as e:
            logger.warning("HAT failed for frame: %s", e)
            results.append(None)

    return results


# =============================================================================
# Video/GIF Frame Enhancement
# =============================================================================

def enhance_frames_hat(
    frames_dir: Union[str, Path],
    output_dir: Union[str, Path],
    media_type: str = "video",
    fps: Optional[float] = None,
    audio=None,
    model_name: str = DEFAULT_HAT_MODEL,
    gpu_id: int = 0,
    tile_size: int = 256,
    model_path: Optional[str] = None,
    frame_durations: Optional[List[int]] = None,
) -> Optional[Union[Tuple[List, str], List[Image.Image]]]:
    """
    Enhance frames using HAT.

    Args:
        frames_dir: Directory containing frames
        output_dir: Output directory
        media_type: "video" or "gif"
        fps: Frames per second (for video)
        audio: Audio track (for video)
        model_name: HAT model variant
        gpu_id: GPU device ID
        tile_size: Tile size for processing
        model_path: Optional local .pth file path
        frame_durations: Frame durations for GIF

    Returns:
        For video: Tuple of (enhanced_frames, output_path)
        For GIF: List of PIL Images
        None if failed
    """
    from tqdm import tqdm

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get frame files
    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        logger.error("No frames found in %s", frames_dir)
        return None

    logger.info("Enhancing %d frames with HAT (model=%s, GPU=%d, tile=%d)",
                len(frame_files), model_name, gpu_id, tile_size)

    enhanced_frames = []
    enhanced_pil_frames = []

    for frame_path in tqdm(frame_files, desc="HAT Enhancement", unit="frame"):
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to read frame: %s", frame_path)
            continue

        enhanced = enhance_image_hat(img, model_name, gpu_id, tile_size, model_path)
        if enhanced is None:
            logger.warning("HAT enhancement failed for frame %s, using original", frame_path.name)
            enhanced = img

        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), enhanced)

        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        enhanced_frames.append(enhanced_rgb)
        enhanced_pil_frames.append(Image.fromarray(enhanced_rgb))

    if not enhanced_frames:
        logger.error("No frames were enhanced")
        return None

    if media_type == "gif":
        logger.info("HAT GIF enhancement complete: %d frames", len(enhanced_pil_frames))
        return enhanced_pil_frames
    else:
        output_path = output_dir / f"enhanced_{Path(frames_dir).name}.mp4"
        from utils import video_io
        success = video_io.write_video_from_pil_frames(
            enhanced_pil_frames,
            str(output_path),
            fps=fps or 30.0,
            codec="libx264",
            preset="medium",
            crf=18,
        )
        if not success:
            logger.error("Failed to write HAT enhanced video to %s", output_path)
            return None

        logger.info("HAT video enhancement complete: %d frames @ %.2f fps -> %s",
                    len(enhanced_frames), fps or 30.0, output_path)
        return enhanced_frames, str(output_path)


# =============================================================================
# Caches & Helpers
# =============================================================================

def _hat_cache_cleanup(model_tuple):
    """Cleanup function for HAT model cache entries."""
    try:
        model = model_tuple[0]
        device = model_tuple[1]
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


_hat_cache = LRUModelCache("HAT", cleanup_fn=_hat_cache_cleanup)


def get_hat_scale(model_name: str) -> int:
    """Get the upscaling factor for a HAT model."""
    if model_name in HAT_MODELS:
        return HAT_MODELS[model_name]["scale"]
    return 4


def get_available_hat_models() -> dict:
    """Get dictionary of available HAT models."""
    return HAT_MODELS.copy()
