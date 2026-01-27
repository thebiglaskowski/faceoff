"""
Image enhancement using SwinIR/Swin2SR transformer models.

This module provides Swin2SR integration via Hugging Face Transformers
as an alternative to Real-ESRGAN for image super-resolution.

Performance optimizations (in order of preference):
1. torch.compile() - PyTorch 2.0+ with Triton (Linux only, ~30-50% speedup)
2. BetterTransformer - HuggingFace optimum (cross-platform, ~20-40% speedup)
3. SDPA (Scaled Dot Product Attention) - PyTorch 2.0+ native (cross-platform)
4. FP16 autocast - Always enabled on CUDA (reduced memory, faster)
"""
import cv2
import gc
import logging
import numpy as np
import sys
import torch
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple, Union

from utils.lru_cache import LRUModelCache

logger = logging.getLogger("FaceOff")


# =============================================================================
# Optimization Availability Checks
# =============================================================================

def _check_torch_compile_available() -> bool:
    """Check if torch.compile() is available and functional (requires Triton on Linux)."""
    if not hasattr(torch, 'compile') or torch.__version__ < '2.0':
        return False
    # Check for Triton (required for inductor backend on CUDA)
    try:
        import triton
        return True
    except ImportError:
        return False


def _check_bettertransformer_available() -> bool:
    """Check if BetterTransformer from optimum is available."""
    try:
        from optimum.bettertransformer import BetterTransformer
        return True
    except ImportError as e:
        logger.debug("BetterTransformer import failed: %s", e)
        return False
    except Exception as e:
        logger.debug("BetterTransformer check failed: %s", e)
        return False


def _check_sdpa_available() -> bool:
    """Check if PyTorch SDPA (Scaled Dot Product Attention) is available."""
    # SDPA is available in PyTorch 2.0+ and provides Flash Attention when possible
    return hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch.__version__ >= '2.0'


# Check available optimizations at module load
TORCH_COMPILE_AVAILABLE = _check_torch_compile_available()
BETTERTRANSFORMER_AVAILABLE = _check_bettertransformer_available()
SDPA_AVAILABLE = _check_sdpa_available()

# Log optimization availability
_opt_status = []
if TORCH_COMPILE_AVAILABLE:
    _opt_status.append("torch.compile")
if BETTERTRANSFORMER_AVAILABLE:
    _opt_status.append("BetterTransformer")
if SDPA_AVAILABLE:
    _opt_status.append("SDPA")
logger.debug("SwinIR optimizations available: %s", ", ".join(_opt_status) if _opt_status else "FP16 only")


def _cleanup_swinir_model(model_tuple):
    """Cleanup function for evicted SwinIR models."""
    try:
        processor, model, device = model_tuple
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# LRU cache for SwinIR models (bounded to prevent memory growth)
_swinir_cache = LRUModelCache("SwinIR", cleanup_fn=_cleanup_swinir_model)

# Available Swin2SR models from Hugging Face
SWINIR_MODELS = {
    "Swin2SR_x2": {
        "model_id": "caidas/swin2SR-classical-sr-x2-64",
        "scale": 2,
        "description": "Classical SR 2x - Fast, good for clean images",
    },
    "Swin2SR_x4": {
        "model_id": "caidas/swin2SR-classical-sr-x4-64",
        "scale": 4,
        "description": "Classical SR 4x - Standard upscaling",
    },
    "Swin2SR_RealWorld_x4": {
        "model_id": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
        "scale": 4,
        "description": "Real-world SR 4x - Best for degraded images (recommended)",
    },
    "Swin2SR_Compressed_x4": {
        "model_id": "caidas/swin2SR-compressed-sr-x4-48",
        "scale": 4,
        "description": "Compressed SR 4x - For JPEG artifacts",
    },
}

DEFAULT_SWINIR_MODEL = "Swin2SR_RealWorld_x4"


def _get_swinir_model(
    model_name: str = DEFAULT_SWINIR_MODEL,
    gpu_id: int = 0
) -> Tuple:
    """
    Get or create Swin2SR model instance (cached).

    Args:
        model_name: Name of the Swin2SR model variant
        gpu_id: GPU device ID

    Returns:
        Tuple of (processor, model)
    """
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    cache_key = (model_name, gpu_id)

    cached = _swinir_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get model ID from config
    if model_name not in SWINIR_MODELS:
        logger.warning("Unknown Swin2SR model '%s', using default", model_name)
        model_name = DEFAULT_SWINIR_MODEL

    model_id = SWINIR_MODELS[model_name]["model_id"]

    logger.info("Loading Swin2SR model: %s (ID: %s) on GPU %d", model_name, model_id, gpu_id)

    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id)

    # Move to GPU
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Apply optimizations in order of preference
    optimization_applied = None

    # Option 1: torch.compile() - Best performance but requires Triton (Linux only)
    if TORCH_COMPILE_AVAILABLE and device.type == 'cuda':
        try:
            logger.info("Applying torch.compile() optimization (Triton backend)...")
            model = torch.compile(model, mode='reduce-overhead')
            optimization_applied = "torch.compile"
            logger.info("torch.compile() applied - expect ~30-50%% speedup after warmup")
        except Exception as e:
            logger.warning("torch.compile() failed: %s", e)

    # Option 2: BetterTransformer - Good cross-platform optimization
    # Try the native HF method first (model.to_bettertransformer()), then fall back to optimum
    if optimization_applied is None and device.type == 'cuda':
        # Method 2a: Native HuggingFace to_bettertransformer() (preferred)
        if hasattr(model, 'to_bettertransformer'):
            try:
                logger.info("Applying BetterTransformer optimization (native HF method)...")
                model = model.to_bettertransformer()
                optimization_applied = "BetterTransformer"
                logger.info("BetterTransformer applied - expect ~20-40%% speedup")
            except Exception as e:
                logger.info("BetterTransformer not compatible with Swin2SR model: %s", type(e).__name__)
                logger.debug("BetterTransformer error details: %s", e)
        else:
            logger.debug("Model does not have to_bettertransformer() method")

        # Method 2b: Optimum BetterTransformer (fallback)
        if optimization_applied is None and BETTERTRANSFORMER_AVAILABLE:
            try:
                from optimum.bettertransformer import BetterTransformer
                logger.info("Applying BetterTransformer optimization (optimum library)...")
                model = BetterTransformer.transform(model)
                optimization_applied = "BetterTransformer"
                logger.info("BetterTransformer applied - expect ~20-40%% speedup")
            except Exception as e:
                logger.debug("Optimum BetterTransformer failed: %s", e)

    # Option 3: Enable SDPA (Scaled Dot Product Attention) for Flash Attention
    if SDPA_AVAILABLE and device.type == 'cuda':
        try:
            # Enable Flash Attention via SDPA if available
            # This is automatic in PyTorch 2.0+ for supported models
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            if optimization_applied is None:
                optimization_applied = "SDPA"
                logger.info("SDPA (Flash Attention) enabled for faster attention computation")
            else:
                logger.debug("SDPA enabled alongside %s", optimization_applied)
        except Exception as e:
            logger.debug("SDPA configuration failed (non-critical): %s", e)

    # Log final optimization status
    if optimization_applied is None and device.type == 'cuda':
        logger.info("Using FP16 autocast only (install 'optimum' for BetterTransformer: pip install optimum)")

    # Cache the model
    model_tuple = (processor, model, device)
    _swinir_cache.put(cache_key, model_tuple)

    logger.info("Swin2SR model loaded successfully")
    return model_tuple


def clear_swinir_cache() -> None:
    """Clear the Swin2SR model cache to free GPU memory."""
    global _swinir_cache
    _swinir_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Swin2SR model cache cleared")


def get_swinir_scale(model_name: str) -> int:
    """Get the upscaling factor for a Swin2SR model."""
    if model_name in SWINIR_MODELS:
        return SWINIR_MODELS[model_name]["scale"]
    return 4  # Default


def enhance_image_swinir(
    img: np.ndarray,
    model_name: str = DEFAULT_SWINIR_MODEL,
    gpu_id: int = 0,
    tile_size: int = 0,  # Not used by Swin2SR but kept for API compatibility
) -> Optional[np.ndarray]:
    """
    Enhance image using Swin2SR.

    Args:
        img: Input image (BGR format, numpy array)
        model_name: Swin2SR model variant name
        gpu_id: GPU device ID
        tile_size: Tile size (unused, for API compatibility)

    Returns:
        Enhanced image (BGR format) or None if failed
    """
    try:
        # Clear some GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get model
        processor, model, device = _get_swinir_model(model_name, gpu_id)

        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Process image
        inputs = processor(pil_image, return_tensors="pt").to(device)

        # Inference with FP16 autocast for speed (if on CUDA)
        with torch.no_grad():
            if device.type == 'cuda':
                # FP16 inference - faster and lower memory
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        # Post-process output
        output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)

        # Convert RGB back to BGR
        result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        logger.debug("Swin2SR enhancement complete: %s -> %s", img.shape, result.shape)
        return result

    except Exception as e:
        logger.error("Swin2SR enhancement failed: %s", e, exc_info=True)
        return None


def enhance_image_swinir_file(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = DEFAULT_SWINIR_MODEL,
    gpu_id: int = 0,
) -> Optional[Path]:
    """
    Enhance an image file using Swin2SR.

    This function replaces the original file with the enhanced version,
    matching the behavior of RealESRGAN's enhance_image_single_gpu.

    Args:
        input_path: Path to input image
        output_dir: Output directory
        model_name: Swin2SR model variant
        gpu_id: GPU device ID

    Returns:
        Path to enhanced image (same as input_path) or None if failed
    """
    import shutil
    import time

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to read image: %s", input_path)
        return None

    # Enhance
    enhanced = enhance_image_swinir(img, model_name, gpu_id)
    if enhanced is None:
        return None

    # Save to temporary file first
    temp_output_path = output_dir / f"{input_path.stem}_swinir_temp{input_path.suffix}"
    cv2.imwrite(str(temp_output_path), enhanced)

    # Replace original with enhanced version (with retry for file locking)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Delete original file first if it exists
            if input_path.exists():
                input_path.unlink()
            time.sleep(0.1)  # Brief delay to ensure file is released
            shutil.move(str(temp_output_path), str(input_path))
            logger.info("Swin2SR enhanced image saved: %s", input_path)
            return input_path
        except (PermissionError, FileExistsError) as e:
            if attempt < max_retries - 1:
                logger.warning("File locked, retrying... (attempt %d/%d)", attempt + 1, max_retries)
                time.sleep(0.5)
            else:
                logger.error("Failed to move enhanced file after %d attempts: %s", max_retries, e)
                # Clean up temp file
                if temp_output_path.exists():
                    temp_output_path.unlink()
                return None

    return None


def enhance_frames_swinir(
    frames_dir: Union[str, Path],
    output_dir: Union[str, Path],
    media_type: str = "video",
    fps: Optional[float] = None,
    audio=None,
    model_name: str = DEFAULT_SWINIR_MODEL,
    gpu_id: int = 0,
    frame_durations: Optional[List[int]] = None,
) -> Optional[Union[Tuple[List, any], List[Image.Image]]]:
    """
    Enhance frames using Swin2SR.

    Args:
        frames_dir: Directory containing frames
        output_dir: Output directory
        media_type: "video" or "gif"
        fps: Frames per second (for video)
        audio: Audio track (for video)
        model_name: Swin2SR model variant
        gpu_id: GPU device ID
        frame_durations: Frame durations for GIF

    Returns:
        For video: Tuple of (enhanced_frames, video_clip)
        For GIF: List of PIL Images
        None if failed
    """
    from moviepy.editor import ImageSequenceClip
    from tqdm import tqdm

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get frame files
    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        logger.error("No frames found in %s", frames_dir)
        return None

    logger.info("Enhancing %d frames with Swin2SR (model=%s, GPU=%d)",
                len(frame_files), model_name, gpu_id)

    enhanced_frames = []
    enhanced_pil_frames = []

    # Process frames
    for frame_path in tqdm(frame_files, desc="Swin2SR Enhancement", unit="frame"):
        # Read frame
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to read frame: %s", frame_path)
            continue

        # Enhance frame
        enhanced = enhance_image_swinir(img, model_name, gpu_id)
        if enhanced is None:
            # Fall back to original frame
            logger.warning("Enhancement failed for frame %s, using original", frame_path.name)
            enhanced = img

        # Save enhanced frame
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), enhanced)

        # Convert for return
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        enhanced_frames.append(enhanced_rgb)
        enhanced_pil_frames.append(Image.fromarray(enhanced_rgb))

    if not enhanced_frames:
        logger.error("No frames were enhanced")
        return None

    # Return based on media type
    if media_type == "gif":
        logger.info("Swin2SR GIF enhancement complete: %d frames", len(enhanced_pil_frames))
        return enhanced_pil_frames
    else:
        # Create video clip
        if fps is None:
            fps = 30.0

        clip = ImageSequenceClip([np.array(f) for f in enhanced_frames], fps=fps)

        if audio is not None:
            clip = clip.set_audio(audio)

        logger.info("Swin2SR video enhancement complete: %d frames @ %.2f fps",
                    len(enhanced_frames), fps)
        return enhanced_frames, clip


def get_available_swinir_models() -> dict:
    """Get dictionary of available Swin2SR models."""
    return SWINIR_MODELS.copy()
