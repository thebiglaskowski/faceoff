"""
Image and video enhancement using Real-ESRGAN and GFPGAN.

This module provides direct Python API integration with Real-ESRGAN
for image/video upscaling and GFPGAN for face restoration.
"""
import cv2
import gc
import logging
import numpy as np
import os
import shutil
import torch
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union
from moviepy.editor import ImageSequenceClip

from utils.temp_manager import get_temp_manager
from utils.lru_cache import LRUModelCache

logger = logging.getLogger("FaceOff")


def _cleanup_upsampler(upsampler):
    """Cleanup function for evicted upsamplers."""
    try:
        del upsampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cleanup_face_enhancer(enhancer):
    """Cleanup function for evicted face enhancers."""
    try:
        del enhancer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Model name to URL mapping for auto-download
MODEL_URLS = {
    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRNet_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
    'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'realesr-general-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    'realesr-animevideov3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
}

# Model scale factors
MODEL_SCALES = {
    'RealESRGAN_x4plus': 4,
    'RealESRNet_x4plus': 4,
    'RealESRGAN_x4plus_anime_6B': 4,
    'RealESRGAN_x2plus': 2,
    'realesr-general-x4v3': 4,
    'realesr-animevideov3': 4,
}

# LRU caches for loaded models (bounded to prevent memory growth)
_upsampler_cache = LRUModelCache("RealESRGAN", cleanup_fn=_cleanup_upsampler)
_face_enhancer_cache = LRUModelCache("GFPGAN", cleanup_fn=_cleanup_face_enhancer)


def _get_upsampler(
    model_name: str = "RealESRGAN_x4plus",
    gpu_id: int = 0,
    tile_size: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    use_fp32: bool = False,
    denoise_strength: float = 0.5
):
    """
    Get or create a RealESRGANer upsampler instance.

    Caches instances to avoid repeated model loading.
    """
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # Create cache key
    cache_key = (model_name, gpu_id, tile_size, pre_pad, use_fp32)

    cached = _upsampler_cache.get(cache_key)
    if cached is not None:
        return cached

    # Determine model architecture based on model name
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name in ['realesr-general-x4v3', 'realesr-animevideov3']:
        # Use SRVGGNetCompact for these models
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
    else:
        # Default to x4plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_name = 'RealESRGAN_x4plus'

    # Get model URL
    model_url = MODEL_URLS.get(model_name, MODEL_URLS['RealESRGAN_x4plus'])

    # Set dni_weight for denoise models
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength is not None:
        dni_weight = denoise_strength
        # For realesr-general-x4v3, we need to load both denoise and no-denoise models
        # The dni_weight interpolates between them
        wdn_model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth'

    # Create upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_url,
        dni_weight=dni_weight,
        model=model,
        tile=tile_size,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not use_fp32,
        gpu_id=gpu_id
    )

    _upsampler_cache.put(cache_key, upsampler)
    logger.debug("Created RealESRGANer for model=%s, gpu=%d", model_name, gpu_id)

    return upsampler


def _get_face_enhancer(gpu_id: int = 0, use_fp32: bool = False):
    """
    Get or create a GFPGANer face enhancer instance.
    """
    from gfpgan import GFPGANer

    cache_key = (gpu_id, use_fp32)

    cached = _face_enhancer_cache.get(cache_key)
    if cached is not None:
        return cached

    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=1,  # We handle upscaling separately
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None  # We'll set this separately if needed
    )

    _face_enhancer_cache.put(cache_key, face_enhancer)
    logger.debug("Created GFPGANer for gpu=%d", gpu_id)

    return face_enhancer


def clear_enhancement_cache():
    """Clear cached upsampler and face enhancer instances."""
    _upsampler_cache.clear()
    _face_enhancer_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Enhancement model cache cleared")


def enhance_image(
    img: np.ndarray,
    tile_size: int = 256,
    outscale: int = 4,
    gpu_id: int = 0,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0,
    face_enhance: bool = True
) -> Optional[np.ndarray]:
    """
    Enhance a single image using Real-ESRGAN with optional GFPGAN face enhancement.

    Args:
        img: Input image as numpy array (BGR format from cv2)
        tile_size: Tile size for processing (0=no tiling, 128-512 for tiled)
        outscale: Output upscaling factor
        gpu_id: GPU device ID
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts
        face_enhance: Apply GFPGAN face enhancement

    Returns:
        Enhanced image as numpy array, or None if failed
    """
    try:
        # Clear GPU cache before enhancement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Get upsampler
        upsampler = _get_upsampler(
            model_name=model_name,
            gpu_id=gpu_id,
            tile_size=tile_size,
            pre_pad=pre_pad,
            use_fp32=use_fp32,
            denoise_strength=denoise_strength
        )

        # Get face enhancer if needed
        face_enhancer = None
        if face_enhance:
            try:
                face_enhancer = _get_face_enhancer(gpu_id=gpu_id, use_fp32=use_fp32)
                # Set the background upsampler for face enhancer
                face_enhancer.bg_upsampler = upsampler
            except Exception as e:
                logger.warning("Failed to load GFPGAN, continuing without face enhancement: %s", e)
                face_enhancer = None

        # Enhance the image
        if face_enhancer is not None:
            # Use GFPGAN with Real-ESRGAN as background upsampler
            _, _, output = face_enhancer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5
            )
        else:
            # Use Real-ESRGAN only
            output, _ = upsampler.enhance(img, outscale=outscale)

        return output

    except Exception as e:
        logger.error("Enhancement failed: %s", e, exc_info=True)
        return None


def apply_realesrgan(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    tile_size: int = 256,
    outscale: int = 4,
    gpu_id: int = 0,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0,
    face_enhance: bool = True
) -> bool:
    """
    Apply Real-ESRGAN enhancement to an image file or directory.

    Args:
        input_path: Path to input file or directory
        output_dir: Output directory for enhanced results
        tile_size: Tile size for processing (128-512, lower = less VRAM)
        outscale: Upscaling factor (2 or 4)
        gpu_id: GPU device ID
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts
        face_enhance: Apply GFPGAN face enhancement

    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Running Real-ESRGAN (model=%s) on GPU %d: tile=%d, outscale=%d, fp32=%s, pre_pad=%d",
                model_name, gpu_id, tile_size, outscale, use_fp32, pre_pad)

    if model_name == "realesr-general-x4v3":
        logger.info("Denoise strength: %.2f", denoise_strength)

    try:
        # Handle directory or single file
        if input_path.is_dir():
            image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
        else:
            image_files = [input_path]

        if not image_files:
            logger.error("No image files found in %s", input_path)
            return False

        # Get upsampler (cached)
        upsampler = _get_upsampler(
            model_name=model_name,
            gpu_id=gpu_id,
            tile_size=tile_size,
            pre_pad=pre_pad,
            use_fp32=use_fp32,
            denoise_strength=denoise_strength
        )

        # Get face enhancer if needed
        face_enhancer = None
        if face_enhance:
            try:
                face_enhancer = _get_face_enhancer(gpu_id=gpu_id, use_fp32=use_fp32)
                face_enhancer.bg_upsampler = upsampler
            except Exception as e:
                logger.warning("Failed to load GFPGAN, continuing without face enhancement: %s", e)

        # Process each image
        for img_path in image_files:
            try:
                # Read image
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.warning("Failed to read image: %s", img_path)
                    continue

                # Enhance
                if face_enhancer is not None:
                    _, _, output = face_enhancer.enhance(
                        img,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                        weight=0.5
                    )
                else:
                    output, _ = upsampler.enhance(img, outscale=outscale)

                # Save with _out suffix (matching original CLI behavior)
                output_path = output_dir / f"{img_path.stem}_out.png"
                cv2.imwrite(str(output_path), output)
                logger.debug("Enhanced: %s -> %s", img_path.name, output_path.name)

            except Exception as e:
                logger.error("Failed to enhance %s: %s", img_path, e)
                continue

        logger.info("✅ Enhancement complete on GPU %d", gpu_id)
        return True

    except Exception as e:
        logger.error("Enhancement failed on GPU %d: %s", gpu_id, e, exc_info=True)
        return False


# Backwards compatibility alias
apply_realesrgan_cli = apply_realesrgan


def enhance_image_single_gpu(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    tile_size: int = 256,
    outscale: int = 4,
    gpu_id: int = 0,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0
) -> Optional[Path]:
    """
    Enhance a single image file using Real-ESRGAN.

    Args:
        input_path: Path to input image
        output_dir: Output directory
        tile_size: Tile size for processing (128-512, lower = less VRAM)
        outscale: Upscaling factor (2 or 4)
        gpu_id: GPU device ID
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts

    Returns:
        Path to enhanced image, or None if failed
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not apply_realesrgan(input_path, output_dir, tile_size, outscale, gpu_id,
                            model_name, denoise_strength, use_fp32, pre_pad):
        return None

    # Real-ESRGAN saves with _out suffix
    enhanced_path = output_dir / f"{input_path.stem}_out.png"

    if enhanced_path.exists():
        # Replace original with enhanced version (with retry for file locking)
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Delete target file first if it exists
                if input_path.exists():
                    input_path.unlink()
                time.sleep(0.1)  # Brief delay to ensure file is released
                shutil.move(str(enhanced_path), str(input_path))
                logger.info("Enhanced image saved to %s", input_path)
                return input_path
            except (PermissionError, FileExistsError) as e:
                if attempt < max_retries - 1:
                    logger.warning("File locked, retrying... (attempt %d/%d)", attempt + 1, max_retries)
                    time.sleep(0.5)
                else:
                    logger.error("Failed to move enhanced file after %d attempts: %s", max_retries, e)
                    return None
    else:
        logger.warning("Enhanced image not found at %s", enhanced_path)
        return None


def enhance_frames_single_gpu(
    frames_dir: Union[str, Path],
    output_dir: Union[str, Path],
    media_type: str = "video",
    fps: Optional[float] = None,
    audio = None,
    tile_size: int = 256,
    outscale: int = 4,
    gpu_id: int = 0,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0,
    maintain_dimensions: bool = False,
    original_size: Optional[Tuple[int, int]] = None
) -> Optional[Union[Tuple[List, ImageSequenceClip], List[Image.Image]]]:
    """
    Enhance video or GIF frames using single GPU.

    Args:
        frames_dir: Directory containing frames
        output_dir: Output directory
        media_type: "video" or "gif"
        fps: Frames per second (for video)
        audio: Audio track (for video)
        tile_size: Tile size for processing (128-512, lower = less VRAM)
        outscale: Upscaling factor (2 or 4)
        gpu_id: GPU device ID
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts
        maintain_dimensions: Resize back to original dimensions after enhancement
        original_size: Original (width, height) if maintain_dimensions is True

    Returns:
        For video: (frames, enhanced_clip)
        For GIF: list of PIL Images
        None if failed
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    enhanced_dir = output_dir / f"temp_{media_type}_enhanced"

    # Remove old enhanced directory
    if enhanced_dir.exists():
        shutil.rmtree(enhanced_dir)

    # Run enhancement
    if not apply_realesrgan(frames_dir, enhanced_dir, tile_size, outscale, gpu_id,
                            model_name, denoise_strength, use_fp32, pre_pad):
        return None

    # Load enhanced frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    expected_frames = len(frame_files)
    enhanced_frames = []

    for i in range(expected_frames):
        # Frame naming differs between video and GIF
        if media_type == "video":
            enhanced_path = enhanced_dir / f"frame_{i:06d}_out.png"
        else:  # gif
            enhanced_path = enhanced_dir / f"frame_{i:04d}_out.png"

        if enhanced_path.exists():
            if media_type == "gif":
                enhanced_frames.append(Image.open(enhanced_path).copy())
            else:  # video
                enhanced_frames.append(np.array(Image.open(enhanced_path)))
        else:
            logger.error("Enhanced frame not found: %s", enhanced_path)
            return None

    # Handle frame count mismatch (Real-ESRGAN sometimes produces +/- 1-2 frames)
    actual_frames = len(enhanced_frames)
    frame_diff = abs(actual_frames - expected_frames)

    if frame_diff > 2:
        logger.error("Frame count mismatch too large. Expected %d, got %d", expected_frames, actual_frames)
        return None

    if frame_diff > 0:
        logger.warning("Frame count mismatch. Expected %d, got %d. Adjusting...", expected_frames, actual_frames)
        if actual_frames > expected_frames:
            enhanced_frames = enhanced_frames[:expected_frames]
        elif actual_frames < expected_frames:
            while len(enhanced_frames) < expected_frames:
                enhanced_frames.append(enhanced_frames[-1])

    # Resize back to original dimensions if requested
    if maintain_dimensions and original_size and outscale > 1:
        original_width, original_height = original_size
        logger.info("Resizing enhanced frames from %dx (scale=%dx) back to original %dx%d",
                   original_width * outscale, outscale, original_width, original_height)

        resized_frames = []
        for frame in enhanced_frames:
            if media_type == "gif":
                # PIL Image
                resized_frame = frame.resize((original_width, original_height), Image.Resampling.LANCZOS)
                resized_frames.append(resized_frame)
            else:  # video
                # NumPy array
                frame_img = Image.fromarray(frame)
                resized_frame = frame_img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                resized_frames.append(np.array(resized_frame))

        enhanced_frames = resized_frames
        logger.info("✅ Resized %d frames to original dimensions", len(enhanced_frames))

    # Return format depends on media type
    if media_type == "video":
        enhanced_clip = ImageSequenceClip(enhanced_frames, fps=fps)
        if audio:
            enhanced_clip = enhanced_clip.set_audio(audio)
        logger.info("Enhanced video: %d frames", len(enhanced_frames))
        return enhanced_frames, enhanced_clip
    else:  # gif
        logger.info("Enhanced GIF: %d frames", len(enhanced_frames))
        return enhanced_frames


def enhance_frames_multi_gpu(
    frame_paths: List[Path],
    output_dir: Path,
    media_type: str,
    device_ids: List[int],
    fps: Optional[float] = None,
    audio = None,
    frame_durations: Optional[List[int]] = None,
    tile_size: int = 256,
    outscale: int = 4,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0
) -> Optional[Union[Tuple[List, ImageSequenceClip], List[Image.Image]]]:
    """
    Enhance frames using multiple GPUs for parallel processing.

    Args:
        frame_paths: List of frame paths to enhance
        output_dir: Output directory
        media_type: "video" or "gif"
        device_ids: List of GPU device IDs to use
        fps: Frames per second (for video)
        audio: Audio track (for video)
        frame_durations: Frame durations (for GIF)
        tile_size: Tile size for processing (128-512, lower = less VRAM)
        outscale: Upscaling factor (2 or 4)
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts

    Returns:
        For video: (frames, enhanced_clip)
        For GIF: list of PIL Images
        None if failed
    """
    if len(device_ids) <= 1:
        # Fall back to single GPU
        return enhance_frames_single_gpu(
            frames_dir=frame_paths[0].parent,
            output_dir=output_dir,
            media_type=media_type,
            fps=fps,
            audio=audio,
            tile_size=tile_size,
            outscale=outscale,
            gpu_id=device_ids[0] if device_ids else 0,
            model_name=model_name,
            denoise_strength=denoise_strength
        )

    logger.info("Multi-GPU Enhancement: %d frames across %d GPUs", len(frame_paths), len(device_ids))

    # Split frames into chunks for each GPU
    chunks = [[] for _ in device_ids]
    for idx, frame_path in enumerate(frame_paths):
        gpu_idx = idx % len(device_ids)
        chunks[gpu_idx].append(frame_path)

    logger.info("Split frames: %s", [len(chunk) for chunk in chunks])

    # Create temporary directories for each GPU using temp manager
    temp_manager = get_temp_manager()
    base_temp_dir = temp_manager.get_temp_dir(media_type) / f"multigpu_enhance_{Path(frame_paths[0]).parent.stem}"
    base_temp_dir.mkdir(exist_ok=True, parents=True)

    temp_dirs = []
    enhanced_dirs = []
    for gpu_idx in range(len(device_ids)):
        temp_dir = base_temp_dir / f"gpu{gpu_idx}_frames"
        enhanced_dir = base_temp_dir / f"gpu{gpu_idx}_enhanced"
        temp_dir.mkdir(exist_ok=True, parents=True)

        if enhanced_dir.exists():
            shutil.rmtree(enhanced_dir)

        temp_dirs.append(temp_dir)
        enhanced_dirs.append(enhanced_dir)

    # Copy frames to GPU directories with sequential numbering
    for gpu_idx, chunk in enumerate(chunks):
        for seq_idx, frame_path in enumerate(chunk):
            dest_path = temp_dirs[gpu_idx] / f"frame_{seq_idx:06d}.png"
            shutil.copy(frame_path, dest_path)

    # Run enhancement on each GPU in parallel
    def enhance_on_gpu(args):
        gpu_idx, gpu_id, temp_dir, enhanced_dir = args
        logger.info("GPU %d: Enhancing %d frames...", gpu_id, len(chunks[gpu_idx]))
        success = apply_realesrgan(temp_dir, enhanced_dir, tile_size, outscale, gpu_id,
                                   model_name, denoise_strength, use_fp32, pre_pad)
        if success:
            logger.info("✅ GPU %d: Enhancement complete", gpu_id)
        return success

    enhancement_args = [
        (gpu_idx, device_ids[gpu_idx], temp_dirs[gpu_idx], enhanced_dirs[gpu_idx])
        for gpu_idx in range(len(device_ids))
    ]

    try:
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            results = list(executor.map(enhance_on_gpu, enhancement_args))

        if not all(results):
            logger.error("Some GPU enhancement tasks failed")
            return None
    except Exception as e:
        logger.error("Multi-GPU enhancement failed: %s", e)
        return None

    # Collect enhanced frames in original order
    logger.info("Collecting enhanced frames from all GPUs...")
    enhanced_frames = [None] * len(frame_paths)

    for idx, frame_path in enumerate(frame_paths):
        gpu_idx = idx % len(device_ids)
        seq_idx = chunks[gpu_idx].index(frame_path)
        enhanced_frame_path = enhanced_dirs[gpu_idx] / f"frame_{seq_idx:06d}_out.png"

        if enhanced_frame_path.exists():
            if media_type == "gif":
                enhanced_frames[idx] = Image.open(enhanced_frame_path).copy()
            else:  # video
                enhanced_frames[idx] = np.array(Image.open(enhanced_frame_path))
        else:
            logger.error("Enhanced frame not found: %s", enhanced_frame_path)
            return None

    # Clean up temporary directories (remove entire base temp directory)
    try:
        if base_temp_dir.exists():
            shutil.rmtree(base_temp_dir)
            logger.info("Cleaned up multi-GPU temp directory: %s", base_temp_dir.name)
    except Exception as e:
        logger.warning("Failed to clean up multi-GPU temp directory: %s", e)

    logger.info("✅ Multi-GPU enhancement complete: %d frames", len(enhanced_frames))

    # Return appropriate format
    if media_type == "video":
        enhanced_clip = ImageSequenceClip(enhanced_frames, fps=fps)
        if audio:
            enhanced_clip = enhanced_clip.set_audio(audio)
        return enhanced_frames, enhanced_clip
    else:  # gif
        return enhanced_frames
