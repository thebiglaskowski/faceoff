"""
Image and video enhancement using Real-ESRGAN and GFPGAN.

This module provides direct Python API integration with Real-ESRGAN
for image/video upscaling and GFPGAN for face restoration.
"""
import gc
import logging

import cv2
import numpy as np
import torch
from typing import Optional

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
    face_enhance: bool = True,
    clear_cache: bool = True,
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
        if clear_cache and torch.cuda.is_available():
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