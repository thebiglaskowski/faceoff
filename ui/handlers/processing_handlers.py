"""
Processing handlers for FaceOff UI.

This module handles:
- Media processing (image, GIF, video)
- Face mapping operations
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
from PIL import Image
import gradio as gr

from processing.facade import FaceMappingManager
from utils.validation import (
    validate_file_size, validate_image_resolution,
    validate_video_duration, validate_gif_frames, validate_media_type
)
from utils.constants import (
    MODEL_OPTIONS, SWINIR_MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_SWINIR_MODEL,
    DEFAULT_TILE_SIZE, DEFAULT_OUTSCALE, DEFAULT_USE_FP32, DEFAULT_PRE_PAD
)
from processing.orchestrator import process_media
from utils.error_handler import ErrorHandler, FriendlyError
from utils.config_manager import config
from ui.helpers.face_mapping import (
    add_face_mapping as helper_add_mapping,
    clear_face_mappings as helper_clear_mappings
)
from ui.helpers.gallery_utils import invalidate_gallery_for_new_file

logger = logging.getLogger("FaceOff")

# Global state for face mappings
_face_mapping_manager: Optional[FaceMappingManager] = None


def get_face_mapping_manager() -> FaceMappingManager:
    """Get or create the face mapping manager singleton."""
    global _face_mapping_manager
    if _face_mapping_manager is None:
        _face_mapping_manager = FaceMappingManager()
    return _face_mapping_manager


def _process_input(
    source_image,
    target_image_path=None,
    target_video_path=None,
    enhance: bool = False,
    confidence: float = 0.5,
    gpu_selection=None,
    face_mappings=None,
    model_selection: str = None,
    denoise_strength: float = 0.5,
    tile_size: int = None,
    outscale: int = None,
    use_fp32: bool = None,
    pre_pad: int = None,
    restore_faces: bool = False,
    restoration_weight: float = 0.5,
    tensorrt_fp16: bool = True,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN"
):
    """
    Core processing function for all media types.

    This is the internal implementation - use process_image, process_gif,
    or process_video for the public API.
    """
    logger.info("=" * 40)
    logger.info("Starting processing...")

    try:
        # Validate source image
        if source_image is None:
            raise gr.Error("Please upload a source image.")

        # Determine media type and target path
        if target_image_path:
            target_path = target_image_path
            if isinstance(target_path, Image.Image):
                timestamp = int(time.time() * 1000)
                temp_path = Path("inputs") / f"temp_target_{timestamp}.png"
                temp_path.parent.mkdir(exist_ok=True)
                target_path.save(temp_path)
                target_path = str(temp_path)
            media_type = "image"
        elif target_video_path:
            target_path = target_video_path.name if hasattr(target_video_path, 'name') else target_video_path
            media_type = validate_media_type(target_path)
        else:
            raise gr.Error("Please upload a target file.")

        # Validate files based on type
        validate_file_size(target_path)

        if media_type == "image":
            validate_image_resolution(target_path)
        elif media_type == "video":
            validate_video_duration(target_path)
        elif media_type == "gif":
            validate_gif_frames(target_path)

        # Apply defaults for missing settings
        if tile_size is None:
            tile_size = DEFAULT_TILE_SIZE
        if outscale is None:
            outscale = DEFAULT_OUTSCALE
        if use_fp32 is None:
            use_fp32 = DEFAULT_USE_FP32
        if pre_pad is None:
            pre_pad = DEFAULT_PRE_PAD

        # Update TensorRT FP16 config for this processing run
        config.update('gpu', 'tensorrt_fp16', value=tensorrt_fp16)

        # Parse model selection based on enhancement framework
        if enhancement_model == "SwinIR":
            if model_selection and model_selection in SWINIR_MODEL_OPTIONS:
                model_name = SWINIR_MODEL_OPTIONS[model_selection]["model_name"]
            else:
                model_name = SWINIR_MODEL_OPTIONS[DEFAULT_SWINIR_MODEL]["model_name"]
        else:
            if model_selection and model_selection in MODEL_OPTIONS:
                model_name = MODEL_OPTIONS[model_selection]["model_name"]
            else:
                model_name = MODEL_OPTIONS[DEFAULT_MODEL]["model_name"]

        # Convert source to numpy array
        source_array = np.array(source_image)

        # Get face mappings from manager
        mappings = get_face_mapping_manager().get()
        logger.info("Face mapping manager returned: %s", mappings)

        # Process media
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        logger.info(
            "Processing %s with enhancement=%s (%s), model=%s, denoise=%.2f, "
            "confidence=%.2f, tile=%d, outscale=%d, fp32=%s, prepad=%d, "
            "restore=%s (%s), weight=%.2f, tensorrt_fp16=%s",
            media_type, enhance, enhancement_model, model_name, denoise_strength, confidence,
            tile_size, outscale, use_fp32, pre_pad, restore_faces, restoration_model,
            restoration_weight, tensorrt_fp16
        )

        result_img, result_vid = process_media(
            source_image=source_array,
            dest_path=target_path,
            media_type=media_type,
            output_dir=str(output_dir),
            enhance=enhance,
            tile_size=tile_size,
            outscale=outscale,
            face_confidence=confidence,
            gpu_selection=gpu_selection,
            face_mappings=mappings,
            model_name=model_name,
            denoise_strength=denoise_strength,
            use_fp32=use_fp32,
            pre_pad=pre_pad,
            restore_faces=restore_faces,
            restoration_weight=restoration_weight,
            enhancement_model=enhancement_model,
            restoration_model=restoration_model
        )

        logger.info("Processing complete!")
        logger.info("=" * 40)

        # Invalidate gallery cache so new file appears immediately
        invalidate_gallery_for_new_file(media_type)

        # Return appropriate result based on media type
        if media_type == "image":
            return result_img
        else:
            return result_vid

    except FriendlyError as fe:
        logger.error("=" * 40)
        raise gr.Error(fe.format_message())
    except gr.Error:
        raise
    except Exception as e:
        # Convert technical errors to friendly messages
        context = {
            'media_type': media_type if 'media_type' in locals() else 'unknown',
            'enhance': enhance,
            'tile_size': tile_size if tile_size else DEFAULT_TILE_SIZE,
            'outscale': outscale if outscale else DEFAULT_OUTSCALE,
            'restore_faces': restore_faces,
            'use_fp32': use_fp32 if use_fp32 else DEFAULT_USE_FP32
        }
        friendly_error = ErrorHandler.handle_error(e, context)
        logger.error("Processing error: %s", e, exc_info=True)
        logger.error("=" * 40)
        raise gr.Error(friendly_error.format_message())


def process_image(
    source_img,
    target_img,
    enhance: bool,
    confidence: float,
    gpu,
    model: str,
    denoise: float,
    tile: int,
    outscale: int,
    fp32: bool,
    prepad: int,
    restore: bool,
    weight: float,
    tensorrt_fp16: bool = True,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN"
):
    """Process an image face swap."""
    return _process_input(
        source_img, target_img, None,
        enhance, confidence, gpu, None,
        model, denoise, tile, outscale, fp32, prepad,
        restore, weight, tensorrt_fp16,
        enhancement_model, restoration_model
    )


def process_gif(
    source_img,
    target_file,
    enhance: bool,
    confidence: float,
    gpu,
    model: str,
    denoise: float,
    tile: int,
    outscale: int,
    fp32: bool,
    prepad: int,
    restore: bool,
    weight: float,
    tensorrt_fp16: bool = True,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN"
):
    """Process a GIF face swap."""
    return _process_input(
        source_img, None, target_file,
        enhance, confidence, gpu, None,
        model, denoise, tile, outscale, fp32, prepad,
        restore, weight, tensorrt_fp16,
        enhancement_model, restoration_model
    )


def process_video(
    source_img,
    target_file,
    enhance: bool,
    confidence: float,
    gpu,
    model: str,
    denoise: float,
    tile: int,
    outscale: int,
    fp32: bool,
    prepad: int,
    restore: bool,
    weight: float,
    tensorrt_fp16: bool = True,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN"
):
    """Process a video face swap."""
    return _process_input(
        source_img, None, target_file,
        enhance, confidence, gpu, None,
        model, denoise, tile, outscale, fp32, prepad,
        restore, weight, tensorrt_fp16,
        enhancement_model, restoration_model
    )


def add_face_mapping_wrapper(
    source_idx: int,
    target_idx: int,
    current_mappings_text: str
) -> Tuple[Any, Any]:
    """Add a face mapping using the global manager."""
    return helper_add_mapping(
        source_idx, target_idx, current_mappings_text,
        get_face_mapping_manager()
    )


def clear_face_mappings_wrapper() -> Tuple[Any, Any]:
    """Clear all face mappings using the global manager."""
    return helper_clear_mappings(get_face_mapping_manager())
