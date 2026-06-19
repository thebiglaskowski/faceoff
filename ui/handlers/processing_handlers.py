"""
Processing handlers for FaceOff UI.

This module handles:
- Media processing (image, GIF, video)
- Face mapping operations
"""

import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
from PIL import Image
import gradio as gr

from processing.facade import FaceMappingManager
from utils.validation import (
    validate_file_size,
    validate_image_resolution,
    validate_video_duration,
    validate_gif_frames,
    validate_media_type,
    validate_safe_path,
    resolve_gradio_file_path,
    is_animated_gif_image,
)
from utils.constants import (
    MODEL_OPTIONS,
    SWINIR_MODEL_OPTIONS,
    HAT_MODEL_OPTIONS,
    DEFAULT_MODEL,
    DEFAULT_SWINIR_MODEL,
    DEFAULT_HAT_MODEL,
    DEFAULT_TILE_SIZE,
    DEFAULT_OUTSCALE,
    DEFAULT_USE_FP32,
    DEFAULT_PRE_PAD,
)
from processing.orchestrator import process_media, ProcessOptions
from utils.error_handler import ErrorHandler, FriendlyError
from utils.config_manager import config
from ui.helpers.face_mapping import (
    add_face_mapping as helper_add_mapping,
    clear_face_mappings as helper_clear_mappings,
)
from ui.helpers.gallery_utils import invalidate_gallery_for_new_file

logger = logging.getLogger("FaceOff")

_MEDIA_KINDS = ("image", "gif", "video")
_face_mapping_managers: dict[str, FaceMappingManager] = {
    kind: FaceMappingManager() for kind in _MEDIA_KINDS
}


def get_face_mapping_manager(media_kind: str = "image") -> FaceMappingManager:
    """Get the face mapping manager for a specific media tab."""
    if media_kind not in _face_mapping_managers:
        raise ValueError(f"Unknown media kind: {media_kind}")
    return _face_mapping_managers[media_kind]


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
    restoration_model: str = "GFPGAN",
    *,
    media_kind: str = "image",
    expected_media_type: Optional[str] = None,
):
    """
    Core processing function for all media types.

    This is the internal implementation - use process_image, process_gif,
    or process_video for the public API.
    """
    logger.info("=" * 40)
    logger.info("Starting processing...")

    temp_target_path = None
    media_type = "unknown"
    target_path = None
    try:
        if source_image is None:
            raise gr.Error("Please upload a source image.")

        if target_image_path:
            if is_animated_gif_image(target_image_path):
                raise gr.Error(
                    "Animated GIF detected on the Image tab. "
                    "Use the **GIF** tab to process GIFs frame-by-frame."
                )
            target_path = target_image_path
            if isinstance(target_path, Image.Image):
                timestamp = int(time.time() * 1000)
                temp_path = Path("inputs") / f"temp_target_{timestamp}.png"
                temp_path.parent.mkdir(exist_ok=True)
                target_path.save(temp_path)
                temp_target_path = temp_path
                target_path = str(temp_path)
            media_type = "image"
        elif target_video_path:
            target_path = resolve_gradio_file_path(target_video_path)
            media_type = validate_media_type(target_path)
        else:
            raise gr.Error("Please upload a target file.")

        if expected_media_type and media_type != expected_media_type:
            if expected_media_type == "gif":
                raise gr.Error(
                    "Target is not a GIF file. Upload a .gif on the **GIF** tab "
                    "(not a single-frame preview from the Image tab)."
                )
            if expected_media_type == "video":
                raise gr.Error("Target is not a video file. Upload a video on the **Video** tab.")
            raise gr.Error(
                f"Expected {expected_media_type} input but received {media_type}. "
                "Check that you are on the correct tab."
            )

        if isinstance(target_path, str) and not isinstance(target_image_path, Image.Image):
            validate_safe_path(target_path)
        validate_file_size(target_path)

        if media_type == "image":
            validate_image_resolution(target_path)
        elif media_type == "video":
            validate_video_duration(target_path)
        elif media_type == "gif":
            validate_gif_frames(target_path)

        if tile_size is None:
            tile_size = DEFAULT_TILE_SIZE
        if outscale is None:
            outscale = DEFAULT_OUTSCALE
        if use_fp32 is None:
            use_fp32 = DEFAULT_USE_FP32
        if pre_pad is None:
            pre_pad = DEFAULT_PRE_PAD

        if enhancement_model == "SwinIR":
            if model_selection and model_selection in SWINIR_MODEL_OPTIONS:
                model_name = SWINIR_MODEL_OPTIONS[model_selection]["model_name"]
            else:
                model_name = SWINIR_MODEL_OPTIONS[DEFAULT_SWINIR_MODEL]["model_name"]
        elif enhancement_model == "HAT":
            if model_selection and model_selection in HAT_MODEL_OPTIONS:
                model_name = HAT_MODEL_OPTIONS[model_selection]["model_name"]
            else:
                model_name = HAT_MODEL_OPTIONS[DEFAULT_HAT_MODEL]["model_name"]
        else:
            if model_selection and model_selection in MODEL_OPTIONS:
                model_name = MODEL_OPTIONS[model_selection]["model_name"]
            else:
                model_name = MODEL_OPTIONS[DEFAULT_MODEL]["model_name"]

        source_array = np.array(source_image)
        mappings = get_face_mapping_manager(media_kind).get()
        logger.info("Face mapping manager returned (%s): %s", media_kind, mappings)

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        logger.info(
            "Processing %s with enhancement=%s (%s), model=%s, denoise=%.2f, "
            "confidence=%.2f, tile=%d, outscale=%d, fp32=%s, prepad=%d, "
            "restore=%s (%s), weight=%.2f, tensorrt_fp16=%s",
            media_type,
            enhance,
            enhancement_model,
            model_name,
            denoise_strength,
            confidence,
            tile_size,
            outscale,
            use_fp32,
            pre_pad,
            restore_faces,
            restoration_model,
            restoration_weight,
            tensorrt_fp16,
        )

        opts = ProcessOptions(
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
            restoration_model=restoration_model,
            tensorrt_fp16=tensorrt_fp16,
            model_display=model_selection,
        )

        result_img, result_vid = process_media(opts)

        logger.info("Processing complete!")
        logger.info("=" * 40)

        invalidate_gallery_for_new_file(media_type)

        if media_type == "image":
            return result_img
        return result_vid

    except FriendlyError as fe:
        logger.error("=" * 40)
        raise gr.Error(fe.format_message())
    except gr.Error:
        raise
    except Exception as e:
        context = {
            "media_type": media_type,
            "file_path": target_path if target_path is not None else "unknown",
            "enhance": enhance,
            "tile_size": tile_size if tile_size else DEFAULT_TILE_SIZE,
            "outscale": outscale if outscale else DEFAULT_OUTSCALE,
            "restore_faces": restore_faces,
            "use_fp32": use_fp32 if use_fp32 else DEFAULT_USE_FP32,
        }
        friendly_error = ErrorHandler.handle_error(e, context)
        logger.error("Processing error: %s", e, exc_info=True)
        logger.error("=" * 40)
        raise gr.Error(friendly_error.format_message())
    finally:
        if temp_target_path is not None:
            with suppress(OSError):
                temp_target_path.unlink(missing_ok=True)


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
    restoration_model: str = "GFPGAN",
):
    """Process an image face swap."""
    return _process_input(
        source_img,
        target_img,
        None,
        enhance,
        confidence,
        gpu,
        None,
        model,
        denoise,
        tile,
        outscale,
        fp32,
        prepad,
        restore,
        weight,
        tensorrt_fp16,
        enhancement_model,
        restoration_model,
        media_kind="image",
        expected_media_type="image",
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
    restoration_model: str = "GFPGAN",
):
    """Process a GIF face swap."""
    return _process_input(
        source_img,
        None,
        target_file,
        enhance,
        confidence,
        gpu,
        None,
        model,
        denoise,
        tile,
        outscale,
        fp32,
        prepad,
        restore,
        weight,
        tensorrt_fp16,
        enhancement_model,
        restoration_model,
        media_kind="gif",
        expected_media_type="gif",
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
    restoration_model: str = "GFPGAN",
):
    """Process a video face swap."""
    return _process_input(
        source_img,
        None,
        target_file,
        enhance,
        confidence,
        gpu,
        None,
        model,
        denoise,
        tile,
        outscale,
        fp32,
        prepad,
        restore,
        weight,
        tensorrt_fp16,
        enhancement_model,
        restoration_model,
        media_kind="video",
        expected_media_type="video",
    )


def _add_face_mapping_for_tab(
    media_kind: str,
    source_idx: int,
    target_idx: int,
    current_mappings_text: str,
) -> Tuple[Any, Any]:
    return helper_add_mapping(
        source_idx,
        target_idx,
        current_mappings_text,
        get_face_mapping_manager(media_kind),
    )


def _clear_face_mappings_for_tab(media_kind: str) -> Tuple[Any, Any]:
    return helper_clear_mappings(get_face_mapping_manager(media_kind))


def add_face_mapping_image(
    source_idx: int,
    target_idx: int,
    current_mappings_text: str,
) -> Tuple[Any, Any]:
    return _add_face_mapping_for_tab("image", source_idx, target_idx, current_mappings_text)


def add_face_mapping_gif(
    source_idx: int,
    target_idx: int,
    current_mappings_text: str,
) -> Tuple[Any, Any]:
    return _add_face_mapping_for_tab("gif", source_idx, target_idx, current_mappings_text)


def add_face_mapping_video(
    source_idx: int,
    target_idx: int,
    current_mappings_text: str,
) -> Tuple[Any, Any]:
    return _add_face_mapping_for_tab("video", source_idx, target_idx, current_mappings_text)


def clear_face_mappings_image() -> Tuple[Any, Any]:
    return _clear_face_mappings_for_tab("image")


def clear_face_mappings_gif() -> Tuple[Any, Any]:
    return _clear_face_mappings_for_tab("gif")


def clear_face_mappings_video() -> Tuple[Any, Any]:
    return _clear_face_mappings_for_tab("video")


# Backward-compatible aliases (default to image tab)
def add_face_mapping_wrapper(
    source_idx: int,
    target_idx: int,
    current_mappings_text: str,
) -> Tuple[Any, Any]:
    return add_face_mapping_image(source_idx, target_idx, current_mappings_text)


def clear_face_mappings_wrapper() -> Tuple[Any, Any]:
    return clear_face_mappings_image()