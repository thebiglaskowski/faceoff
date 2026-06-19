"""
Face detection helpers for UI.
Consolidates duplicate face detection logic across UI components.
"""
import logging
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

from core.face_processor import sort_faces_by_position, filter_faces_by_confidence
from core.media_processor import MediaProcessor
from utils.config_manager import config

logger = logging.getLogger("FaceOff")

_ui_processor: Optional[MediaProcessor] = None


def _get_ui_processor() -> MediaProcessor:
    """Reuse preloaded model-pool processor (TensorRT when enabled)."""
    global _ui_processor
    if _ui_processor is None:
        _ui_processor = MediaProcessor(
            device_id=0,
            use_tensorrt=config.tensorrt_enabled,
            optimize_models=False,
        )
        logger.debug("UI face-detection processor initialized (TensorRT=%s)", config.tensorrt_enabled)
    else:
        _ui_processor._ensure_bound()
    return _ui_processor


def _pil_to_rgb_array(pil_image: Image.Image) -> np.ndarray:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return np.array(pil_image)


def add_index_overlay(face_pil: Image.Image, index: int) -> Image.Image:
    """
    Add index number overlay to face thumbnail.
    
    Args:
        face_pil: Face thumbnail as PIL Image
        index: Face index number
        
    Returns:
        Face image with index overlay
    """
    draw = ImageDraw.Draw(face_pil)
    font_size = max(24, int(face_pil.width * 0.15))
    
    # Try to load a nice font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text = str(index)
    
    # Get text size (handle both old and new PIL versions)
    try:
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
    except:
        text_width, text_height = draw.textsize(text, font=font)
    
    # Draw background rectangle
    padding_text = 8
    x, y = padding_text, padding_text
    bg_rect = [x - 4, y - 4, x + text_width + 4, y + text_height + 4]
    draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return face_pil


def extract_face_thumbnails(image_rgb: 'np.ndarray', faces_raw: List, with_index: bool = True) -> List[Image.Image]:
    """
    Extract face thumbnails from image with optional index overlays.
    
    Args:
        image_rgb: Image as RGB numpy array
        faces_raw: List of detected face objects
        with_index: Whether to add index number overlays
        
    Returns:
        List of face thumbnails as PIL Images
    """
    face_thumbnails = []
    
    for idx, face in enumerate(faces_raw):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Add padding
        padding = 20
        h, w = image_rgb.shape[:2]
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
        
        # Crop face
        face_crop = image_rgb[y1:y2, x1:x2]
        face_pil = Image.fromarray(face_crop)
        
        # Add index overlay if requested
        if with_index:
            face_pil = add_index_overlay(face_pil, idx)
        
        face_thumbnails.append(face_pil)
    
    return face_thumbnails


def detect_faces_simple(pil_image: Image.Image, confidence: float = 0.5) -> str:
    """
    Simple face detection for UI preview (just returns status text).

    Args:
        pil_image: PIL Image to detect faces in
        confidence: Detection confidence threshold

    Returns:
        Status text describing detection results
    """
    if pil_image is None:
        return ""

    try:
        processor = _get_ui_processor()
        faces = processor.get_faces(_pil_to_rgb_array(pil_image))
        faces = filter_faces_by_confidence(faces, confidence)
        if not faces:
            return "No faces detected"
        scores = ", ".join(f"{getattr(f, 'det_score', 0):.2f}" for f in faces[:5])
        return f"Detected {len(faces)} face(s) (scores: {scores})"
    except Exception as e:
        logger.error("Face detection error: %s", e)
        return f"❌ Error: {str(e)}"


def detect_faces_for_mapping(
    source_img: Image.Image,
    target_img: Image.Image,
    face_confidence: float
) -> Tuple[List[Image.Image], List[Image.Image], str, dict, dict, dict, dict]:
    """
    Detect faces in both images for mapping UI.
    
    Args:
        source_img: Source PIL Image
        target_img: Target PIL Image
        face_confidence: Detection confidence threshold
        
    Returns:
        Tuple of (src_faces, tgt_faces, status, src_dropdown, tgt_dropdown, 
                 src_visibility, tgt_visibility)
    """
    if source_img is None or target_img is None:
        return (
            [], [],
            "Upload both images first",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    processor = _get_ui_processor()

    try:
        src_rgb = _pil_to_rgb_array(source_img)
        tgt_rgb = _pil_to_rgb_array(target_img)

        src_faces_raw = processor.get_faces(src_rgb)
        src_faces_raw = filter_faces_by_confidence(src_faces_raw, face_confidence)
        src_faces_raw = sort_faces_by_position(src_faces_raw)
        src_faces = extract_face_thumbnails(src_rgb, src_faces_raw, with_index=True)

        tgt_faces_raw = processor.get_faces(tgt_rgb)
        tgt_faces_raw = filter_faces_by_confidence(tgt_faces_raw, face_confidence)
        tgt_faces_raw = sort_faces_by_position(tgt_faces_raw)
        tgt_faces = extract_face_thumbnails(tgt_rgb, tgt_faces_raw, with_index=True)
        
        # Check if faces were detected
        if not src_faces or not tgt_faces:
            status = f"⚠️ Detection failed - Source: {len(src_faces)} faces, Target: {len(tgt_faces)} faces"
            return (
                [], [],
                status,
                gr.update(choices=[]),
                gr.update(choices=[]),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Create dropdown choices
        src_choices = [f"Source Face {i}" for i in range(len(src_faces))]
        tgt_choices = [f"Target Face {i}" for i in range(len(tgt_faces))]
        status = f"✅ Detected {len(src_faces)} source face(s) and {len(tgt_faces)} target face(s)"
        
        return (
            src_faces,
            tgt_faces,
            status,
            gr.update(choices=src_choices, value=src_choices[0] if src_choices else None),
            gr.update(choices=tgt_choices, value=tgt_choices[0] if tgt_choices else None),
            gr.update(visible=True),
            gr.update(visible=True)
        )
        
    except Exception as e:
        logger.error(f"Face detection for mapping failed: {e}")
        return (
            [], [],
            f"❌ Error: {str(e)}",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )


def detect_faces_with_thumbnails(source_img, target_file, face_confidence):
    """
    Detect faces in source image and target GIF/video (first frame).
    Returns 7 values for Gradio event handler compatibility.
    
    Args:
        source_img: Source PIL Image
        target_file: Target file (video/GIF)
        face_confidence: Detection confidence threshold
        
    Returns:
        Tuple of 7 values: (src_gallery, tgt_gallery, status, src_dropdown, tgt_dropdown, src_visible, tgt_visible)
    """
    if source_img is None or target_file is None:
        return (
            [], [],
            "Upload both source and target file first",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    from utils import video_io
    from utils.validation import resolve_gradio_file_path

    processor = _get_ui_processor()

    try:
        src_rgb = _pil_to_rgb_array(source_img)
        src_faces_raw = processor.get_faces(src_rgb)
        src_faces_raw = filter_faces_by_confidence(src_faces_raw, face_confidence)
        src_faces_raw = sort_faces_by_position(src_faces_raw)
        src_faces = extract_face_thumbnails(src_rgb, src_faces_raw, with_index=True)

        target_path = resolve_gradio_file_path(target_file)

        if target_path.lower().endswith(".gif"):
            with video_io.StreamingFrameReader(
                target_path,
                fps=float(config.streaming_gif_decode_fps),
                hwaccel=False,
            ) as reader:
                frames = reader.read_chunk(1)
            first_frame_rgb = frames[0] if frames else np.array(Image.open(target_path).convert("RGB"))
        else:
            with video_io.StreamingFrameReader(target_path, hwaccel=False) as reader:
                frames = reader.read_chunk(1)
            first_frame_rgb = frames[0] if frames else np.zeros((64, 64, 3), dtype=np.uint8)

        tgt_faces_raw = processor.get_faces(first_frame_rgb)
        tgt_faces_raw = filter_faces_by_confidence(tgt_faces_raw, face_confidence)
        tgt_faces_raw = sort_faces_by_position(tgt_faces_raw)
        
        tgt_faces = extract_face_thumbnails(first_frame_rgb, tgt_faces_raw, with_index=True)
        
        if not src_faces or not tgt_faces:
            status = f"⚠️ Detection failed - Source: {len(src_faces)} faces, Target: {len(tgt_faces)} faces"
            return (
                [], [],
                status,
                gr.update(choices=[]),
                gr.update(choices=[]),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Create dropdown choices
        src_choices = [f"Source Face {i}" for i in range(len(src_faces))]
        tgt_choices = [f"Target Face {i}" for i in range(len(tgt_faces))]
        status = f"✅ Detected {len(src_faces)} source face(s) and {len(tgt_faces)} target face(s)"
        
        return (
            src_faces,
            tgt_faces,
            status,
            gr.update(choices=src_choices, value=src_choices[0] if src_choices else None),
            gr.update(choices=tgt_choices, value=tgt_choices[0] if tgt_choices else None),
            gr.update(visible=True),
            gr.update(visible=True)
        )
        
    except Exception as e:
        logger.error(f"Face detection with thumbnails failed: {e}")
        return (
            [], [],
            f"❌ Error: {str(e)}",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )
