"""
Preview utilities for UI.
Helper functions for displaying file previews.
"""
import logging
import subprocess
from pathlib import Path
from typing import Any, Tuple

import gradio as gr

logger = logging.getLogger("FaceOff")


def show_gif_preview(file):
    """
    Show GIF preview when uploaded.

    Args:
        file: Uploaded file object from Gradio

    Returns:
        Gradio update dict for image component
    """
    if file is None:
        return gr.update(visible=False, value=None)

    from utils.validation import resolve_gradio_file_path

    file_path = resolve_gradio_file_path(file)
    return gr.update(visible=True, value=file_path)


def _extract_first_frame_png(video_path: str, output_path: Path) -> bool:
    """Extract frame 0 as PNG for UI preview."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-ss", "0", "-i", video_path,
            "-frames:v", "1", "-q:v", "2", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(
            "First-frame extract failed for %s: %s",
            video_path,
            (result.stderr or "").strip()[:300],
        )
        return False
    return output_path.is_file()


def validate_target_video_upload(target_file) -> Tuple[str, str, Any]:
    """
    Validate an uploaded target video, clear stale mappings, and show frame preview.

    Uses gr.File for upload so Gradio never needs to decode the video in-browser
    (avoids "Video not playable" on HEVC / phone / download codecs).
    """
    from ui.helpers.face_mapping import clear_face_mappings as helper_clear_mappings
    from ui.handlers.processing_handlers import get_face_mapping_manager
    from utils.temp_manager import get_temp_manager

    _, mappings_text = helper_clear_mappings(get_face_mapping_manager("video"))
    hidden_preview = gr.update(visible=False, value=None)

    if target_file is None:
        return "Upload source image and target video", mappings_text, hidden_preview

    from utils.validation import (
        resolve_gradio_file_path,
        validate_safe_path,
        validate_file_size,
        validate_media_type,
        validate_video_duration,
    )

    try:
        path = resolve_gradio_file_path(target_file)
        validate_safe_path(path)
        validate_file_size(path)
        media_type = validate_media_type(path)
        if media_type != "video":
            return (
                f"⚠️ Expected a video file but detected '{media_type}'. "
                "Use the **GIF** tab for .gif files.",
                mappings_text,
                hidden_preview,
            )
        validate_video_duration(path)

        preview_path = get_temp_manager().get_temp_dir("ui") / "target_first_frame.png"
        preview_update = hidden_preview
        if _extract_first_frame_png(path, preview_path):
            preview_update = gr.update(visible=True, value=str(preview_path))

        name = Path(path).name
        return f"✅ Video ready: {name}", mappings_text, preview_update
    except Exception as exc:
        logger.warning("Target video validation failed: %s", exc)
        return f"❌ {exc}", mappings_text, hidden_preview


def show_video_preview(file):
    """
    Show video preview when uploaded.

    Args:
        file: Uploaded file object from Gradio

    Returns:
        Gradio update dict for video component
    """
    if file is None:
        return gr.update(visible=False, value=None)

    from utils.validation import resolve_gradio_file_path

    file_path = resolve_gradio_file_path(file)
    return gr.update(visible=True, value=file_path)