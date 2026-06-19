"""
Input validation functions for files, media, and parameters.
"""
import logging
import tempfile
import magic
from pathlib import Path
from typing import Any, List, Optional, Tuple
from PIL import Image
from utils.constants import (
    MAX_FILE_SIZE_MB,
    MAX_VIDEO_DURATION_SEC,
    MAX_IMAGE_PIXELS,
    MAX_GIF_FRAMES
)

logger = logging.getLogger("FaceOff")


def _allowed_path_roots() -> tuple[Path, ...]:
    """Directories from which user-supplied media paths are accepted."""
    roots = [
        Path("inputs").resolve(),
        Path("outputs").resolve(),
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]
    # Gradio uploads land under $TMPDIR/gradio/ on Linux/WSL
    gradio_tmp = Path(tempfile.gettempdir()) / "gradio"
    if gradio_tmp.exists():
        roots.append(gradio_tmp.resolve())
    return tuple(roots)


def validate_safe_path(file_path: str) -> Path:
    """
    Resolve a user-supplied path and reject directory traversal escapes.

    Allows project dirs (inputs/, outputs/) and OS/Gradio temp upload paths.

    Raises:
        ValueError: If the resolved path is outside allowed directories.
    """
    raw = Path(file_path)
    if any(part == ".." for part in raw.parts):
        raise ValueError(f"Path traversal not allowed: {file_path}")

    resolved = raw.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Path does not exist: {file_path}")

    for root in _allowed_path_roots():
        if resolved == root or root in resolved.parents:
            return resolved

    raise ValueError(f"Path not allowed: {file_path}")


def validate_file_size(file_path: str, max_size_mb: int = MAX_FILE_SIZE_MB) -> None:
    """
    Validate file size is within limits.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum allowed size in MB
        
    Raises:
        ValueError: If file exceeds size limit
    """
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB. Maximum allowed: {max_size_mb}MB")
    logger.info("File size: %.1f MB", file_size_mb)


def validate_image_resolution(image_path: str) -> None:
    """
    Validate image resolution is within limits.
    
    Args:
        image_path: Path to image file
        
    Raises:
        ValueError: If image resolution exceeds limit
    """
    img = Image.open(image_path)
    pixels = img.width * img.height
    if pixels > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image resolution too high: {img.width}x{img.height}. "
            f"Maximum: 4096x4096 pixels"
        )
    logger.info("Image resolution: %dx%d", img.width, img.height)


def validate_video_duration(video_path: str) -> None:
    """
    Validate video duration is within limits.
    
    Args:
        video_path: Path to video file
        
    Raises:
        ValueError: If video duration exceeds limit
    """
    try:
        from utils import video_io
        meta = video_io.probe_video(video_path)
        duration = meta.get('duration', 0) or 0
        
        if duration > MAX_VIDEO_DURATION_SEC:
            minutes = duration / 60
            max_minutes = MAX_VIDEO_DURATION_SEC / 60
            raise ValueError(
                f"Video too long: {minutes:.1f} minutes. "
                f"Maximum: {max_minutes:.0f} minutes"
            )
        
        logger.info("Video duration: %.1f seconds", duration)
    except ValueError:
        raise
    except Exception as e:
        logger.error("Failed to validate video: %s", e)
        raise ValueError("Invalid video file or unable to read video metadata")


def validate_gif_frames(gif_path: str) -> None:
    """
    Validate GIF frame count is within limits.
    
    Args:
        gif_path: Path to GIF file
        
    Raises:
        ValueError: If GIF has too many frames
    """
    try:
        gif = Image.open(gif_path)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                gif.seek(frame_count)
        except EOFError:
            pass
        
        if frame_count > MAX_GIF_FRAMES:
            raise ValueError(
                f"GIF has too many frames: {frame_count}. "
                f"Maximum: {MAX_GIF_FRAMES} frames"
            )
        
        logger.info("GIF frame count: %d", frame_count)
    except ValueError:
        raise
    except Exception as e:
        logger.error("Failed to validate GIF: %s", e)
        raise ValueError("Invalid GIF file or unable to read GIF metadata")


def validate_media_type(file_path: str) -> str:
    """
    Validate and detect media type.

    Args:
        file_path: Path to media file

    Returns:
        Media type: 'image', 'video', or 'gif'

    Raises:
        ValueError: If media type is unsupported
    """
    mime_type = magic.from_file(file_path, mime=True)

    if mime_type == "image/gif":
        return "gif"
    elif mime_type.startswith("image"):
        return "image"
    elif mime_type.startswith("video"):
        return "video"

    raise ValueError(
        f"Unsupported media type: {mime_type}. "
        "Only images, GIFs, and videos are allowed."
    )


def resolve_gradio_file_path(file_obj: Any) -> str:
    """Resolve a Gradio File/upload value to a filesystem path."""
    if file_obj is None:
        raise ValueError("No file provided")

    # Gradio Video may pass (video_path, subtitles_path) tuples.
    if isinstance(file_obj, (tuple, list)) and file_obj:
        return resolve_gradio_file_path(file_obj[0])

    if isinstance(file_obj, dict):
        for key in ("path", "name"):
            value = file_obj.get(key)
            if value:
                return str(value)
        raise ValueError("Invalid file object: missing path")

    if hasattr(file_obj, "path") and file_obj.path:
        return str(file_obj.path)
    if hasattr(file_obj, "name") and file_obj.name:
        return str(file_obj.name)
    return str(file_obj)


def is_animated_gif_image(image: Image.Image) -> bool:
    """Return True when a PIL image represents a multi-frame GIF."""
    if image is None:
        return False
    if getattr(image, "format", None) == "GIF":
        return getattr(image, "n_frames", 1) > 1
    return getattr(image, "n_frames", 1) > 1


def validate_face_mappings_or_raise(
    face_mappings: Optional[List[Tuple[int, int]]],
    src_face_count: int,
    dst_face_count: int,
) -> None:
    """
    Ensure at least one face mapping is valid for detected face counts.

    Raises:
        ValueError: When mappings were provided but none apply to detected faces.
    """
    if not face_mappings:
        return

    valid = [
        (src_idx, dst_idx)
        for src_idx, dst_idx in face_mappings
        if src_idx < src_face_count and dst_idx < dst_face_count
    ]
    if valid:
        return

    raise ValueError(
        f"No valid face mappings for detected faces "
        f"(source: {src_face_count}, target: {dst_face_count}). "
        f"Configured mappings: {face_mappings}. "
        "Use Detect Faces on this target, then update mappings."
    )


# =============================================================================
# Convenience wrapper functions
# =============================================================================

def validate_image_file(
    file_path: str,
    max_size_mb: int = MAX_FILE_SIZE_MB,
    max_pixels: int = MAX_IMAGE_PIXELS
) -> bool:
    """
    Validate an image file (size and resolution).

    Args:
        file_path: Path to image file
        max_size_mb: Maximum file size in MB
        max_pixels: Maximum image resolution in pixels

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    validate_file_size(file_path, max_size_mb)

    img = Image.open(file_path)
    pixels = img.width * img.height
    if pixels > max_pixels:
        raise ValueError(
            f"Image resolution too high: {img.width}x{img.height} ({pixels} pixels). "
            f"Maximum: {max_pixels} pixels"
        )

    logger.info("Image validated: %dx%d, %.1f MB",
                img.width, img.height,
                Path(file_path).stat().st_size / (1024 * 1024))
    return True


def validate_video_file(
    file_path: str,
    max_size_mb: int = MAX_FILE_SIZE_MB,
    max_duration_sec: int = MAX_VIDEO_DURATION_SEC
) -> bool:
    """
    Validate a video file (size and duration).

    Args:
        file_path: Path to video file
        max_size_mb: Maximum file size in MB
        max_duration_sec: Maximum duration in seconds

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    validate_file_size(file_path, max_size_mb)
    validate_video_duration(file_path)

    logger.info("Video file validated: %s", file_path)
    return True


def validate_gif_file(
    file_path: str,
    max_size_mb: int = MAX_FILE_SIZE_MB,
    max_frames: int = MAX_GIF_FRAMES
) -> bool:
    """
    Validate a GIF file (size and frame count).

    Args:
        file_path: Path to GIF file
        max_size_mb: Maximum file size in MB
        max_frames: Maximum number of frames

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    validate_file_size(file_path, max_size_mb)

    try:
        gif = Image.open(file_path)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                gif.seek(frame_count)
        except EOFError:
            pass

        if frame_count > max_frames:
            raise ValueError(
                f"GIF has too many frames: {frame_count}. "
                f"Maximum: {max_frames} frames"
            )

        logger.info("GIF validated: %d frames", frame_count)
        return True

    except ValueError:
        raise
    except Exception as e:
        logger.error("Failed to validate GIF: %s", e)
        raise ValueError("Invalid GIF file or unable to read GIF metadata")
