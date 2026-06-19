"""
Input validation functions for files, media, and parameters.
"""
import logging
import magic
from pathlib import Path
from PIL import Image
from utils.constants import (
    MAX_FILE_SIZE_MB,
    MAX_VIDEO_DURATION_SEC,
    MAX_IMAGE_PIXELS,
    MAX_GIF_FRAMES
)

logger = logging.getLogger("FaceOff")


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
