"""
Gallery utilities for displaying processed media files.
"""
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("FaceOff")

# Cache for gallery file lists
_gallery_cache: Dict[str, Dict[str, Any]] = {
    'image': {'files': [], 'timestamp': 0, 'dir_mtime': 0},
    'gif': {'files': [], 'timestamp': 0, 'dir_mtime': 0},
    'video': {'files': [], 'timestamp': 0, 'dir_mtime': 0}
}
CACHE_DURATION = 30  # Cache duration in seconds


def _should_refresh_cache(media_type: str, media_path: Path) -> bool:
    """Check if cache should be refreshed based on time and directory modification."""
    cache_entry = _gallery_cache.get(media_type)
    if not cache_entry or not cache_entry['files']:
        return True
    
    # Check if cache has expired
    current_time = time.time()
    if current_time - cache_entry['timestamp'] > CACHE_DURATION:
        return True
    
    # Check if directory has been modified
    if media_path.exists():
        dir_mtime = media_path.stat().st_mtime
        if dir_mtime > cache_entry['dir_mtime']:
            return True
    
    return False


def get_media_files(output_dir: str, media_type: str, max_files: int = 50) -> List[Tuple[str, str]]:
    """
    Get list of media files from the specified output directory.
    Uses caching to improve performance.
    
    Args:
        output_dir: Base output directory (e.g., "outputs")
        media_type: Type of media ("image", "gif", or "video")
        max_files: Maximum number of files to return (default: 50)
        
    Returns:
        List of tuples (file_path, caption) suitable for Gradio Gallery
    """
    base_path = Path(output_dir)
    media_path = base_path / media_type
    
    if not media_path.exists():
        logger.warning(f"Media directory does not exist: {media_path}")
        return []
    
    # Check if we should use cached data
    if not _should_refresh_cache(media_type, media_path):
        cached_files = _gallery_cache[media_type]['files']
        logger.info(f"Using cached data for {media_type} ({len(cached_files)} files, returning {min(len(cached_files), max_files)})")
        # Return limited results from cache
        return cached_files[:max_files]
    
    # Cache miss or expired - scan directory
    logger.info(f"Refreshing cache for {media_type}")
    
    # Define file extensions for each media type
    extensions = {
        "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
        "gif": {".gif"},
        "video": {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    }
    
    valid_extensions = extensions.get(media_type, set())
    
    # Scan directory for valid media files
    media_files = []
    try:
        # Sort by modification time (newest first)
        all_files = sorted(
            [f for f in media_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        total_count = len(all_files)
        
        # Build full file list for cache (not limited)
        for file_path in all_files:
            # Get file modification time for caption
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Create caption with filename and timestamp
            caption = f"{file_path.name}\n{time_str}"
            
            media_files.append((str(file_path), caption))
        
        # Update cache with all files
        _gallery_cache[media_type] = {
            'files': media_files,
            'timestamp': time.time(),
            'dir_mtime': media_path.stat().st_mtime
        }
        
        if total_count > max_files:
            logger.info(f"Showing {max_files} of {total_count} {media_type} files (most recent)")
        else:
            logger.info(f"Found {len(media_files)} {media_type} files in {media_path}")
        
    except Exception as e:
        logger.error(f"Error scanning {media_type} directory: {e}")
        return []
    
    # Return limited results
    return media_files[:max_files]


def get_image_files(output_dir: str = "outputs", max_files: int = 50) -> List[Tuple[str, str]]:
    """Get list of processed images."""
    return get_media_files(output_dir, "image", max_files)


def get_gif_files(output_dir: str = "outputs", max_files: int = 50) -> List[Tuple[str, str]]:
    """Get list of processed GIFs."""
    return get_media_files(output_dir, "gif", max_files)


def get_video_files(output_dir: str = "outputs", max_files: int = 50) -> List[Tuple[str, str]]:
    """Get list of processed videos."""
    return get_media_files(output_dir, "video", max_files)


def clear_gallery_cache(media_type: Optional[str] = None):
    """
    Clear the gallery cache for specified media type or all types.
    
    Args:
        media_type: Type to clear ("image", "gif", "video") or None for all
    """
    if media_type:
        _gallery_cache[media_type] = {'files': [], 'timestamp': 0, 'dir_mtime': 0}
        logger.debug(f"Cleared gallery cache for {media_type}")
    else:
        for key in _gallery_cache:
            _gallery_cache[key] = {'files': [], 'timestamp': 0, 'dir_mtime': 0}
        logger.debug("Cleared all gallery caches")


def count_media_files(output_dir: str = "outputs") -> dict:
    """
    Count media files in each category.
    
    Returns:
        Dict with counts: {"image": 5, "gif": 2, "video": 3}
    """
    return {
        "image": len(get_image_files(output_dir)),
        "gif": len(get_gif_files(output_dir)),
        "video": len(get_video_files(output_dir))
    }


def delete_file(file_path: str, media_type: str) -> Tuple[bool, str]:
    """
    Delete a file from the gallery and invalidate cache.
    
    Args:
        file_path: Absolute path to the file to delete
        media_type: Type of media ("image", "gif", or "video")
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    path = None
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"❌ File not found: {path.name}"
        
        if not path.is_file():
            return False, f"❌ Not a file: {path.name}"
        
        # Verify the file is in the correct media type folder
        if media_type not in str(path.parent):
            return False, f"❌ File type mismatch: {path.name}"
        
        # Delete the file
        path.unlink()
        
        # Invalidate cache for this media type
        _gallery_cache[media_type] = {'files': [], 'timestamp': 0, 'dir_mtime': 0}
        logger.info(f"Deleted {media_type} file: {file_path} (cache invalidated)")
        
        return True, f"✅ Deleted: {path.name}"
        
    except PermissionError:
        logger.error(f"Permission denied deleting file: {file_path}")
        return False, f"❌ Permission denied: {path.name if path else 'unknown'}"
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        return False, f"❌ Error deleting file: {str(e)}"
