"""
Centralized temporary file management for FaceOff.
Ensures all temp files are organized in the temp/ directory with automatic cleanup.
"""
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger("FaceOff")


class TempManager:
    """Manages temporary file creation and cleanup."""
    
    def __init__(self, base_temp_dir: str = "temp"):
        """
        Initialize temp manager.
        
        Args:
            base_temp_dir: Base directory for all temp files (default: "temp")
        """
        self.base_dir = Path(base_temp_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different purposes
        self.ui_dir = self.base_dir / "ui"
        self.video_dir = self.base_dir / "video"
        self.gif_dir = self.base_dir / "gif"
        self.image_dir = self.base_dir / "image"
        
        # Ensure subdirectories exist
        for subdir in [self.ui_dir, self.video_dir, self.gif_dir, self.image_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
    
    def get_temp_dir(self, category: str = "general") -> Path:
        """
        Get a temp directory for a specific category.
        
        Args:
            category: Category of temp files ("ui", "video", "gif", "image", "general")
            
        Returns:
            Path to temp directory
        """
        category_map = {
            "ui": self.ui_dir,
            "video": self.video_dir,
            "gif": self.gif_dir,
            "image": self.image_dir,
            "general": self.base_dir
        }
        
        temp_dir = category_map.get(category, self.base_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def get_temp_file(self, category: str = "general", suffix: str = "", prefix: str = "tmp_") -> Path:
        """
        Create a temporary file path (doesn't create the file).
        
        Args:
            category: Category of temp file
            suffix: File suffix/extension (e.g., ".png")
            prefix: Filename prefix
            
        Returns:
            Path to temp file
        """
        import time
        temp_dir = self.get_temp_dir(category)
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}{timestamp}{suffix}"
        return temp_dir / filename
    
    @contextmanager
    def temp_file(self, category: str = "general", suffix: str = "", prefix: str = "tmp_", auto_cleanup: bool = True):
        """
        Context manager for temporary files with automatic cleanup.
        
        Args:
            category: Category of temp file
            suffix: File suffix/extension
            prefix: Filename prefix
            auto_cleanup: Whether to delete file on context exit
            
        Yields:
            Path to temp file
            
        Example:
            with temp_manager.temp_file("ui", suffix=".png") as temp_path:
                image.save(temp_path)
                process(temp_path)
            # File automatically deleted
        """
        temp_path = self.get_temp_file(category, suffix, prefix)
        try:
            yield temp_path
        finally:
            if auto_cleanup and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
    
    @contextmanager
    def temp_directory(self, category: str = "general", prefix: str = "tmp_", auto_cleanup: bool = True):
        """
        Context manager for temporary directories with automatic cleanup.
        
        Args:
            category: Category of temp directory
            prefix: Directory name prefix
            auto_cleanup: Whether to delete directory on context exit
            
        Yields:
            Path to temp directory
            
        Example:
            with temp_manager.temp_directory("video") as temp_dir:
                # Save frames
                for i, frame in enumerate(frames):
                    cv2.imwrite(str(temp_dir / f"frame_{i:06d}.png"), frame)
            # Directory and all contents automatically deleted
        """
        import time
        temp_dir = self.get_temp_dir(category)
        timestamp = int(time.time() * 1000)
        temp_subdir = temp_dir / f"{prefix}{timestamp}"
        temp_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            yield temp_subdir
        finally:
            if auto_cleanup and temp_subdir.exists():
                try:
                    shutil.rmtree(temp_subdir)
                    logger.debug(f"Cleaned up temp directory: {temp_subdir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory {temp_subdir}: {e}")
    
    def cleanup_category(self, category: str = "general") -> int:
        """
        Clean up all temp files in a category.
        
        Args:
            category: Category to clean ("ui", "video", "gif", "image", "all")
            
        Returns:
            Number of files/dirs deleted
        """
        deleted_count = 0
        
        if category == "all":
            dirs_to_clean = [self.ui_dir, self.video_dir, self.gif_dir, self.image_dir]
        else:
            dirs_to_clean = [self.get_temp_dir(category)]
        
        for temp_dir in dirs_to_clean:
            if not temp_dir.exists():
                continue
                
            try:
                for item in temp_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                            deleted_count += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {item}: {e}")
                
                logger.info(f"Cleaned up {deleted_count} temp items from {category}")
            except Exception as e:
                logger.error(f"Failed to cleanup category {category}: {e}")
        
        return deleted_count
    
    def cleanup_all(self) -> int:
        """
        Clean up all temp files across all categories.
        
        Returns:
            Number of files/dirs deleted
        """
        return self.cleanup_category("all")
    
    def get_size_mb(self, category: str = "all") -> float:
        """
        Get total size of temp files in MB.
        
        Args:
            category: Category to check ("ui", "video", "gif", "image", "all")
            
        Returns:
            Total size in MB
        """
        total_bytes = 0
        
        if category == "all":
            dirs_to_check = [self.ui_dir, self.video_dir, self.gif_dir, self.image_dir]
        else:
            dirs_to_check = [self.get_temp_dir(category)]
        
        for temp_dir in dirs_to_check:
            if not temp_dir.exists():
                continue
                
            for item in temp_dir.rglob("*"):
                if item.is_file():
                    total_bytes += item.stat().st_size
        
        return total_bytes / (1024 * 1024)


# Global instance
_temp_manager = None


def get_temp_manager() -> TempManager:
    """Get the global TempManager instance."""
    global _temp_manager
    if _temp_manager is None:
        _temp_manager = TempManager()
    return _temp_manager
