"""
Cleanup manager for removing old temporary and output files.

This module provides automatic cleanup of old files to prevent
disk space from being consumed by stale outputs.
"""

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from utils.config_manager import config

logger = logging.getLogger("FaceOff")


class CleanupManager:
    """
    Manages cleanup of old temporary and output files.

    Provides configurable cleanup based on file age and size limits.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        temp_dir: str = "temp",
        max_age_hours: float = 24,
        enabled: bool = True
    ):
        """
        Initialize cleanup manager.

        Args:
            output_dir: Directory containing output files
            temp_dir: Directory containing temporary files
            max_age_hours: Maximum file age in hours before cleanup
            enabled: Whether cleanup is enabled
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.max_age_hours = max_age_hours
        self.enabled = enabled

    def _get_file_age_hours(self, file_path: Path) -> float:
        """Get file age in hours."""
        try:
            mtime = file_path.stat().st_mtime
            age_seconds = time.time() - mtime
            return age_seconds / 3600
        except OSError:
            return 0

    def _is_file_old(self, file_path: Path) -> bool:
        """Check if file exceeds max age."""
        return self._get_file_age_hours(file_path) > self.max_age_hours

    def cleanup_directory(
        self,
        directory: Path,
        extensions: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> Tuple[int, int, float]:
        """
        Clean up old files in a directory.

        Args:
            directory: Directory to clean
            extensions: Optional list of file extensions to clean (e.g., ['.png', '.mp4'])
            dry_run: If True, only report what would be deleted

        Returns:
            Tuple of (files_deleted, files_skipped, bytes_freed)
        """
        if not self.enabled:
            return 0, 0, 0

        if not directory.exists():
            return 0, 0, 0

        files_deleted = 0
        files_skipped = 0
        bytes_freed = 0.0

        try:
            for item in directory.rglob('*'):
                if not item.is_file():
                    continue

                # Check extension filter
                if extensions and item.suffix.lower() not in extensions:
                    continue

                # Check age
                if not self._is_file_old(item):
                    files_skipped += 1
                    continue

                file_size = item.stat().st_size

                if dry_run:
                    logger.info("[DRY RUN] Would delete: %s (%.2f MB, %.1f hours old)",
                               item, file_size / (1024*1024), self._get_file_age_hours(item))
                else:
                    try:
                        item.unlink()
                        files_deleted += 1
                        bytes_freed += file_size
                        logger.debug("Deleted old file: %s", item)
                    except OSError as e:
                        logger.warning("Failed to delete %s: %s", item, e)
                        files_skipped += 1

        except Exception as e:
            logger.error("Error during cleanup of %s: %s", directory, e)

        return files_deleted, files_skipped, bytes_freed

    def cleanup_outputs(self, dry_run: bool = False) -> Tuple[int, int, float]:
        """
        Clean up old output files.

        Returns:
            Tuple of (files_deleted, files_skipped, bytes_freed)
        """
        total_deleted = 0
        total_skipped = 0
        total_bytes = 0.0

        # Clean each output subdirectory
        for subdir in ['image', 'gif', 'video']:
            dir_path = self.output_dir / subdir
            deleted, skipped, bytes_freed = self.cleanup_directory(
                dir_path,
                dry_run=dry_run
            )
            total_deleted += deleted
            total_skipped += skipped
            total_bytes += bytes_freed

        if total_deleted > 0:
            logger.info("Output cleanup: deleted %d files, freed %.2f MB",
                       total_deleted, total_bytes / (1024*1024))

        return total_deleted, total_skipped, total_bytes

    def cleanup_temp(self, dry_run: bool = False) -> Tuple[int, int, float]:
        """
        Clean up old temporary files.

        Returns:
            Tuple of (files_deleted, files_skipped, bytes_freed)
        """
        deleted, skipped, bytes_freed = self.cleanup_directory(
            self.temp_dir,
            dry_run=dry_run
        )

        if deleted > 0:
            logger.info("Temp cleanup: deleted %d files, freed %.2f MB",
                       deleted, bytes_freed / (1024*1024))

        # Also clean empty directories
        self._cleanup_empty_dirs(self.temp_dir)

        return deleted, skipped, bytes_freed

    def _cleanup_empty_dirs(self, directory: Path):
        """Remove empty directories recursively."""
        if not directory.exists():
            return

        for item in list(directory.rglob('*')):
            if item.is_dir():
                try:
                    if not any(item.iterdir()):
                        item.rmdir()
                        logger.debug("Removed empty directory: %s", item)
                except OSError:
                    pass

    def cleanup_all(self, dry_run: bool = False) -> Tuple[int, int, float]:
        """
        Clean up all old files (outputs and temp).

        Returns:
            Tuple of (files_deleted, files_skipped, bytes_freed)
        """
        out_deleted, out_skipped, out_bytes = self.cleanup_outputs(dry_run)
        temp_deleted, temp_skipped, temp_bytes = self.cleanup_temp(dry_run)

        return (
            out_deleted + temp_deleted,
            out_skipped + temp_skipped,
            out_bytes + temp_bytes
        )

    def get_disk_usage(self) -> Tuple[float, float]:
        """
        Get disk usage for output and temp directories.

        Returns:
            Tuple of (output_size_mb, temp_size_mb)
        """
        output_size = self._get_directory_size(self.output_dir)
        temp_size = self._get_directory_size(self.temp_dir)

        return output_size / (1024*1024), temp_size / (1024*1024)

    def _get_directory_size(self, directory: Path) -> float:
        """Get total size of files in directory in bytes."""
        if not directory.exists():
            return 0

        total = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except OSError:
            pass

        return total


# Module-level cleanup manager instance
_cleanup_manager: Optional[CleanupManager] = None


def get_cleanup_manager() -> CleanupManager:
    """Get the global cleanup manager instance."""
    global _cleanup_manager
    if _cleanup_manager is None:
        # Read from config
        enabled = config.get('cleanup', 'enabled', default=True)
        max_age = config.get('cleanup', 'max_age_hours', default=24)
        _cleanup_manager = CleanupManager(
            max_age_hours=max_age,
            enabled=enabled
        )
    return _cleanup_manager


def run_cleanup(dry_run: bool = False) -> Tuple[int, int, float]:
    """
    Run cleanup using the global manager.

    Returns:
        Tuple of (files_deleted, files_skipped, bytes_freed)
    """
    return get_cleanup_manager().cleanup_all(dry_run)
