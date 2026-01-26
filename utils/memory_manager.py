"""
Memory management utilities for FaceOff.

This module handles VRAM monitoring, automatic cache clearing,
and adaptive batch size adjustment to prevent OOM errors.
"""

import logging
import time
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from utils.config_manager import config

logger = logging.getLogger("FaceOff")


@dataclass
class CachedMemoryStats:
    """Cached memory statistics with timestamp."""
    stats: Dict[str, float]
    timestamp: float

    def is_valid(self, max_age_seconds: float = 0.5) -> bool:
        """Check if cache is still valid."""
        return (time.time() - self.timestamp) < max_age_seconds


class MemoryManager:
    """Manages GPU memory to prevent OOM errors."""

    # Class-level cache for memory stats (shared across instances per device)
    _stats_cache: Dict[int, CachedMemoryStats] = {}
    _cache_max_age: float = 0.5  # Cache valid for 500ms

    def __init__(self, device_id: int = 0, cache_stats: bool = True):
        """
        Initialize memory manager.

        Args:
            device_id: GPU device ID to monitor
            cache_stats: Whether to cache memory stats for performance
        """
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.auto_clear = config.auto_clear_cache
        self.clear_threshold_mb = config.clear_cache_threshold_mb
        self.reduce_batch_on_oom = config.reduce_batch_on_oom
        self.min_batch_size = config.min_batch_size
        self.cache_stats = cache_stats
        self.mb_per_batch = config.mb_per_batch_estimate

        logger.info("MemoryManager initialized for device %d (auto_clear=%s, threshold=%dMB)",
                   device_id, self.auto_clear, self.clear_threshold_mb)
    
    def get_memory_stats(self, use_cache: bool = True) -> Dict[str, float]:
        """
        Get current GPU memory statistics.

        Args:
            use_cache: Whether to use cached stats if available (default True)

        Returns:
            Dict with memory stats in MB
        """
        if not torch.cuda.is_available():
            return {
                'allocated_mb': 0,
                'reserved_mb': 0,
                'free_mb': 0,
                'total_mb': 0,
                'utilization_pct': 0
            }

        # Check cache if enabled
        if self.cache_stats and use_cache:
            cached = self._stats_cache.get(self.device_id)
            if cached and cached.is_valid(self._cache_max_age):
                return cached.stats

        # Fetch fresh stats
        allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024
        free = total - allocated
        utilization = (allocated / total * 100) if total > 0 else 0

        stats = {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total,
            'utilization_pct': utilization
        }

        # Update cache
        if self.cache_stats:
            self._stats_cache[self.device_id] = CachedMemoryStats(
                stats=stats,
                timestamp=time.time()
            )

        return stats

    def invalidate_cache(self) -> None:
        """Invalidate the memory stats cache for this device."""
        if self.device_id in self._stats_cache:
            del self._stats_cache[self.device_id]

    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear all cached memory stats."""
        cls._stats_cache.clear()
    
    def should_clear_cache(self) -> bool:
        """
        Check if cache should be cleared based on memory usage.
        
        Returns:
            True if cache should be cleared
        """
        if not self.auto_clear:
            return False
        
        stats = self.get_memory_stats()
        allocated_mb = stats['allocated_mb']
        
        return allocated_mb > self.clear_threshold_mb
    
    def clear_cache(self, force: bool = False) -> None:
        """
        Clear CUDA cache if needed or forced.

        Args:
            force: Force cache clear even if threshold not exceeded
        """
        if not torch.cuda.is_available():
            return

        stats_before = self.get_memory_stats(use_cache=False)

        if force or self.should_clear_cache():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

            # Invalidate cache after clearing
            self.invalidate_cache()

            stats_after = self.get_memory_stats(use_cache=False)
            freed_mb = stats_before['reserved_mb'] - stats_after['reserved_mb']

            logger.info("CUDA cache cleared on device %d: freed %.2f MB (%.1f%% → %.1f%% utilization)",
                       self.device_id, freed_mb,
                       stats_before['utilization_pct'], stats_after['utilization_pct'])
    
    def get_optimal_batch_size(self, current_batch_size: int, available_vram_mb: Optional[float] = None) -> int:
        """
        Calculate optimal batch size based on available VRAM.

        Args:
            current_batch_size: Current batch size
            available_vram_mb: Optional override for available VRAM

        Returns:
            Recommended batch size
        """
        stats = self.get_memory_stats()
        available = available_vram_mb or stats['free_mb']

        # Use configurable MB per batch estimate (default 500MB)
        # This is conservative and depends on image resolution
        max_batch_size = max(self.min_batch_size, int(available / self.mb_per_batch))
        
        # Cap at configured max
        max_batch_size = min(max_batch_size, config.max_batch_size)
        
        # If current batch is already optimal, keep it
        if current_batch_size <= max_batch_size:
            return current_batch_size
        
        logger.info("Reducing batch size: %d → %d (available VRAM: %.0f MB)",
                   current_batch_size, max_batch_size, available)
        return max_batch_size
    
    def handle_oom_error(self, current_batch_size: int) -> Tuple[int, bool]:
        """
        Handle OOM error by reducing batch size and clearing cache.
        
        Args:
            current_batch_size: Batch size that caused OOM
            
        Returns:
            Tuple of (new_batch_size, should_retry)
        """
        logger.warning("OOM error detected with batch_size=%d", current_batch_size)
        
        # Clear cache first
        self.clear_cache(force=True)
        
        if not self.reduce_batch_on_oom:
            logger.error("OOM recovery disabled in config - cannot continue")
            return current_batch_size, False
        
        # Reduce batch size
        new_batch_size = max(self.min_batch_size, current_batch_size // 2)
        
        if new_batch_size < self.min_batch_size:
            logger.error("Batch size cannot be reduced below minimum (%d) - OOM unrecoverable",
                        self.min_batch_size)
            return self.min_batch_size, False
        
        logger.info("Reducing batch size for OOM recovery: %d → %d",
                   current_batch_size, new_batch_size)
        return new_batch_size, True
    
    def log_memory_stats(self, prefix: str = "") -> None:
        """
        Log current memory statistics.
        
        Args:
            prefix: Optional prefix for log message
        """
        stats = self.get_memory_stats()
        logger.info("%sGPU Memory (device %d): %.0f/%.0f MB allocated (%.1f%%), %.0f MB free",
                   prefix + " " if prefix else "",
                   self.device_id,
                   stats['allocated_mb'],
                   stats['total_mb'],
                   stats['utilization_pct'],
                   stats['free_mb'])


# Context manager for automatic memory management
class AutoMemoryManager:
    """Context manager for automatic memory management during operations."""
    
    def __init__(self, device_id: int = 0, clear_on_exit: bool = True):
        """
        Initialize auto memory manager.
        
        Args:
            device_id: GPU device ID
            clear_on_exit: Whether to clear cache on exit
        """
        self.manager = MemoryManager(device_id)
        self.clear_on_exit = clear_on_exit
    
    def __enter__(self):
        """Enter context - log initial memory state."""
        self.manager.log_memory_stats("Before operation:")
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - optionally clear cache and log final state."""
        if self.clear_on_exit:
            self.manager.clear_cache()
        self.manager.log_memory_stats("After operation:")
        return False  # Don't suppress exceptions


def clear_cuda_cache(device_id: int = 0) -> None:
    """
    Convenience function to clear CUDA cache.
    
    Args:
        device_id: GPU device ID
    """
    manager = MemoryManager(device_id)
    manager.clear_cache(force=True)


def get_memory_stats(device_id: int = 0) -> dict:
    """
    Convenience function to get memory stats.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Dict with memory statistics
    """
    manager = MemoryManager(device_id)
    return manager.get_memory_stats()
