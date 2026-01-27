"""
Thread-safe LRU cache for model management.

Provides bounded caching with configurable size limits to prevent
unbounded memory growth when loading multiple models.
"""
import logging
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, TypeVar

from utils.config_manager import config

logger = logging.getLogger("FaceOff")

T = TypeVar('T')


class LRUModelCache:
    """
    Thread-safe LRU cache for GPU models.

    Features:
    - Configurable maximum size
    - LRU eviction when full
    - Thread-safe operations
    - Optional cleanup callback for evicted items
    """

    def __init__(
        self,
        name: str,
        max_size: Optional[int] = None,
        cleanup_fn: Optional[Callable[[Any], None]] = None
    ):
        """
        Initialize LRU cache.

        Args:
            name: Cache name for logging
            max_size: Maximum items to cache (None = use config default)
            cleanup_fn: Optional function to call when evicting items
        """
        self.name = name
        self._max_size = max_size or config.get(
            'model_cache_limits', 'max_models_per_cache', default=5
        )
        self._cleanup_fn = cleanup_fn
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    @property
    def max_size(self) -> int:
        """Get maximum cache size."""
        return self._max_size

    def get(self, key: Any) -> Optional[Any]:
        """
        Get item from cache, moving it to end (most recent).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                logger.debug("[%s] Cache hit: %s", self.name, key)
                return self._cache[key]
            return None

    def put(self, key: Any, value: Any) -> None:
        """
        Add item to cache, evicting oldest if full.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = value
                self._cache.move_to_end(key)
                return

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                logger.info(
                    "[%s] Evicting model (LRU): %s (cache size: %d/%d)",
                    self.name, oldest_key, len(self._cache), self._max_size
                )
                if self._cleanup_fn:
                    try:
                        self._cleanup_fn(oldest_value)
                    except Exception as e:
                        logger.warning(
                            "[%s] Cleanup failed for %s: %s",
                            self.name, oldest_key, e
                        )

            # Add new item
            self._cache[key] = value
            logger.debug(
                "[%s] Cached: %s (size: %d/%d)",
                self.name, key, len(self._cache), self._max_size
            )

    def remove(self, key: Any) -> bool:
        """
        Remove specific item from cache.

        Args:
            key: Cache key

        Returns:
            True if item was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                if self._cleanup_fn:
                    try:
                        self._cleanup_fn(value)
                    except Exception as e:
                        logger.warning("[%s] Cleanup failed: %s", self.name, e)
                return True
            return False

    def clear(self) -> int:
        """
        Clear all items from cache.

        Returns:
            Number of items cleared
        """
        with self._lock:
            count = len(self._cache)
            if self._cleanup_fn:
                for key, value in self._cache.items():
                    try:
                        self._cleanup_fn(value)
                    except Exception as e:
                        logger.warning(
                            "[%s] Cleanup failed for %s: %s",
                            self.name, key, e
                        )
            self._cache.clear()
            logger.info("[%s] Cache cleared (%d items)", self.name, count)
            return count

    def __len__(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache

    def keys(self) -> list:
        """Get list of cached keys (oldest to newest)."""
        with self._lock:
            return list(self._cache.keys())
