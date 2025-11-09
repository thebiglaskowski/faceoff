"""
Model caching utilities for FaceOff.

This module handles caching of TensorRT engine files to eliminate
first-run compilation delays. TensorRT engines are hardware-specific
and must be recompiled if the GPU changes.
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
from utils.config_manager import config

logger = logging.getLogger("FaceOff")


class ModelCache:
    """Manages caching of compiled TensorRT engines."""
    
    def __init__(self):
        """Initialize model cache."""
        self.cache_dir = Path(config.tensorrt_cache_dir)
        self.enabled = config.tensorrt_cache_enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Model cache initialized: %s", self.cache_dir)
        else:
            logger.info("Model cache disabled in config")
    
    def _get_cache_key(self, model_path: str, device_id: int, **kwargs) -> str:
        """
        Generate a unique cache key for a model.
        
        Args:
            model_path: Path to the model file
            device_id: GPU device ID
            **kwargs: Additional parameters that affect compilation (fp16, workspace_size, etc.)
            
        Returns:
            Unique cache key string
        """
        # Include model path, device ID, and any compilation parameters
        key_parts = [
            str(model_path),
            f"device_{device_id}",
        ]
        
        # Add any additional parameters
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{v}")
        
        # Create hash
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{Path(model_path).stem}_{key_hash}.cache"
    
    def get_cached_engine(self, model_path: str, device_id: int, **kwargs) -> Optional[bytes]:
        """
        Retrieve a cached TensorRT engine if available.
        
        Args:
            model_path: Path to the model file
            device_id: GPU device ID
            **kwargs: Additional compilation parameters
            
        Returns:
            Serialized TensorRT engine bytes, or None if not cached
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(model_path, device_id, **kwargs)
        cache_file = self.cache_dir / cache_key
        
        if cache_file.exists():
            try:
                logger.debug("Loading cached TensorRT engine: %s", cache_key)
                with open(cache_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning("Failed to load cached engine %s: %s", cache_key, e)
                return None
        
        return None
    
    def save_engine(self, model_path: str, device_id: int, engine_bytes: bytes, **kwargs) -> bool:
        """
        Save a compiled TensorRT engine to cache.
        
        Args:
            model_path: Path to the model file
            device_id: GPU device ID
            engine_bytes: Serialized TensorRT engine
            **kwargs: Additional compilation parameters
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        cache_key = self._get_cache_key(model_path, device_id, **kwargs)
        cache_file = self.cache_dir / cache_key
        
        try:
            with open(cache_file, 'wb') as f:
                f.write(engine_bytes)
            logger.info("Cached TensorRT engine: %s (%.2f MB)", 
                       cache_key, len(engine_bytes) / 1024 / 1024)
            return True
        except Exception as e:
            logger.error("Failed to cache engine %s: %s", cache_key, e)
            return False
    
    def clear_cache(self) -> int:
        """
        Clear all cached TensorRT engines.
        
        Returns:
            Number of cache files deleted
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning("Failed to delete cache file %s: %s", cache_file, e)
        
        logger.info("Cleared %d cached engine(s)", count)
        return count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.
        
        Returns:
            Dict with cache statistics
        """
        if not self.cache_dir.exists():
            return {
                'enabled': self.enabled,
                'cache_dir': str(self.cache_dir),
                'num_files': 0,
                'total_size_mb': 0.0
            }
        
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'enabled': self.enabled,
            'cache_dir': str(self.cache_dir),
            'num_files': len(cache_files),
            'total_size_mb': total_size / 1024 / 1024
        }


# Global cache instance
model_cache = ModelCache()


def preload_models(device_id: int = 0) -> None:
    """
    Preload models at startup to reduce first-run delay.
    
    This function can be called during application startup to compile
    TensorRT engines in the background before the user makes a request.
    
    Args:
        device_id: GPU device ID to use for preloading
    """
    if not config.preload_on_startup:
        logger.info("Model preloading disabled in config")
        return
    
    logger.info("Starting model preloading on device %d...", device_id)
    
    try:
        # Import here to avoid circular dependencies
        from core.media_processor import MediaProcessor
        import numpy as np
        
        # Create a dummy processor to trigger TensorRT compilation
        logger.info("Preloading face detection model (buffalo_l)...")
        processor = MediaProcessor(device_id=device_id, use_tensorrt=True, optimize_models=False)
        
        # Create a small dummy image to trigger model initialization
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Trigger face detection to compile TensorRT engine
        _ = processor.get_faces(dummy_image)
        
        logger.info("✅ Model preloading complete (TensorRT engines compiled and cached)")
        
    except Exception as e:
        logger.warning("Model preloading failed: %s (will compile on first use)", e)


def clear_model_cache() -> int:
    """
    Clear all cached TensorRT engines.
    
    Returns:
        Number of cache files deleted
    """
    return model_cache.clear_cache()


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the model cache.
    
    Returns:
        Dict with cache statistics
    """
    return model_cache.get_cache_info()
