"""
Configuration management for FaceOff application.

This module handles loading and accessing configuration from config.yaml,
with fallbacks to default values if config is missing or invalid.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager with fallback to defaults."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: Path = Path(__file__).parent.parent / "config.yaml"  # Root directory, not utils/
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Configuration loaded from {self._config_path}")
            else:
                logger.warning(f"Config file not found: {self._config_path}. Using defaults.")
                self._config = {}
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            self._config = {}
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Keys to traverse the config hierarchy
            default: Default value if key not found
            
        Example:
            config.get('gpu', 'batch_size', default=4)
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    # =============================================================================
    # Convenience properties for commonly accessed values
    # =============================================================================
    
    # File limits
    @property
    def max_file_size_mb(self) -> int:
        return self.get('limits', 'max_file_size_mb', default=500)
    
    @property
    def max_video_duration_sec(self) -> int:
        return self.get('limits', 'max_video_duration_sec', default=300)
    
    @property
    def max_image_pixels(self) -> int:
        return self.get('limits', 'max_image_pixels', default=16777216)
    
    @property
    def max_gif_frames(self) -> int:
        return self.get('limits', 'max_gif_frames', default=500)
    
    # GPU settings
    @property
    def batch_size(self) -> int:
        return self.get('gpu', 'batch_size', default=4)
    
    @property
    def max_batch_size(self) -> int:
        return self.get('gpu', 'max_batch_size', default=16)
    
    @property
    def workers_per_gpu(self) -> int:
        return self.get('gpu', 'workers_per_gpu', default=4)
    
    @property
    def tensorrt_enabled(self) -> bool:
        return self.get('gpu', 'tensorrt_enabled', default=True)
    
    @property
    def tensorrt_fp16(self) -> bool:
        return self.get('gpu', 'tensorrt_fp16', default=True)
    
    @property
    def tensorrt_workspace_mb(self) -> int:
        return self.get('gpu', 'tensorrt_workspace_mb', default=2048)
    
    # Face detection
    @property
    def inswapper_model_path(self) -> str:
        return self.get('face_detection', 'inswapper_model_path', default='inswapper_128.onnx')
    
    @property
    def buffalo_model_path(self) -> str:
        return self.get('face_detection', 'buffalo_model_path', default='models/buffalo_l')
    
    @property
    def face_analysis_name(self) -> str:
        return self.get('face_detection', 'face_analysis_name', default='buffalo_l')
    
    @property
    def face_analysis_det_size(self) -> list:
        return self.get('face_detection', 'face_analysis_det_size', default=[640, 640])
    
    @property
    def face_confidence_threshold(self) -> float:
        return self.get('face_detection', 'confidence_threshold', default=0.5)
    
    @property
    def adaptive_detection_enabled(self) -> bool:
        return self.get('face_detection', 'adaptive_enabled', default=True)
    
    @property
    def detection_scale(self) -> float:
        return self.get('face_detection', 'detection_scale', default=0.5)
    
    @property
    def min_detection_resolution(self) -> int:
        return self.get('face_detection', 'min_resolution', default=640)
    
    # Enhancement
    @property
    def default_enhancement_model(self) -> str:
        return self.get('enhancement', 'default_model', default='RealESRGAN_x4plus')
    
    @property
    def enhancement_models(self) -> Dict[str, Dict[str, Any]]:
        return self.get('enhancement', 'models', default={})
    
    @property
    def default_tile_size(self) -> int:
        return self.get('enhancement', 'defaults', 'tile_size', default=256)
    
    @property
    def default_outscale(self) -> int:
        return self.get('enhancement', 'defaults', 'outscale', default=4)
    
    @property
    def default_pre_pad(self) -> int:
        return self.get('enhancement', 'defaults', 'pre_pad', default=0)
    
    @property
    def default_use_fp32(self) -> bool:
        return self.get('enhancement', 'defaults', 'use_fp32', default=False)
    
    @property
    def default_denoise_strength(self) -> float:
        return self.get('enhancement', 'defaults', 'denoise_strength', default=0.5)
    
    # Face restoration
    @property
    def gfpgan_enabled_by_default(self) -> bool:
        return self.get('face_restoration', 'enabled_by_default', default=False)
    
    @property
    def gfpgan_model_version(self) -> str:
        return self.get('face_restoration', 'model_version', default='1.3')
    
    @property
    def gfpgan_default_weight(self) -> float:
        return self.get('face_restoration', 'default_weight', default=0.5)
    
    # Async pipeline
    @property
    def async_pipeline_enabled(self) -> bool:
        return self.get('async_pipeline', 'enabled', default=True)
    
    @property
    def async_min_frames_threshold(self) -> int:
        return self.get('async_pipeline', 'min_frames_threshold', default=10)
    
    # Logging
    @property
    def log_file(self) -> str:
        return self.get('logging', 'log_file', default='app.log')
    
    @property
    def log_max_file_size_mb(self) -> int:
        return self.get('logging', 'max_file_size_mb', default=10)
    
    @property
    def log_backup_count(self) -> int:
        return self.get('logging', 'backup_count', default=5)
    
    @property
    def log_console_level(self) -> str:
        return self.get('logging', 'console_level', default='INFO')
    
    @property
    def log_file_level(self) -> str:
        return self.get('logging', 'file_level', default='DEBUG')
    
    @property
    def log_format(self) -> str:
        return self.get('logging', 'format', default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    @property
    def log_date_format(self) -> str:
        return self.get('logging', 'date_format', default='%Y-%m-%d %H:%M:%S')
    
    # Model cache
    @property
    def tensorrt_cache_dir(self) -> str:
        return self.get('model_cache', 'tensorrt_cache_dir', default='cache/tensorrt')
    
    @property
    def tensorrt_cache_enabled(self) -> bool:
        return self.get('model_cache', 'tensorrt_cache_enabled', default=True)
    
    @property
    def preload_on_startup(self) -> bool:
        return self.get('model_cache', 'preload_on_startup', default=False)
    
    @property
    def preload_models(self) -> list:
        return self.get('model_cache', 'preload_models', default=['buffalo_l', 'inswapper_128'])
    
    # Memory management
    @property
    def auto_clear_cache(self) -> bool:
        return self.get('memory', 'auto_clear_cache', default=True)
    
    @property
    def clear_cache_threshold_mb(self) -> int:
        return self.get('memory', 'clear_cache_threshold_mb', default=1024)
    
    @property
    def reduce_batch_on_oom(self) -> bool:
        return self.get('memory', 'reduce_batch_on_oom', default=True)
    
    @property
    def min_batch_size(self) -> int:
        return self.get('memory', 'min_batch_size', default=1)
    
    # File formats
    @property
    def supported_image_formats(self) -> list:
        return self.get('file_formats', 'images', default=['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
    
    @property
    def supported_video_formats(self) -> list:
        return self.get('file_formats', 'videos', default=['.mp4', '.webp', '.avi', '.mov'])
    
    @property
    def supported_gif_formats(self) -> list:
        return self.get('file_formats', 'gifs', default=['.gif'])
    
    # Directories
    @property
    def temp_gif_frames_dir(self) -> str:
        return self.get('directories', 'temp_gif_frames', default='temp_gif_frames')
    
    @property
    def temp_gif_enhanced_dir(self) -> str:
        return self.get('directories', 'temp_gif_enhanced', default='temp_gif_enhanced')
    
    @property
    def output_dir(self) -> str:
        return self.get('directories', 'output', default='outputs')
    
    @property
    def models_dir(self) -> str:
        return self.get('directories', 'models', default='models')
    
    @property
    def cache_dir(self) -> str:
        return self.get('directories', 'cache', default='cache')
    
    # UI settings
    @property
    def ui_server_name(self) -> str:
        return self.get('ui', 'server_name', default='127.0.0.1')
    
    @property
    def server_name(self) -> str:
        """Alias for ui_server_name."""
        return self.ui_server_name
    
    @property
    def ui_server_port(self) -> int:
        return self.get('ui', 'server_port', default=7860)
    
    @property
    def server_port(self) -> int:
        """Alias for ui_server_port."""
        return self.ui_server_port
    
    @property
    def ui_share(self) -> bool:
        return self.get('ui', 'share', default=False)
    
    @property
    def share(self) -> bool:
        """Alias for ui_share."""
        return self.ui_share
    
    @property
    def ui_theme(self) -> str:
        return self.get('ui', 'theme', default='default')
    
    @property
    def theme(self) -> str:
        """Alias for ui_theme."""
        return self.ui_theme


# Global config instance
config = Config()


# Helper function for getting model options in UI format
def get_model_options() -> Dict[str, Dict[str, Any]]:
    """
    Get enhancement models in UI-compatible format.
    
    Returns:
        Dict mapping display names to model configurations
        Example: {"RealESRGAN_x4plus (General)": {"model_name": "RealESRGAN_x4plus", ...}}
    """
    models = config.enhancement_models
    if not models:
        # Fallback to defaults if config fails to load
        return {
            "RealESRGAN_x4plus (General - Best for Photos)": {
                "model_name": "RealESRGAN_x4plus",
                "supports_denoise": False,
                "description": "General purpose, best for photorealistic images"
            },
            "RealESRGAN_x4plus_anime_6B (Anime/Illustrations)": {
                "model_name": "RealESRGAN_x4plus_anime_6B",
                "supports_denoise": False,
                "description": "Optimized for anime and illustrated content"
            },
            "RealESRNet_x4plus (Conservative)": {
                "model_name": "RealESRNet_x4plus",
                "supports_denoise": False,
                "description": "More conservative, fewer artifacts"
            },
            "realesr-general-x4v3 (With Denoise)": {
                "model_name": "realesr-general-x4v3",
                "supports_denoise": True,
                "description": "General model with adjustable denoising"
            },
            "realesr-animevideov3 (Anime Video)": {
                "model_name": "realesr-animevideov3",
                "supports_denoise": False,
                "description": "Designed specifically for anime videos"
            },
            "RealESRGAN_x2plus (Fast 2x)": {
                "model_name": "RealESRGAN_x2plus",
                "supports_denoise": False,
                "description": "Faster 2x upscaling"
            }
        }
    
    # Build options dict from config
    result = {}
    for model_name, model_config in models.items():
        display_name = model_config.get('display_name', model_name)
        result[display_name] = {
            'model_name': model_name,
            'supports_denoise': model_config.get('supports_denoise', False),
            'description': model_config.get('description', '')
        }
    return result
