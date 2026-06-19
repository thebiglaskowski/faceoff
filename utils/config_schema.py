"""
Configuration validation schema for FaceOff.

This module provides validation for config.yaml values with
reasonable defaults and warnings for invalid configurations.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

logger = logging.getLogger("FaceOff")


class ConfigValidator:
    """
    Validates configuration values against a schema.

    Uses a schema definition to validate types, ranges, and
    provide sensible defaults for invalid values.
    """

    # Schema definition: key_path -> (type, validator_func, default, description)
    # validator_func returns (is_valid, corrected_value)
    SCHEMA: Dict[str, Tuple[type, Optional[Callable], Any, str]] = {
        # Limits
        'limits.max_file_size_mb': (int, lambda v: (1 <= v <= 2000, max(1, min(v, 2000))),
                                    500, "Maximum file size in MB (1-2000)"),
        'limits.max_video_duration_sec': (int, lambda v: (1 <= v <= 3600, max(1, min(v, 3600))),
                                          300, "Maximum video duration in seconds (1-3600)"),
        'limits.max_image_pixels': (int, lambda v: (1000000 <= v <= 100000000, max(1000000, min(v, 100000000))),
                                    16777216, "Maximum image pixels (1M-100M)"),
        'limits.max_gif_frames': (int, lambda v: (1 <= v <= 2000, max(1, min(v, 2000))),
                                  500, "Maximum GIF frames (1-2000)"),

        # GPU
        'gpu.batch_size': (int, lambda v: (1 <= v <= 32, max(1, min(v, 32))),
                          4, "Batch size for processing (1-32)"),
        'gpu.max_batch_size': (int, lambda v: (1 <= v <= 64, max(1, min(v, 64))),
                               16, "Maximum batch size (1-64)"),
        'gpu.workers_per_gpu': (int, lambda v: (1 <= v <= 16, max(1, min(v, 16))),
                                4, "Worker threads per GPU (1-16)"),
        'gpu.multi_gpu_video_enabled': (bool, None, True, "Enable multi-GPU for video"),
        'gpu.tensorrt_enabled': (bool, None, True, "Enable TensorRT"),
        'gpu.tensorrt_fp16': (bool, None, True, "Use FP16 with TensorRT"),
        'gpu.tensorrt_workspace_mb': (int, lambda v: (256 <= v <= 8192, max(256, min(v, 8192))),
                                      2048, "TensorRT workspace in MB (256-8192)"),

        # Face detection
        'face_detection.confidence_threshold': (float, lambda v: (0.1 <= v <= 1.0, max(0.1, min(v, 1.0))),
                                                0.5, "Face confidence threshold (0.1-1.0)"),
        'face_detection.iou_threshold': (float, lambda v: (0.1 <= v <= 0.9, max(0.1, min(v, 0.9))),
                                         0.3, "IoU threshold for tracking (0.1-0.9)"),
        'face_detection.detection_scale': (float, lambda v: (0.25 <= v <= 1.0, max(0.25, min(v, 1.0))),
                                           0.5, "Detection scale factor (0.25-1.0)"),
        'face_detection.min_resolution': (int, lambda v: (160 <= v <= 1280, max(160, min(v, 1280))),
                                          640, "Minimum detection resolution (160-1280)"),

        # Enhancement
        'enhancement.defaults.tile_size': (int, lambda v: (v in [64, 128, 256, 512, 1024], 256),
                                           256, "Tile size (64, 128, 256, 512, 1024)"),
        'enhancement.defaults.outscale': (int, lambda v: (v in [2, 4], 4),
                                          4, "Output scale (2 or 4)"),
        'enhancement.defaults.pre_pad': (int, lambda v: (0 <= v <= 100, max(0, min(v, 100))),
                                         0, "Pre-padding (0-100)"),
        'enhancement.defaults.denoise_strength': (float, lambda v: (0.0 <= v <= 1.0, max(0.0, min(v, 1.0))),
                                                  0.5, "Denoise strength (0.0-1.0)"),

        # Face restoration
        'face_restoration.default_weight': (float, lambda v: (0.0 <= v <= 1.0, max(0.0, min(v, 1.0))),
                                            0.5, "Restoration weight (0.0-1.0)"),

        # Streaming pipeline
        'streaming.chunk_size': (int, lambda v: (1 <= v <= 256, max(1, min(v, 256))),
                               32, "Frames per streaming chunk (1-256)"),
        'streaming.gif_decode_fps': (float, lambda v: (1.0 <= v <= 60.0, max(1.0, min(v, 60.0))),
                                    10.0, "GIF decode FPS (1-60)"),

        # Memory
        'memory.clear_cache_threshold_mb': (int, lambda v: (128 <= v <= 16384, max(128, min(v, 16384))),
                                            1024, "Cache clear threshold in MB (128-16384)"),
        'memory.mb_per_batch_estimate': (int, lambda v: (100 <= v <= 2000, max(100, min(v, 2000))),
                                         500, "Estimated MB per batch (100-2000)"),
        'memory.min_batch_size': (int, lambda v: (1 <= v <= 8, max(1, min(v, 8))),
                                  1, "Minimum batch size (1-8)"),

        # UI
        'ui.server_port': (int, lambda v: (1024 <= v <= 65535, max(1024, min(v, 65535))),
                          7860, "Server port (1024-65535)"),
    }

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize validator with config dictionary.

        Args:
            config_dict: The configuration dictionary to validate
        """
        self.config = config_dict
        self.warnings: List[str] = []
        self.corrections: Dict[str, Tuple[Any, Any]] = {}  # key -> (original, corrected)

    def _get_nested_value(self, key_path: str) -> Optional[Any]:
        """Get a nested value from config using dot notation."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def validate_value(self, key_path: str) -> Optional[Any]:
        """
        Validate a single configuration value.

        Args:
            key_path: Dot-separated path to config value

        Returns:
            Validated/corrected value, or None if not in schema
        """
        if key_path not in self.SCHEMA:
            return None

        expected_type, validator_func, default, description = self.SCHEMA[key_path]
        value = self._get_nested_value(key_path)

        # Use default if value is None
        if value is None:
            return default

        # Check type
        if not isinstance(value, expected_type):
            self.warnings.append(
                f"Config '{key_path}' has wrong type: expected {expected_type.__name__}, "
                f"got {type(value).__name__}. Using default: {default}"
            )
            self.corrections[key_path] = (value, default)
            return default

        # Apply validator if present
        if validator_func:
            is_valid, corrected = validator_func(value)
            if not is_valid:
                self.warnings.append(
                    f"Config '{key_path}' value {value} is out of range for {description}. "
                    f"Corrected to: {corrected}"
                )
                self.corrections[key_path] = (value, corrected)
                return corrected

        return value

    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all configuration values in schema.

        Returns:
            Dictionary of validated values
        """
        validated = {}
        for key_path in self.SCHEMA:
            validated[key_path] = self.validate_value(key_path)
        return validated

    def get_warnings(self) -> List[str]:
        """Get list of validation warnings."""
        return self.warnings

    def get_corrections(self) -> Dict[str, Tuple[Any, Any]]:
        """Get dictionary of corrected values."""
        return self.corrections

    def log_warnings(self):
        """Log all validation warnings."""
        for warning in self.warnings:
            logger.warning(warning)


def validate_config(config_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate a configuration dictionary.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        Tuple of (validated_values, warnings)
    """
    validator = ConfigValidator(config_dict)
    validated = validator.validate_all()
    validator.log_warnings()
    return validated, validator.get_warnings()
