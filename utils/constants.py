"""
Application constants and configuration values.

This module now uses the centralized config manager.
Legacy constants are maintained for backward compatibility.
"""

from utils.config_manager import config, get_model_options

# File size and resource limits (from config)
MAX_FILE_SIZE_MB = config.max_file_size_mb
MAX_VIDEO_DURATION_SEC = config.max_video_duration_sec
MAX_IMAGE_PIXELS = config.max_image_pixels
MAX_GIF_FRAMES = config.max_gif_frames

# Real-ESRGAN model options (from config)
MODEL_OPTIONS = get_model_options()

# Default model (from config)
DEFAULT_MODEL = list(MODEL_OPTIONS.keys())[0] if MODEL_OPTIONS else "RealESRGAN_x4plus (General - Best for Photos)"

# Default enhancement settings (from config)
DEFAULT_TILE_SIZE = config.default_tile_size
DEFAULT_OUTSCALE = config.default_outscale
DEFAULT_PRE_PAD = config.default_pre_pad
DEFAULT_USE_FP32 = config.default_use_fp32

# Face detection settings (from config)
DEFAULT_FACE_CONFIDENCE = config.face_confidence_threshold

# Face restoration settings (GFPGAN) (from config)
DEFAULT_GFPGAN_ENABLED = config.gfpgan_enabled_by_default
DEFAULT_GFPGAN_WEIGHT = config.gfpgan_default_weight
GFPGAN_MODEL_VERSION = config.gfpgan_model_version

# Temporal smoothing settings (deprecated - kept for backward compatibility)
DEFAULT_TEMPORAL_SMOOTHING_ENABLED = False
DEFAULT_TEMPORAL_BLEND_STRENGTH = 0.1
USE_OPTICAL_FLOW = False

# TensorRT optimization settings (from config)
DEFAULT_TENSORRT_ENABLED = config.tensorrt_enabled
TENSORRT_FP16_MODE = config.tensorrt_fp16
TENSORRT_WORKSPACE_SIZE = config.tensorrt_workspace_mb * 1024 * 1024  # Convert MB to bytes

# Frame batching settings (from config)
DEFAULT_BATCH_SIZE = config.batch_size
MAX_BATCH_SIZE = config.max_batch_size

# Resolution-adaptive processing settings (from config)
DEFAULT_ADAPTIVE_DETECTION = config.adaptive_detection_enabled
DEFAULT_DETECTION_SCALE = config.detection_scale
MIN_DETECTION_RESOLUTION = config.min_detection_resolution

# Supported file types (from config)
SUPPORTED_IMAGE_FORMATS = config.supported_image_formats
SUPPORTED_VIDEO_FORMATS = config.supported_video_formats
SUPPORTED_GIF_FORMAT = config.supported_gif_formats

# Temporary directory names (from config)
TEMP_GIF_FRAMES_DIR = config.temp_gif_frames_dir
TEMP_GIF_ENHANCED_DIR = config.temp_gif_enhanced_dir

# Multi-GPU settings (from config)
WORKERS_PER_GPU = config.workers_per_gpu

# Logging (from config)
LOG_FILE = config.log_file
