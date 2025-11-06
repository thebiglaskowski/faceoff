"""
Application constants and configuration values.
"""

# File size and resource limits
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
MAX_VIDEO_DURATION_SEC = 300  # Maximum video duration (5 minutes)
MAX_IMAGE_PIXELS = 4096 * 4096  # Maximum image resolution (4K)
MAX_GIF_FRAMES = 500  # Maximum GIF frames

# Real-ESRGAN model options
MODEL_OPTIONS = {
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

# Default model
DEFAULT_MODEL = "RealESRGAN_x4plus (General - Best for Photos)"

# Default enhancement settings
DEFAULT_TILE_SIZE = 256  # Good balance of speed and VRAM usage (128-512)
DEFAULT_OUTSCALE = 4  # Standard upscaling factor (2 or 4)
DEFAULT_PRE_PAD = 0  # Pre-padding to reduce edge artifacts (0-20)
DEFAULT_USE_FP32 = False  # Use FP32 precision (more VRAM but slightly better quality)

# Face detection settings
DEFAULT_FACE_CONFIDENCE = 0.5  # Minimum confidence threshold for face detection

# Supported file types
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".webp", ".avi", ".mov"]
SUPPORTED_GIF_FORMAT = [".gif"]

# Temporary directory names
TEMP_GIF_FRAMES_DIR = "temp_gif_frames"
TEMP_GIF_ENHANCED_DIR = "temp_gif_enhanced"

# Multi-GPU settings
WORKERS_PER_GPU = 4  # Number of workers per GPU for parallel processing

# Logging
LOG_FILE = "app.log"
