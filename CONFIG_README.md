# Configuration System

## Overview

FaceOff now uses a centralized configuration system via `config.yaml`. This makes it easy to customize settings without modifying code.

## Files

- **`config.yaml`** - Main configuration file (user-editable)
- **`utils/config_manager.py`** - Configuration loader and manager
- **`utils/constants.py`** - Backward compatibility layer (imports from config)

## Quick Start

1. Copy `config.yaml` to customize settings
2. Edit values as needed
3. Restart the application

## Configuration Sections

### limits
File and processing limits:
```yaml
max_file_size_mb: 500
max_video_duration_sec: 300
max_image_pixels: 16777216
max_gif_frames: 500
```

### gpu
GPU processing settings:
```yaml
batch_size: 4                # Frames processed per batch
max_batch_size: 16           # Maximum batch size
tensorrt_enabled: true       # Enable TensorRT optimization
tensorrt_fp16: true          # Use FP16 precision
tensorrt_workspace_mb: 2048  # TensorRT workspace size
```

### face_detection
Face detection configuration:
```yaml
confidence_threshold: 0.5    # Minimum face detection confidence
adaptive_enabled: true       # Use resolution-adaptive detection
detection_scale: 0.5         # Detection scale (0.25-1.0)
min_resolution: 640          # Minimum resolution for detection
```

### enhancement
Real-ESRGAN enhancement settings:
```yaml
default_model: "RealESRGAN_x4plus"
default_tile_size: 256
default_outscale: 4
default_pre_pad: 0
default_use_fp32: false
default_denoise_strength: 0.5

models:
  - name: "RealESRGAN_x4plus"
    scale: 4
    description: "General 4x upscaler"
  - name: "RealESRNet_x4plus"
    scale: 4
    description: "Sharper 4x upscaler (no GAN)"
  # ... more models
```

### face_restoration
GFPGAN face restoration:
```yaml
enabled_by_default: false
model_version: "1.3"
default_weight: 0.5
```

### async_pipeline
Async processing configuration:
```yaml
enabled: true
min_frames_threshold: 10  # Minimum frames to use async pipeline
```

### logging
Logging settings (prepared for rotation feature):
```yaml
log_file: "app.log"
max_file_size_mb: 10
backup_count: 5
console_level: "INFO"
file_level: "DEBUG"
```

### model_cache
Model caching (prepared for caching feature):
```yaml
tensorrt_cache_dir: "cache/tensorrt"
tensorrt_cache_enabled: true
preload_on_startup: false
```

### memory
Memory management (prepared for OOM handling):
```yaml
auto_clear_cache: true
clear_cache_threshold_mb: 1024
reduce_batch_on_oom: true
min_batch_size: 1
```

### file_formats
Supported file formats:
```yaml
supported_image_formats:
  - ".jpg"
  - ".jpeg"
  - ".png"
  - ".bmp"
  - ".webp"
# ... video and GIF formats
```

### directories
Temporary directory names:
```yaml
temp_gif_frames_dir: "temp_gif_frames"
temp_gif_enhanced_dir: "temp_gif_enhanced"
output_dir: "outputs"
models_dir: "models"
cache_dir: "cache"
```

### ui
Gradio UI settings:
```yaml
server_name: "127.0.0.1"
server_port: 7860
share: false
theme: "default"
```

## Usage in Code

### Importing
```python
from utils.config_manager import config
```

### Accessing Values
```python
# Direct property access (recommended)
batch_size = config.batch_size
max_file_size = config.max_file_size_mb

# Nested access with default
value = config.get('gpu', 'batch_size', default=4)
```

### Backward Compatibility
Existing code using `utils.constants` continues to work:
```python
from utils.constants import MAX_FILE_SIZE_MB  # Still works!
```

## Adding New Settings

1. **Add to config.yaml**:
   ```yaml
   my_section:
     my_setting: value
   ```

2. **Add property to Config class** (`utils/config_manager.py`):
   ```python
   @property
   def my_setting(self) -> str:
       return self.get('my_section', 'my_setting', default='value')
   ```

3. **Use in code**:
   ```python
   from utils.config_manager import config
   value = config.my_setting
   ```

## Future Features Using Config

### 1. Logging Rotation (Next)
Uses `logging.*` settings to implement `RotatingFileHandler`:
- `max_file_size_mb`: Rotate when log exceeds this size
- `backup_count`: Number of backup logs to keep

### 2. Model Caching
Uses `model_cache.*` settings to cache TensorRT engines:
- `tensorrt_cache_dir`: Directory for cached engines
- `tensorrt_cache_enabled`: Enable/disable caching
- `preload_on_startup`: Preload models at startup

### 3. Memory Management
Uses `memory.*` settings for OOM prevention:
- `auto_clear_cache`: Automatically clear CUDA cache
- `clear_cache_threshold_mb`: Clear cache when VRAM exceeds threshold
- `reduce_batch_on_oom`: Reduce batch size on OOM errors
- `min_batch_size`: Minimum batch size

## Notes

- All settings have sensible defaults
- Config is loaded once at startup (singleton pattern)
- If `config.yaml` is missing, defaults from code are used
- Type checking ensures type safety
- Settings are validated on load

## Troubleshooting

### Config Not Loading
- Check `config.yaml` exists in root directory
- Verify YAML syntax (use online validator)
- Check console for error messages

### Invalid Values
- Config manager uses defaults if value is invalid
- Check logs for "Config error" messages
- Verify value types match expected types

### Changes Not Applied
- Restart application after editing `config.yaml`
- Config is loaded once at startup

## Performance Tips

1. **Batch Size**: Increase for better GPU utilization (4-16)
2. **Detection Scale**: Lower for faster processing (0.25-0.5)
3. **TensorRT**: Keep enabled for 2-3x speedup
4. **Async Pipeline**: Keep enabled for videos >10 frames

## Safety Tips

1. **Max File Size**: Adjust based on available RAM
2. **Max Batch Size**: Adjust based on VRAM (16 for 12GB+)
3. **Memory Settings**: Enable auto_clear_cache to prevent OOM
4. **Logging**: Enable rotation to prevent disk fill
