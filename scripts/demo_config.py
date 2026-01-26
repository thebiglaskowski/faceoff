"""Test script to validate config system."""
from utils.config_manager import config, get_model_options

print("=" * 60)
print("FaceOff Config System Test")
print("=" * 60)

# Test basic config loading
print("\n✅ Config loaded successfully!")

# Test limits section
print("\n[Limits]")
print(f"  Max file size: {config.max_file_size_mb} MB")
print(f"  Max video duration: {config.max_video_duration_sec} sec")
print(f"  Max image pixels: {config.max_image_pixels:,}")
print(f"  Max GIF frames: {config.max_gif_frames}")

# Test GPU section
print("\n[GPU]")
print(f"  Batch size: {config.batch_size}")
print(f"  Max batch size: {config.max_batch_size}")
print(f"  TensorRT enabled: {config.tensorrt_enabled}")
print(f"  TensorRT FP16: {config.tensorrt_fp16}")
print(f"  TensorRT workspace: {config.tensorrt_workspace_mb} MB")

# Test face detection section
print("\n[Face Detection]")
print(f"  Confidence threshold: {config.face_confidence_threshold}")
print(f"  Adaptive enabled: {config.adaptive_detection_enabled}")
print(f"  Detection scale: {config.detection_scale}")
print(f"  Min resolution: {config.min_detection_resolution}")

# Test enhancement section
print("\n[Enhancement]")
print(f"  Default model: {config.default_enhancement_model}")
print(f"  Default tile size: {config.default_tile_size}")
print(f"  Default outscale: {config.default_outscale}")
print(f"  Default use FP32: {config.default_use_fp32}")

# Test model options
print("\n[Model Options]")
models = get_model_options()
for name, details in models.items():
    denoise = " (supports denoise)" if details['supports_denoise'] else ""
    print(f"  {name}{denoise}")
    print(f"    Model: {details['model_name']}")
    print(f"    Description: {details['description']}")

# Test logging section
print("\n[Logging]")
print(f"  Log file: {config.log_file}")
print(f"  Max file size: {config.log_max_file_size_mb} MB")
print(f"  Backup count: {config.log_backup_count}")
print(f"  Console level: {config.log_console_level}")
print(f"  File level: {config.log_file_level}")

# Test model cache section
print("\n[Model Cache]")
print(f"  TensorRT cache dir: {config.tensorrt_cache_dir}")
print(f"  TensorRT cache enabled: {config.tensorrt_cache_enabled}")
print(f"  Preload on startup: {config.preload_on_startup}")

# Test memory section
print("\n[Memory]")
print(f"  Auto clear cache: {config.auto_clear_cache}")
print(f"  Clear cache threshold: {config.clear_cache_threshold_mb} MB")
print(f"  Reduce batch on OOM: {config.reduce_batch_on_oom}")
print(f"  Min batch size: {config.min_batch_size}")

# Test file formats
print("\n[File Formats]")
print(f"  Image formats: {', '.join(config.supported_image_formats)}")
print(f"  Video formats: {', '.join(config.supported_video_formats)}")
print(f"  GIF formats: {', '.join(config.supported_gif_formats)}")

# Test directories
print("\n[Directories]")
print(f"  Temp GIF frames: {config.temp_gif_frames_dir}")
print(f"  Temp GIF enhanced: {config.temp_gif_enhanced_dir}")
print(f"  Output: {config.output_dir}")
print(f"  Models: {config.models_dir}")
print(f"  Cache: {config.cache_dir}")

# Test UI section
print("\n[UI]")
print(f"  Server name: {config.server_name}")
print(f"  Server port: {config.server_port}")
print(f"  Share: {config.share}")
print(f"  Theme: {config.theme}")

print("\n" + "=" * 60)
print("✅ All config sections validated successfully!")
print("=" * 60)
