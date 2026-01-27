# FaceOff

**Production-Ready AI Face Swapper - Image | GIF | Video**

High-performance face swapping with GPU acceleration, intelligent caching, and optional Real-ESRGAN enhancement. Built for reliability and scale with comprehensive configuration management.

Read more at [https://thebiglaskowski.com/posts/face-swapping-with-ai](https://thebiglaskowski.com/posts/face-swapping-with-ai/)

## ✨ Features

### Core Capabilities

- **Universal Face Swapping**: Swap faces from source image to:
  - 🖼️ **Images** (PNG, JPG, WEBP, BMP)
  - 🎞️ **Animated GIFs** with frame preservation
  - 🎬 **Videos** (MP4, WEBP, AVI, MOV) with audio preservation

- **AI Enhancement**: 6 Real-ESRGAN models for quality upscaling:
  - 🎯 **RealESRGAN_x4plus**: Best for photorealistic images (Default)
  - 🎨 **RealESRGAN_x4plus_anime_6B**: Optimized for anime/illustrations
  - 💎 **RealESRNet_x4plus**: Conservative enhancement, fewer artifacts
  - 🔧 **realesr-general-x4v3**: With adjustable denoise control (0-1)
  - 🎬 **realesr-animevideov3**: Specialized for anime videos
  - ⚡ **RealESRGAN_x2plus**: Fast 2x upscaling

### Performance & Reliability

- **TensorRT Optimization**: Automatic model acceleration with persistent caching
- **Intelligent Memory Management**: Auto cache clearing, OOM recovery, dynamic batch sizing
- **Async Pipeline**: 3-stage overlapped processing (detection → swap → enhancement)
- **Multi-GPU Support**: Distribute workload across multiple CUDA devices
- **Logging Rotation**: Size-based log rotation with configurable retention
- **YAML Configuration**: Centralized settings with runtime validation

📖 **Documentation**: See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed feature documentation and [CONFIG_README.md](CONFIG_README.md) for configuration guide

## Demo Videos

### Faceoff Img2MP4 vs DeepFaceLab Deepfake

Faceoff on the left, DeepFaceLab on the right

<https://github.com/thebiglaskowski/faceoff/assets/5170343/9f30932c-bfed-4dbe-9131-eaee92a854de>

### Demo: Basic vs Enhanced (w/Real-ESRGAN)

Basic on the left, Enhanced on the right

<https://github.com/thebiglaskowski/faceoff/assets/5170343/cd2eec67-2233-4813-ae16-5d3554c61884>

Sarah Connor?

[![Watch the video](https://img.youtube.com/vi/H7KS8ZoulGw/hqdefault.jpg)](https://www.youtube.com/embed/H7KS8ZoulGw)

## Installation

### Prerequisites

1. **Visual Studio 2022** with C++ Build Tools:
   - [Download Visual Studio 2022](https://visualstudio.microsoft.com/vs/)
   - Install "Desktop development with C++" workload
   - **Important**: Use **Developer PowerShell for VS 2022** for all installation steps

2. **FFmpeg** - [Download and install FFmpeg](https://ffmpeg.org/download.html)

3. **CUDA 11.8+ or 12.x** (for GPU acceleration):
   - **Recommended**: [CUDA 12.1 or later](https://developer.nvidia.com/cuda-downloads) (latest drivers)
   - **Alternative**: [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (older but stable)
   - **Check compatibility**: Run `nvidia-smi` to see your driver's max CUDA version

4. **TensorRT** (Recommended, for 2-3x faster GPU acceleration):
   - **Modern Method**: Installed automatically with pip (see step 4 below)
   - **Manual Method**: [Download TensorRT](https://developer.nvidia.com/tensorrt) if needed
   - **Performance**: Provides 2-3x faster face detection and model inference

5. **Python 3.11** - **Recommended** for 10-60% better performance vs 3.10
   - Anaconda/Miniconda recommended for easy environment management

## Optimized Setup Instructions

⚠️ **Important**: Open **Developer PowerShell for VS 2022** before proceeding with installation.

### Quick Setup (Recommended)

1. **Clone Repository**:
```powershell
git clone https://github.com/thebiglaskowski/faceoff.git
cd faceoff
```

2. **Create Environment** (Python 3.11 for best performance):
```powershell
conda create -n faceoff python=3.11 -y
conda activate faceoff
```

3. **Install PyTorch with CUDA**:

   ⚠️ **PyTorch Version Note**: Avoid 2.4.0/2.4.1 with Python 3.11 due to typing issues.

   **For CUDA 12.1+ (Recommended - newest systems):**
   ```powershell
   # Use stable 2.3.1 or latest 2.5.0+
   pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```
   
   **For CUDA 11.8 (older systems):**
   ```powershell
   pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **CPU-only (no GPU acceleration):**
   ```powershell
   pip install torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install All Dependencies** (including TensorRT):
```powershell
pip install -r requirements.txt
```

5. **Fix BasicSR Compatibility** (if needed):
```powershell
# Run this if you get torchvision import errors
python -c "
import sys, os
site_packages = [p for p in sys.path if 'site-packages' in p][0]
file_path = os.path.join(site_packages, 'basicsr', 'data', 'degradations.py')
with open(file_path, 'r') as f: content = f.read()
content = content.replace(
    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
    'try:\\n    from torchvision.transforms.functional_tensor import rgb_to_grayscale\\nexcept ImportError:\\n    from torchvision.transforms.functional import rgb_to_grayscale'
)
with open(file_path, 'w') as f: f.write(content)
print('✅ Fixed BasicSR compatibility')
"
```

6. **Download Models**:
   - [inswapper_128.onnx](https://huggingface.co/thebiglaskowski/inswapper_128.onnx/tree/main) → place in project root
   - [buffalo_l models](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) → extract to `models/buffalo_l/`
   - Real-ESRGAN weights download automatically on first use

7. **Optional: Install gifsicle for better GIF compression**:
   - [Download gifsicle for Windows](https://www.lcdf.org/gifsicle/)
   - Extract `gifsicle.exe` to `external/gifsicle/` folder
   - Provides ~60% better GIF compression than default PIL optimization

8. **Verify Installation**:

```powershell
python -c "
import torch, onnxruntime, gradio, cv2
from realesrgan import RealESRGANer
import insightface, tensorrt
print(f'✅ Python: 3.11')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
print(f'✅ TensorRT: {tensorrt.__version__}')
print(f'✅ ONNX Providers: {onnxruntime.get_available_providers()}')
print('🚀 All dependencies working!')
"
```

9. **Fix TensorRT DLL Errors** (if you see "nvinfer_10.dll missing"):

```powershell
# Install NVIDIA CUDA libraries for TensorRT
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12
pip install nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12
```

10. **Configure** (Optional):
   - Copy `config.example.yaml` to `config.yaml` (auto-created with defaults if missing)
   - Edit settings as needed - see [CONFIG_README.md](CONFIG_README.md) for details

### Alternative Setup (Python 3.10)

If you prefer the stable Python 3.10 setup, use `python=3.10` in step 2. Performance will be 10-60% slower but may be more compatible with some systems.

### Performance Comparison: Python 3.11 vs 3.10

| Performance Metric | Python 3.10 | Python 3.11 | Improvement |
|-------------------|--------------|--------------|-------------|
| **Face Detection** | 1.2s | 0.8s | **33% faster** |
| **Face Swapping** | 2.1s | 1.4s | **33% faster** |
| **Real-ESRGAN Enhancement** | 4.5s | 3.2s | **29% faster** |
| **GIF Processing (30 frames)** | 15.2s | 10.8s | **29% faster** |
| **Memory Usage** | Higher | Lower | **15% reduction** |

*Results from RTX 3060 Ti, 1080p images. Your performance may vary.*

### Troubleshooting

**Common Issues:**

1. **"ModuleNotFoundError: No module named 'magic'"**:
   ```powershell
   pip install python-magic-bin
   ```

2. **torchvision import errors with BasicSR**:
   - Run the BasicSR compatibility fix in step 5 above

3. **NumPy compatibility warnings**:
   ```powershell
   pip install "numpy>=1.21.2,<2.0" --force-reinstall
   ```

4. **MoviePy import errors**:
   ```powershell
   pip install moviepy==1.0.3 imageio==2.31.6 --force-reinstall
   ```

5. **CUDA/GPU not detected**:
   - Verify CUDA installation: `nvidia-smi`
   - Reinstall PyTorch with correct CUDA version
   - Check GPU drivers are up to date

## Usage

### Unified Interface

```powershell
python main.py
```

Opens at <http://127.0.0.1:7860/>

The unified interface provides:

- **Image Tab**: Single image face swapping
- **GIF Tab**: Animated GIF processing with frame preservation
- **Video Tab**: Video processing with audio preservation
- **Gallery Tab**: View and manage processed outputs

### Enhancement Options

All tabs support optional Real-ESRGAN enhancement:

1. **Enable Enhancement**: Check the enhancement checkbox
2. **Select Model**: Choose based on content type (photo/anime/video)
3. **Adjust Denoise**: For `realesr-general-x4v3` model only (0 = none, 1 = maximum)
4. **Process**: Enhanced output automatically saved

## Performance & Configuration

### GPU Memory Requirements

- **12GB+ VRAM**: All models, full resolution
- **8GB VRAM**: All models except anime_6B, standard settings recommended
- **6GB VRAM**: Use x2plus or x4plus models, consider lower tile sizes
- **4GB VRAM**: Disable enhancement or use minimal settings

### Processing Time Estimates

10-second 1080p video (300 frames):

- **Face swap only**: ~30-60 seconds
- **With enhancement**: ~10-20 minutes (GPU-dependent)

*Actual times vary based on GPU, resolution, scene complexity, and selected model*

### Configuration

FaceOff uses `config.yaml` for all settings. See [CONFIG_README.md](CONFIG_README.md) for comprehensive documentation.

**Quick customization:**
```yaml
# GPU settings
gpu:
  batch_size: 4              # Frames processed per batch
  tensorrt_enabled: true     # Use TensorRT acceleration
  
# Model cache
model_cache:
  tensorrt_cache_enabled: true
  preload_on_startup: false
  
# Memory management  
memory:
  auto_clear_cache: true
  clear_cache_threshold_mb: 1024
  reduce_batch_on_oom: true
```

The config system includes validation, defaults, and automatic migration. Changes take effect on next run.

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory` during processing

**Solutions**:
- Use `RealESRGAN_x2plus` model (lower memory requirements)
- Reduce `batch_size` in `config.yaml`
- Process shorter videos or lower resolution inputs
- Close other GPU applications
- Check GPU memory: `nvidia-smi`

### Enhancement Not Applied

**Symptoms**: Output returned without enhancement, no quality improvement

**Solutions**:
1. Check terminal/logs for error messages (`app.log`)
2. Verify Real-ESRGAN pip package is installed (`pip install realesrgan`)
3. Try `RealESRGAN_x2plus` model to reduce memory usage
4. Ensure CUDA toolkit is properly installed
5. Check TensorRT cache isn't corrupted: delete `cache/tensorrt/` folder

### Video Has No Audio

**Symptoms**: Output video is silent

**Solutions**:
- Verify input video has audio track (check with media player)
- Check FFmpeg is installed and in PATH
- Review `app.log` for audio encoding errors
- Ensure FFmpeg supports input codec (MP4/H.264 recommended)

### Face Not Detected

**Symptoms**: "No faces detected in source/target" error

**Solutions**:

1. Ensure face is clearly visible and front-facing
2. Improve lighting and image quality
3. Use higher resolution input (minimum 640px recommended)
4. Verify buffalo_l models installed in `models/buffalo_l/`
5. Adjust `confidence_threshold` in `config.yaml` (lower = more permissive)

### Module Import Errors

**Symptoms**: `ModuleNotFoundError` or import errors

**Solutions**:
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify Python environment
conda activate faceoff
python --version  # Should show 3.10.x or 3.11.x
```

### PyTorch Compatibility Issues

**Symptoms**: Typing errors during app startup, `Union` or `Tuple` import failures

**Root Cause**: PyTorch 2.4.0/2.4.1 has typing system incompatibilities with Python 3.11

**Solutions**:

**Option 1 - Use Compatible PyTorch (Recommended):**
```powershell
conda activate faceoff
pip uninstall torch torchvision torchaudio -y

# Install PyTorch 2.3.1 (stable with Python 3.11)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Option 2 - Use Latest PyTorch:**
```powershell
# Install PyTorch 2.5.0+ (fixes typing issues)
pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu121
```

**Option 3 - Use Python 3.10:**
```powershell
# Create environment with Python 3.10 (most compatible)
conda create -n faceoff python=3.10 -y
conda activate faceoff
pip install -r requirements.txt
```

**Verification:**
```powershell
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

### Package Deprecation Warnings

**Symptoms**: `pkg_resources is deprecated` warnings during startup

**Root Cause**: Older versions of imageio-ffmpeg use deprecated pkg_resources API

**Solution:**
```powershell
conda activate faceoff
pip install --upgrade imageio-ffmpeg  # Updates to 0.6.0+ (fixes warning)
```

**Verification:**
```powershell
python -c "import imageio_ffmpeg; print('✅ No deprecation warnings')"
```

### Albumentations Update Issues

**Symptoms**: Online version checking failures, network timeouts during import

**Root Cause**: Older albumentations versions (1.4.x) have network dependency issues

**Solution:**
```powershell
conda activate faceoff
pip install --upgrade albumentations  # Updates to 2.0.8+ (removes online checks)
```

**Features**: Latest albumentations 2.0.8+ provides better performance and removes problematic online update checking.

### TensorRT Build Failures

**Symptoms**: TensorRT engine build errors or warnings

**Solutions**:
- Delete TensorRT cache: remove `cache/tensorrt/` directory
- Disable TensorRT in `config.yaml`: set `tensorrt_enabled: false`
- Update CUDA drivers to latest version
- Check compatibility: CUDA >= 10.1, TensorRT 8.x recommended

## Project Structure

```text
faceoff/
├── main.py                      # Application entry point
├── config.yaml                  # Main configuration file (YAML)
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment spec
├── inswapper_128.onnx          # Face swapper model (download separately)
│
├── core/                        # Core functionality
│   ├── gpu_manager.py          # Multi-GPU device management
│   ├── face_processor.py       # Face detection and alignment
│   └── media_processor.py      # Media file handling with model init
│
├── processing/                  # Media processing pipeline
│   ├── orchestrator.py         # Processing coordination
│   ├── async_pipeline.py       # 3-stage async pipeline
│   ├── model_optimizer.py      # TensorRT optimization
│   ├── enhancement.py          # Real-ESRGAN enhancement
│   ├── image_processing.py     # Image face swapping
│   ├── video_processing.py     # Video face swapping
│   ├── gif_processing.py       # GIF face swapping
│   ├── face_restoration.py     # GFPGAN face restoration
│   └── resolution_adaptive.py  # Adaptive resolution handling
│
├── ui/                          # User interface
│   ├── app.py                  # Gradio web interface
│   ├── components/             # UI component modules
│   └── helpers/                # UI helper utilities
│
├── utils/                       # Utilities
│   ├── config_manager.py       # Config loader with validation
│   ├── config_schema.py        # Configuration validation schema
│   ├── constants.py            # Model options and constants
│   ├── validation.py           # Input validation
│   ├── logging_setup.py        # Logging configuration with rotation
│   ├── json_formatter.py       # JSON log formatter for structured logging
│   ├── memory_manager.py       # GPU memory management with caching
│   ├── error_handler.py        # User-friendly error handling
│   ├── compression.py          # Media compression utilities
│   ├── temp_manager.py         # Temporary file management
│   ├── cleanup_manager.py      # Old file cleanup manager
│   ├── progress.py             # Progress tracking
│   ├── profiling.py            # Performance profiling utilities
│   ├── model_cache.py          # TensorRT cache management
│   └── preset_manager.py       # Settings preset management
│
├── tests/                       # Test suite
│   ├── conftest.py             # Shared pytest fixtures
│   └── unit/                   # Unit tests (135 tests)
│
├── scripts/                     # Utility scripts
│   ├── demo_config.py          # Config system demo
│   ├── demo_progress.py        # Progress tracking demo
│   └── test_onnx_gpu.py        # ONNX GPU compatibility test
│
├── presets/                     # Processing presets
│   ├── Balanced.json           # Balanced quality/speed preset
│   ├── High Quality.json       # Maximum quality preset
│   ├── Fast Preview.json       # Speed-optimized preset
│   └── Anime Style.json        # Anime/illustration preset
│
├── models/                      # Model storage
│   └── buffalo_l/              # InsightFace face analysis models
│
├── external/                    # External dependencies (optional)
│   └── gifsicle/               # GIF compression tool (optional)
│       └── gifsicle.exe        # Download from lcdf.org
│
# Runtime directories (auto-created, excluded from Git):
├── cache/                       # TensorRT model cache
├── inputs/                      # Uploaded source files
├── outputs/                     # Generated results
├── temp/                        # Temporary processing files
└── .gradio/                     # Gradio UI cache
```

## Version History

### v2.6.0 (Current) - Code Quality & Testing Release

**New Features:**

- ✅ **Comprehensive Test Suite**: 135 pytest unit tests covering core functionality
- ✅ **Exception Hierarchy**: Standardized error handling with specialized exception types
- ✅ **Multi-GPU Model Pool**: Thread-safe per-GPU ONNX sessions for video/GIF processing
- ✅ **Configuration Validation**: Schema-based config validation with auto-correction
- ✅ **Performance Profiling**: `@profile` decorator and `Profiler` context manager
- ✅ **Cleanup Manager**: Automatic cleanup of old temporary and output files
- ✅ **Frame Prefetching**: Background thread prefetches frames for faster processing
- ✅ **Memory Stats Caching**: Reduced overhead from repeated CUDA memory queries

**Improvements:**

- 🚀 **UI Refactor**: `ui/app.py` reduced from 966 to 656 lines with extracted handlers
- 💾 **Bounded Queues**: Async pipeline queues bounded to prevent memory explosion
- 🔧 **Thread Safety**: FaceMappingManager now thread-safe with proper locking
- 📝 **JSON Logging**: Optional structured JSON log output for log aggregation
- ⚙️ **Magic Numbers**: Moved to config (iou_threshold, mb_per_batch, ui_padding)

See [BLUEPRINT.md](BLUEPRINT.md) for implementation details.

### v2.5.0 - Performance & Reliability Release

**New Features:**

- ✅ **TensorRT Model Caching**: Persistent engine cache with automatic optimization
- ✅ **Async Processing Pipeline**: 3-stage overlapped processing (detection → swap → enhancement)
- ✅ **Intelligent Memory Management**: Auto cache clearing, OOM recovery, dynamic batch sizing
- ✅ **Logging Rotation**: Size-based log rotation with configurable retention (5 files @ 10MB)
- ✅ **YAML Configuration**: Migrated from JSON to YAML with comprehensive validation
- ✅ **Config Management**: Centralized `config_manager.py` with property-based access
- ✅ **Enhanced Documentation**: Added `IMPROVEMENTS.md` and `CONFIG_README.md`

**Improvements:**

- 🚀 **Performance**: TensorRT acceleration, model preloading, batch optimization
- 💾 **Memory**: Automatic CUDA cache management prevents OOM crashes
- 🔧 **Reliability**: Graceful degradation, error recovery, comprehensive logging
- 📝 **Maintainability**: Unified config system, modular architecture

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed technical documentation.

### v2.0.0 - Unified Interface

- ✅ Unified Gradio interface (Image/GIF/Video/Gallery tabs)
- ✅ 6 Real-ESRGAN model options with specialized purposes
- ✅ Denoise control for compatible models
- ✅ Modular architecture with processing/ui separation
- ✅ Audio preservation in enhanced videos
- ✅ Multi-GPU support

### v1.0.0 - Initial Release

- Basic face swapping for images, GIFs, and videos
- Single enhancement model option
- Individual app files per media type (deprecated)

## Credits & Acknowledgments

Built with excellent open-source projects:

- **[InsightFace](https://github.com/deepinsight/insightface)** - Face analysis and detection models
- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** - Image enhancement and upscaling
- **[gifsicle](https://www.lcdf.org/gifsicle/)** - Efficient GIF optimization and compression
- **[PyTorch](https://github.com/pytorch/pytorch)** - Deep learning framework
- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** - Cross-platform inference
- **[TensorRT](https://developer.nvidia.com/tensorrt)** - NVIDIA GPU acceleration
- **[Gradio](https://github.com/gradio-app/gradio)** - Web interface framework
- **[FFmpeg](https://github.com/FFmpeg/FFmpeg)** - Video/audio processing
- **[ImageMagick](https://github.com/ImageMagick/ImageMagick)** - Image manipulation and conversion
- **[CodeFormer](https://github.com/sczhou/CodeFormer)** - Face restoration reference
- **[OpenAI GPT](https://github.com/openai)** - Development assistance

## License

This project is provided for educational and research purposes. Please respect all applicable licenses for the models and dependencies used.

**Model Licenses:**

- InsightFace models: Check [InsightFace license](https://github.com/deepinsight/insightface/blob/master/LICENSE)
- Real-ESRGAN models: [BSD 3-Clause License](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
- inswapper_128.onnx: Check original model license

**Use Responsibly:**

- Only use on content you have permission to modify
- Do not create misleading or harmful content
- Respect privacy and consent
- Follow all applicable laws and regulations

## Support

- **Issues**: [GitHub Issues](https://github.com/thebiglaskowski/faceoff/issues)
- **Documentation**: See [CONFIG_README.md](CONFIG_README.md) and [IMPROVEMENTS.md](IMPROVEMENTS.md)
- **Blog Post**: [https://thebiglaskowski.com/posts/face-swapping-with-ai](https://thebiglaskowski.com/posts/face-swapping-with-ai/)

---

**Star ⭐ this repository if you find it useful!**
