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
  - 📦 **Batch Processing** for multiple files simultaneously

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

3. **CUDA >= 10.1** - [Download and install CUDA](https://developer.nvidia.com/cuda-10.1-download-archive-base)

4. **TensorRT** (Optional, for GPU acceleration):
   - [Download TensorRT](https://developer.nvidia.com/tensorrt) 
   - Follow NVIDIA's installation guide for your CUDA version
   - Provides 2-3x faster face detection performance

5. **Python 3.10** - Anaconda/Miniconda recommended

### Setup Instructions

⚠️ **Important**: Open **Developer PowerShell for VS 2022** before proceeding with installation.

1. **Clone Repository**:
```powershell
git clone https://github.com/thebiglaskowski/faceoff.git
cd faceoff
```

2. **Create Environment**:
```powershell
conda create -n faceoff python=3.10 -y
conda activate faceoff
```

3. **Install PyTorch with CUDA**:
```powershell
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

4. **Install Dependencies**:
```powershell
pip install -r requirements.txt
```

5. **Download Models**:
   - [inswapper_128.onnx](https://huggingface.co/thebiglaskowski/inswapper_128.onnx/tree/main) → place in project root
   - [buffalo_l models](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) → extract to `models/buffalo_l/`
   - Real-ESRGAN weights download automatically on first use

6. **Optional: Install gifsicle for better GIF compression**:
   - [Download gifsicle for Windows](https://www.lcdf.org/gifsicle/)
   - Extract `gifsicle.exe` to `external/gifsicle/` folder
   - Provides ~60% better GIF compression than default PIL optimization

7. **Configure** (Optional):
   - Copy `config.example.yaml` to `config.yaml` (auto-created with defaults if missing)
   - Edit settings as needed - see [CONFIG_README.md](CONFIG_README.md) for details

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
- **Batch Tab**: Process multiple files simultaneously

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
2. Verify Real-ESRGAN weights downloaded to `external/Real-ESRGAN/weights/`
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
python --version  # Should show 3.10.x
```

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
│   ├── constants.py            # Model options and constants
│   ├── validation.py           # Input validation
│   ├── logging_setup.py        # Logging configuration with rotation
│   ├── memory_manager.py       # GPU memory management
│   ├── compression.py          # Media compression utilities
│   ├── temp_manager.py         # Temporary file management
│   ├── progress.py             # Progress tracking
│   ├── model_cache.py          # TensorRT cache management
│   └── preset_manager.py       # Settings preset management
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
├── external/                    # External dependencies
│   ├── Real-ESRGAN/            # Enhancement engine
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

### v2.5.0 (Current) - Performance & Reliability Release

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

- ✅ Unified Gradio interface (Image/GIF/Video/Batch tabs)
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
