# FaceOff

**Unified AI Face Swapper - Image | GIF | Video**

Swap faces from a source image to a destination medium (image, GIF, or video) with optional AI enhancement using Real-ESRGAN. All features available in a single, unified Gradio interface.

Read more at [https://thebiglaskowski.com/posts/face-swapping-with-ai](https://thebiglaskowski.com/posts/face-swapping-with-ai/)

## ✨ Features

- **Face Swapping**: Swap faces from source image to:
  - 🖼️ Images (PNG, JPG, WEBP)
  - 🎞️ Animated GIFs
  - 🎬 Videos (MP4, WEBP) with audio preservation
  
- **AI Enhancement**: Optional Real-ESRGAN upscaling with 6 model options:
  - 🎯 **RealESRGAN_x4plus**: General purpose - Best for photos (Default)
  - 🎨 **RealESRGAN_x4plus_anime_6B**: Optimized for anime/illustration content
  - 💎 **RealESRGAN_x4plus (Conservative)**: Less aggressive enhancement
  - 🔧 **realesr-general-x4v3**: General purpose with denoise control
  - 🎬 **realesr-animevideov3**: Specialized for anime videos
  - ⚡ **RealESRGAN_x2plus**: 2x upscale for faster processing

- **Advanced Enhancement Controls**:
  - Model selection per processing task
  - Denoise strength control (0-1) for compatible models
  - Automatic quality optimization based on selected model

- **Smart Processing**:
  - Automatic GPU memory management
  - Frame-by-frame enhancement for GIFs and videos
  - Audio preservation in enhanced videos
  - Automatic cleanup of temporary files

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

1. **FFmpeg** - [Download and install FFmpeg](https://ffmpeg.org/download.html)
2. **CUDA >= 10.1** - [Download and install CUDA](https://developer.nvidia.com/cuda-10.1-download-archive-base)
3. **Python 3.10** - Anaconda/Miniconda recommended

### Setup Instructions

```powershell
# Clone the repository
git clone https://github.com/thebiglaskowski/faceoff.git
cd faceoff

# Create conda environment
conda create -n faceoff python=3.10 -y
conda activate faceoff

# Install PyTorch with CUDA support
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt
```

4. **Download Models**:
   - Download [inswapper_128.onnx](https://huggingface.co/thebiglaskowski/inswapper_128.onnx/tree/main) and place in project root
   - Download [buffalo_l models](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) and extract to `models/buffalo_l/`
   - Real-ESRGAN weights will download automatically on first use

## Usage

### Unified Interface (Recommended)

```powershell
python faceoff_unified.py
```

Opens at <http://127.0.0.1:7860/>

The unified interface provides three tabs:

- **Image Tab**: Face swap for static images
- **GIF Tab**: Face swap for animated GIFs with frame preservation
- **Video Tab**: Face swap for videos with audio preservation
- **Batch Tab**: Process multiple files at once

#### Enhancement Options

1. **Enable Enhancement**: Check the "Enable Enhancement (Real-ESRGAN)" checkbox
2. **Select Model**:
   - **RealESRGAN_x4plus**: Best general-purpose model for photos
   - **RealESRGAN_x4plus_anime_6B**: Optimized for anime/illustration
   - **RealESRGAN_x4plus (Conservative)**: More subtle enhancement
   - **realesr-general-x4v3**: Includes denoise strength control (0-1 slider)
   - **realesr-animevideov3**: Best for anime videos
   - **RealESRGAN_x2plus**: Faster 2x upscaling
3. **Adjust Denoise** (only for realesr-general-x4v3): Control noise reduction strength from 0 (none) to 1 (maximum)

## Performance Tips

### GPU Memory Optimization

- **8GB VRAM**: Use any model except anime_6B (which requires more memory)
- **6GB VRAM**: Use RealESRGAN_x2plus or x4plus models
- **4GB VRAM or less**: Disable enhancement or upgrade GPU

### Processing Time Estimates

For a 10-second 30fps video (300 frames):

- **No Enhancement**: ~30 seconds
- **With Enhancement (any model)**: ~10-20 minutes depending on model and GPU

## Configuration

### GPU Memory Optimization

- **8GB VRAM**: Use "Balanced" preset (Tile 256) or "Fast" preset (Tile 512)
- **6GB VRAM**: Use "Fast" preset only
- **4GB VRAM or less**: Disable enhancement or upgrade GPU

### Processing Time Estimates

For a 10-second 30fps video (300 frames):

- **No Enhancement**: ~30 seconds
- **With Enhancement (any model)**: ~10-20 minutes depending on model and GPU

*Times vary based on GPU, video resolution, and scene complexity*

## Troubleshooting

### CUDA Out of Memory Error

**Solution**: Use RealESRGAN_x2plus model or lower resolution input

```powershell
# Check GPU memory
nvidia-smi
```

### Enhancement Not Applied

**Issue**: Video/GIF returned without enhancement

**Solutions**:

1. Check terminal for error messages
2. Verify Real-ESRGAN weights exist in `external/Real-ESRGAN/`
3. Try RealESRGAN_x2plus model to reduce memory usage
4. Ensure CUDA is properly installed

### Video Has No Audio

**Issue**: Output video is silent

**Solution**: Ensure input video has audio track - enhancement automatically preserves audio

### Module Not Found Errors

**Solution**: Ensure all dependencies are installed

```powershell
pip install -r requirements.txt
```

### Face Not Detected

**Issue**: "No faces detected" error

**Solutions**:
1. Ensure face is clearly visible and front-facing
2. Try better lighting in source image
3. Use higher resolution input
4. Verify buffalo_l models are properly installed

## Configuration

Edit `config.json` to customize settings:

```json
{
  "inswapper_model_path": "inswapper_128.onnx",
  "buffalo_model_path": "models/buffalo_l",
  "default_output_dir": "outputs",
  "cuda_provider": "CUDAExecutionProvider",
  "face_analysis_det_size": [640, 640],
  "real_esrgan_scaling_factor": 4
}
```

## Project Structure

```
faceoff/
├── faceoff_unified.py      # Main unified interface (RECOMMENDED)
├── media_processing.py     # Core processing logic
├── media_utils.py          # MediaProcessor class
├── config.py               # Configuration loader
├── logging_utils.py        # Logging setup
├── enhancement_utils.py    # Enhancement processor
├── config.json             # Configuration file
├── inputs/                 # Temporary input files
├── outputs/                # Processed outputs
├── models/                 # Model files
│   └── buffalo_l/         # InsightFace models
├── external/              # External dependencies
│   └── Real-ESRGAN/       # Enhancement engine
└── tests/                 # Test files
```

## Changelog

### v2.0.0 (Current)

- ✅ **Unified Interface**: All four modes (Image/GIF/Video/Batch) in single app
- ✅ **Advanced Model Selection**: 6 Real-ESRGAN models with specialized purposes
- ✅ **Denoise Control**: Fine-tune noise reduction for compatible models
- ✅ **Modular Architecture**: Clean, maintainable codebase with processing/ui separation
- ✅ **Audio Preservation**: Videos maintain original audio after enhancement
- ✅ **Unique Filenames**: Timestamp-based naming prevents output overwrites
- ✅ **Multi-GPU Support**: Leverage multiple GPUs for faster processing

### v1.0.0 (Legacy)

- Basic face swapping for images, GIFs, and videos
- Single model enhancement option
- Legacy individual app files (deprecated)

## Roadmap

Future enhancements planned:

- [ ] Progress bars showing enhancement progress for videos/GIFs
- [ ] Multiple face selection (choose which face to swap if multiple detected)
- [x] Batch processing mode (process multiple files at once) - **COMPLETED**
- [ ] Face confidence threshold controls
- [ ] Video preview before enhancement
- [ ] Output format options (codec, resolution, compression)
- [ ] Side-by-side comparison view
- [ ] Processing history and favorites
- [x] Multiple Real-ESRGAN model options - **COMPLETED**
- [x] Denoise strength control - **COMPLETED**

## Special Thanks To

- [FFMpeg](https://github.com/FFmpeg/FFmpeg)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [CodeFormer](https://github.com/sczhou/CodeFormer)
- [Open-AI GPT](https://github.com/openai)
- [PyTorch](https://github.com/pytorch/pytorch)
- [Torchvision](https://github.com/pytorch/pytorch)
