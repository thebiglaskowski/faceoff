# FaceOff

**Unified AI Face Swapper - Image | GIF | Video**

Swap faces from a source image to a destination medium (image, GIF, or video) with optional AI enhancement using Real-ESRGAN. All features available in a single, unified Gradio interface.

Read more at [https://thebiglaskowski.com/posts/face-swapping-with-ai](https://thebiglaskowski.com/posts/face-swapping-with-ai/)

## ✨ Features

- **Face Swapping**: Swap faces from source image to:
  - 🖼️ Images (PNG, JPG, WEBP)
  - 🎞️ Animated GIFs
  - 🎬 Videos (MP4, WEBP) with audio preservation
  
- **AI Enhancement**: Optional Real-ESRGAN upscaling with 3 quality presets:
  - ⚡ **Fast**: 2x upscale, faster processing (Tile 512)
  - ⚖️ **Balanced**: 4x upscale, good quality/speed balance (Tile 256) - Recommended
  - 💎 **Quality**: 4x upscale, maximum quality (Tile 128) - Slower but best results

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
3. **Python 3.8+** - Anaconda/Miniconda recommended

### Setup Instructions

```powershell
# Clone the repository
git clone https://github.com/thebiglaskowski/faceoff.git
cd faceoff

# Create conda environment
conda create -n faceoff python=3.8 -y
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

#### Enhancement Options

1. **Enable Enhancement**: Check the "Enable Enhancement (Real-ESRGAN)" checkbox
2. **Select Quality Preset**:
   - **Fast (2x, Tile 512)**: Fastest processing, 2x upscaling - Use for quick previews or when processing many videos
   - **Balanced (4x, Tile 256)**: Recommended - 4x upscaling with good speed/quality balance
   - **Quality (4x, Tile 128)**: Best quality, slowest - Use for final renders

### Legacy Individual Apps

For backward compatibility, individual apps are still available:

```powershell
# Image processing only
python faceoff.py  # Port 5000

# GIF processing only
python faceoff_gif.py  # Port 5001

# Video processing only
python faceoff_video.py  # Port 5002
```

## Performance Tips

### GPU Memory Optimization

- **8GB VRAM**: Use "Balanced" preset (Tile 256) or "Fast" preset (Tile 512)
- **6GB VRAM**: Use "Fast" preset only
- **4GB VRAM or less**: Disable enhancement or upgrade GPU

### Processing Time Estimates

For a 10-second 30fps video (300 frames):
- **No Enhancement**: ~30 seconds
- **Fast (2x)**: ~5-8 minutes
- **Balanced (4x)**: ~10-15 minutes
- **Quality (4x)**: ~20-30 minutes

*Times vary based on GPU, video resolution, and scene complexity*

## Troubleshooting

### CUDA Out of Memory Error

**Solution**: Use a faster quality preset or lower resolution input

```powershell
# Check GPU memory
nvidia-smi
```

### Enhancement Not Applied

**Issue**: Video/GIF returned without enhancement

**Solutions**:
1. Check terminal for error messages
2. Verify Real-ESRGAN weights exist in `external/Real-ESRGAN/`
3. Try "Fast" preset to reduce memory usage
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
- ✅ **Unified Interface**: All three modes (Image/GIF/Video) in single app
- ✅ **Enhanced Processing**: Real-ESRGAN enhancement for all media types
- ✅ **Quality Presets**: Fast/Balanced/Quality options for user control
- ✅ **Code Optimization**: Reduced duplicate code, improved performance
- ✅ **Audio Preservation**: Videos maintain original audio after enhancement
- ✅ **Smart Memory Management**: Automatic GPU cache clearing and tiling
- ✅ **Frame Handling**: Intelligent frame count matching for enhanced videos

### v1.0.0 (Legacy)
- Basic face swapping for images, GIFs, and videos
- Separate apps for each media type
- CodeFormer enhancement (deprecated)

## Roadmap

Future enhancements planned:

- [ ] Progress bars showing enhancement progress for videos/GIFs
- [ ] Multiple face selection (choose which face to swap if multiple detected)
- [ ] Batch processing mode (process multiple files at once)
- [ ] Face confidence threshold controls
- [ ] Video preview before enhancement
- [ ] Output format options (codec, resolution, compression)
- [ ] Side-by-side comparison view
- [ ] Processing history and favorites

## Special Thanks To

- [FFMpeg](https://github.com/FFmpeg/FFmpeg)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [CodeFormer](https://github.com/sczhou/CodeFormer)
- [Open-AI GPT](https://github.com/openai)
- [PyTorch](https://github.com/pytorch/pytorch)
- [Torchvision](https://github.com/pytorch/pytorch)
