<div align="center">

# FaceOff

**Local, GPU-accelerated face swapping — Image · GIF · Video**

High-performance face swapping with multi-GPU ONNX inference, chunked streaming, and optional super-resolution. Everything runs on your machine — no cloud, no accounts, no uploads.

![Python](https://img.shields.io/badge/python-3.10--3.12-3776ab?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-22863a?logo=opensourceinitiative&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2%20%7C%20Windows-lightgrey?logo=linux&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-multi--GPU-76b900?logo=nvidia&logoColor=white)
![UI](https://img.shields.io/badge/UI-Gradio-ff7c00?logo=gradio&logoColor=white)
![Tests](https://img.shields.io/badge/tests-316%2B-blue?logo=pytest&logoColor=white)

[Quick Start](#quick-start) · [Features](#features) · [Usage](#usage) · [Performance](#performance) · [Troubleshooting](#troubleshooting)

</div>

---

## Showcase

<div align="center">

| Source | FaceOff result |
|:------:|:--------------:|
| ![Arnold source](assets/arnold.jpg) | ![Arnold swapped](assets/arnold-faceoff.png) |
| ![Elon source](assets/elon.webp) | ![Elon swapped](assets/elon-faceoff.png) |
| ![Nacho source](assets/nacho-libre.gif) | ![Nacho swapped](assets/nacho-libre-faceoff.gif) |
| ![Rock source](assets/smell-rock.gif) | ![Rock swapped](assets/smell-rock-faceoff.gif) |

</div>

<details>
<summary>▶️ Demo videos — FaceOff vs DeepFaceLab, Basic vs Enhanced</summary>

**FaceOff Img2MP4 vs DeepFaceLab** — FaceOff on the left, DeepFaceLab on the right

<https://github.com/thebiglaskowski/faceoff/assets/5170343/9f30932c-bfed-4dbe-9131-eaee92a854de>

**Basic vs Enhanced (Real-ESRGAN)** — Basic on the left, Enhanced on the right

<https://github.com/thebiglaskowski/faceoff/assets/5170343/cd2eec67-2233-4813-ae16-5d3554c61884>

[![Watch the video](https://img.youtube.com/vi/H7KS8ZoulGw/hqdefault.jpg)](https://www.youtube.com/embed/H7KS8ZoulGw)

</details>

Read the write-up at [thebiglaskowski.com/blog/face-swapping-with-ai](https://thebiglaskowski.com/blog/face-swapping-with-ai/).

---

## Features

<table>
<tr>
<td>🎭 <b>Universal face swap</b><br>Images (PNG/JPG/WEBP/BMP), animated GIFs, and video (MP4/WEBP/AVI/MOV) — audio preserved</td>
<td>👥 <b>Multi-face mapping</b><br>Per-face source assignment right in the UI when several faces are detected</td>
</tr>
<tr>
<td>✨ <b>Face restoration</b><br>Optional post-swap GFPGAN or CodeFormer for sharper, cleaner faces</td>
<td>🔍 <b>Super-resolution</b><br>Real-ESRGAN (6 models), HAT / HAT-GAN, and SwinIR / Swin2SR with tiled inference</td>
</tr>
<tr>
<td>🖥️ <b>Multi-GPU by default</b><br>VRAM-aware scheduling spreads the swap across every CUDA device you have</td>
<td>🌊 <b>Streaming pipeline</b><br>Chunked decode → swap → enhance → encode keeps RAM bounded on long clips</td>
</tr>
<tr>
<td>⚡ <b>NVDEC + TensorRT</b><br>Native PyNvVideoCodec decode and optional ORT TensorRT EP with persistent engine cache</td>
<td>🧠 <b>Memory-safe</b><br>Auto cache clearing, OOM-triggered batch reduction, and bounded LRU model caches</td>
</tr>
<tr>
<td>🔒 <b>Fully local</b><br>No cloud, no accounts, no telemetry — media never leaves your machine</td>
<td>🎛️ <b>Everything configurable</b><br>Every tunable lives in <code>config.yaml</code>, created with sane defaults on first run</td>
</tr>
</table>

📖 Deep dives: [CONFIG_README.md](CONFIG_README.md) (all settings) · [IMPROVEMENTS.md](IMPROVEMENTS.md) (implementation notes)

---

## Contents

- [FaceOff](#faceoff)
  - [Showcase](#showcase)
  - [Features](#features)
  - [Contents](#contents)
  - [Requirements](#requirements)
  - [Quick Start](#quick-start)
    - [Platform notes](#platform-notes)
  - [Usage](#usage)
  - [Performance](#performance)
  - [Configuration](#configuration)
  - [Troubleshooting](#troubleshooting)
  - [Project structure](#project-structure)
  - [Version history](#version-history)
  - [Credits](#credits)
  - [License](#license)

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| **Python 3.12** | Recommended. 3.10–3.11 supported. 3.13 not compatible. |
| **NVIDIA GPU + drivers** | CUDA execution provider required for production use |
| **FFmpeg** | On `PATH` — video/GIF decode and encode |
| **uv** | Dependency manager ([install](https://github.com/astral-sh/uv)) |
| **Models** | `inswapper_128.onnx` + `models/buffalo_l/` (see [Quick Start](#quick-start)) |

> **VRAM:** 8 GB handles most workloads; 12 GB+ recommended for 4× enhancement at 1080p. HAT/SwinIR run on a single GPU (most free VRAM) to avoid multi-GPU CUDA races on WSL2.

---

## Quick Start

```bash
git clone https://github.com/thebiglaskowski/faceoff.git
cd faceoff

# Create env and install locked deps (PyTorch CUDA wheels included via pyproject.toml)
uv sync

# Smoke-test the GPU stack (optional but recommended)
uv run python scripts/verify_gpu_stack.py

# Download models (manual):
#   inswapper_128.onnx → project root
#     https://huggingface.co/thebiglaskowski/inswapper_128.onnx
#   buffalo_l → models/buffalo_l/
#     https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

uv run python main.py
```

The UI opens at **<http://127.0.0.1:7860/>**. `config.yaml` is created with defaults on first run — edit and restart to apply changes.

### Platform notes

<details>
<summary><b>Linux / WSL2</b> (recommended)</summary>

- Install system packages: `ffmpeg`, `libmagic1`, `gifsicle`, `build-essential`
- WSL2 inherits Windows NVIDIA drivers — run `nvidia-smi` inside WSL to confirm
- `torch.compile()` + Triton may give a 30–50% speedup on Linux for SwinIR
- TensorRT: bundled in the venv via `uv sync`; `main.py` preloads the `.so` libs for ORT
- If `libnvinfer.so.10` is missing, set `gpu.tensorrt_enabled: false` — the CUDA EP still works

</details>

<details>
<summary><b>Windows</b></summary>

- Use **Developer PowerShell for VS 2022** (C++ build tools) if building `insightface` from source on Python 3.12
- `python-magic-bin` is **not** needed on Linux; on Windows the lockfile may pull platform-specific packages via `uv sync`
- Optional: `external/gifsicle/gifsicle.exe` for better GIF compression

</details>

<details>
<summary><b>Python 3.12 on any platform</b></summary>

`basicsr` and `insightface` may need extra steps. The repo includes a **torchvision compatibility shim** in `main.py` for GFPGAN/RealESRGAN. If `uv sync` fails on these packages, drop to Python 3.11:

```bash
uv venv --python 3.11 .venv
uv sync
```

</details>

---

## Usage

The Gradio UI has tabs for **Image**, **GIF**, **Video**, **Gallery**, and **Terminal** (live log tail).

1. Upload a **source** face image and **target** media
2. Map faces if multiple are detected
3. Optionally enable **enhancement** and pick a model / framework
4. Process — outputs land in `outputs/`

> **Speed tip:** Disable enhancement for previews. For video, `streaming.video_face_enhance: false` (default) skips per-frame GFPGAN inside ESRGAN.

---

## Performance

Times vary by resolution, GPU count, TensorRT availability, and scene complexity. Rough guides for **1080p, 30 fps**:

| Clip | Frames | Swap only (1–2 GPUs) | + Real-ESRGAN 4× | + HAT 4× |
|------|--------|----------------------|------------------|----------|
| 10 s | 300 | ~1–2 min | ~10–20 min | ~15–25 min |
| 15 s | 450 | ~1.5–3 min | ~15–30 min | ~20–35 min |

<details>
<summary><b>"Is 27+ minutes normal for a 15-second clip?"</b></summary>

- **With enhancement enabled (HAT or 4× ESRGAN):** Yes. ~450 frames at ~3–4 s/frame ≈ 22–30 minutes is typical on a single enhancement GPU.
- **Swap only, no enhancement:** No. Expect roughly **2–3 minutes** on dual GPUs. If you see 27 minutes without enhancement, check logs for accidental enhance-on, very high resolution, or CPU fallback.

Check the Terminal tab or `app.log` for lines like `Enhancement enabled` and the model name. Wave 3 frame retention reduces H2D overhead for swapping but does not speed up enhancement much.

</details>

<details>
<summary><b>Tuning for speed</b></summary>

```yaml
gpu:
  batch_size: 8                    # Raise if VRAM allows; auto-reduces on OOM
  frame_retention_enabled: true    # Wave 3 — keep on unless debugging

enhancement:
  multi_gpu_enabled: false         # HAT/SwinIR always single-GPU anyway

streaming:
  chunk_size: 32
  video_face_enhance: false        # Faster video enhance path
```

Use `RealESRGAN_x2plus` or `default_outscale: 2` for faster, lighter upscaling.

</details>

---

## Configuration

All tunables live in `config.yaml`. Highlights:

```yaml
gpu:
  batch_size: 8
  tensorrt_enabled: true
  frame_retention_enabled: true   # Wave 3: chunk GPU upload + swap IoBinding

streaming:
  enabled: true
  chunk_size: 32
  hwaccel_decode: true
  nvenc_encode: true

enhancement:
  multi_gpu_enabled: false
```

Full reference: [CONFIG_README.md](CONFIG_README.md)

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| CUDA OOM | Lower `gpu.batch_size`, use an x2 model, shorter clips |
| TensorRT / `libnvinfer` errors | `uv sync`, run `scripts/verify_gpu_stack.py`, or disable `gpu.tensorrt_enabled` |
| No faces detected | Better lighting, higher-res input, check `models/buffalo_l/`, lower `face_detection.confidence_threshold` |
| Silent video output | Confirm input has audio; FFmpeg on `PATH` |
| Enhancement skipped | Check `app.log`; verify `realesrgan` installed; try an x2 model |
| WSL2 Gradio reload issues | Raise `fs.inotify.max_user_watches` (see [WSL docs](https://learn.microsoft.com/en-us/windows/wsl/compare-versions)) |

```bash
uv sync                    # Reinstall deps
uv run pytest tests/ -v    # Full test suite
```

---

## Project structure

<details>
<summary>Directory layout</summary>

```text
faceoff/
├── main.py                      # Entry point, torchvision shim, TensorRT preload
├── config.yaml                  # User-editable settings
├── pyproject.toml / uv.lock     # Dependencies — use `uv sync`
│
├── core/
│   ├── model_pool.py            # Per-GPU ONNX sessions, swap IoBinding
│   ├── gpu_frame.py             # ChunkFrameBuffer (Wave 3)
│   ├── face_processor.py        # Detection, tracking, mapping
│   └── media_processor.py       # Swap facade
│
├── processing/
│   ├── orchestrator.py          # process_media() — single entry
│   ├── streaming_media.py       # Chunked video/GIF pipeline
│   ├── frame_batch.py           # Multi-GPU batching + GPU buffers
│   ├── gpu_scheduler.py         # VRAM-aware assignment
│   ├── in_memory_enhancement.py # HAT / ESRGAN / SwinIR in RAM
│   └── image_processing.py      # Single-image path
│
├── ui/                          # Gradio app, tabs, handlers
├── utils/                       # config, video_io, memory, logging
├── tests/                       # 316+ pytest tests
└── scripts/verify_gpu_stack.py  # GPU smoke test
```

</details>

---

## Version history

**v2.12.0 (current) — Wave 5 NVDEC**

- **PyNvVideoCodec decode** (`streaming.nvcodec_decode`): `SimpleDecoder` NVDEC path for video via `pynvvideocodec`
- Auto workload profiles enable NVCodec for video jobs; FFmpeg CUDA hwaccel remains the fallback
- Pinned decode buffers work with both backends via `open_streaming_reader()`

<details>
<summary>Earlier releases</summary>

**v2.11.0 — Wave 4 Auto Workloads**

- **Automatic workload profiles** (`gpu.auto_workload_tune`): swap-only, HAT/RealESRGAN/SwinIR chains, restore-faces, face-mapping paths pick chunk size and GPU flags per job
- **RealESRGAN GPU enhancement chain** — frames stay on GPU through swap → enhance → single D2H
- Logs show the profile name at job start (e.g. `stream_swap_only`, `stream_hat_chain`)

**v2.10.0 — Wave 3 Complete**

- **Wave 3 phase 3**: GPU detection (`gpu.detection_on_gpu`), NVDEC + pinned decode (`streaming.zero_copy_enabled`), GPU HAT enhancement chain (`gpu.enhancement_chain_enabled`)

**v2.9.0 — Streaming & Wave 3 (phases 1–2)**

- **Streaming pipeline** replaces legacy `async_pipeline.py` for video/GIF
- **Wave 3 phase 1**: `ChunkFrameBuffer` + swapper ORT IoBinding (`gpu.frame_retention_enabled`)
- **HAT stability**: Serialized load, single-GPU inference for HAT/SwinIR
- **Face detection rebind** after VRAM release in the model pool
- **uv** as the primary package manager; `pyproject.toml` + `uv.lock`
- **316+ tests**; FFmpeg-native decode/encode (moviepy removed)

**v2.8.0 — HAT Enhancement**

- HAT and HAT-GAN super-resolution alongside Real-ESRGAN and SwinIR
- LRU caches, tiled inference, integration tests

Full history: [IMPROVEMENTS.md](IMPROVEMENTS.md)

</details>

---

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) — face analysis & swapping
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — upscaling
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) / [TensorRT](https://developer.nvidia.com/tensorrt) — inference
- [Gradio](https://github.com/gradio-app/gradio) — UI
- [FFmpeg](https://ffmpeg.org/) — media I/O

---

## License

FaceOff's own source code is released under the **MIT License** — see [LICENSE](LICENSE).

> [!IMPORTANT]
> FaceOff depends on pretrained models that carry their own, more restrictive terms. In particular the InsightFace `inswapper` model and **CodeFormer** are licensed for **non-commercial research use only**, which makes the project *as a whole* effectively non-commercial. See [NOTICE](NOTICE) for the full breakdown of third-party model licenses and the responsible-use disclaimer. **Obtain consent before modifying anyone's likeness.**

<div align="center">

Built with [InsightFace](https://github.com/deepinsight/insightface) · [ONNX Runtime](https://github.com/microsoft/onnxruntime) · [Gradio](https://github.com/gradio-app/gradio) · [uv](https://docs.astral.sh/uv/) &nbsp;|&nbsp; MIT License

[GitHub Issues](https://github.com/thebiglaskowski/faceoff/issues) · [Blog post](https://thebiglaskowski.com/blog/face-swapping-with-ai/)

</div>
