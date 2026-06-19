# FaceOff Project Intent

> **Status**: Authoritative operating specification
> **Last Updated**: 2026-06-19
> **Supersedes**: BLUEPRINT.md (now historical reference only)

---

## What FaceOff IS

FaceOff is a **local, GPU-accelerated face swapping application** that processes:
- **Images** (PNG, JPG, WEBP, BMP)
- **Animated GIFs** with frame preservation
- **Videos** (MP4, AVI, MOV, WEBP) with audio preservation

It provides a **Gradio web UI** for interactive use on a single machine or local network.

### Core Capabilities

| Capability | Implementation | Evidence |
|------------|----------------|----------|
| Face detection | InsightFace buffalo_l models | `core/media_processor.py` |
| Face swapping | inswapper_128 ONNX model | `core/model_pool.py` |
| Enhancement | Real-ESRGAN, SwinIR (optional) | `processing/enhancement.py`, `processing/swinir_enhancement.py` |
| Face restoration | GFPGAN, CodeFormer (optional) | `processing/face_restoration.py`, `processing/codeformer_restoration.py` |
| Multi-GPU | Per-GPU ONNX session isolation | `core/model_pool.py` |
| Streaming video/GIF | Chunked decode→swap→enhance→encode | `processing/streaming_media.py` |

---

## What FaceOff is NOT

- **NOT a SaaS/cloud service** - Runs locally only
- **NOT a distributed system** - Single machine (multi-GPU supported)
- **NOT a training framework** - Inference only, no model training
- **NOT a real-time streaming tool** - Batch processing only
- **NOT a deepfake detection tool** - Creation only
- **NOT an API service** - Gradio UI is the interface (no REST API)

---

## Performance Goals

| Goal | Target | Implementation |
|------|--------|----------------|
| GPU acceleration | CUDA required for production use | ONNX Runtime CUDA provider |
| TensorRT optimization | 2-3x faster inference | Optional, cached engines in `cache/tensorrt/` |
| Batch processing | Configurable batch size (1-16) | `config.yaml: gpu.batch_size` |
| Memory efficiency | Prevent OOM, auto-reduce batch | `utils/memory_manager.py` |
| Streaming pipeline | Bounded RAM; FFmpeg I/O; in-memory enhance | `processing/streaming_media.py`, `utils/video_io.py` |
| Multi-GPU | VRAM-aware frame split (swap); single-GPU PyTorch enhance | `core/model_pool.py`, `processing/gpu_scheduler.py` |

---

## Reliability Goals

| Goal | Implementation | Evidence |
|------|----------------|----------|
| OOM recovery | Detect OOM, reduce batch, retry | `utils/memory_manager.py` |
| Graceful shutdown | Signal handlers (SIGINT/SIGTERM) | `main.py` |
| Resource cleanup | Temp files, CUDA cache, model pool | `utils/cleanup_manager.py`, `utils/temp_manager.py` |
| Error messaging | User-friendly with suggestions | `utils/error_handler.py` |
| Input validation | File size, resolution, duration | `utils/validation.py` |
| Config validation | Schema-based with defaults | `utils/config_schema.py` |

---

## Architecture Boundaries

```
┌─────────────────────────────────────────────────────────┐
│ UI Layer (ui/)                                          │
│ - Gradio interface, event handlers, components          │
│ - MAY call: processing/, utils/                         │
│ - MUST NOT call: core/ directly (use orchestrator)      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Processing Layer (processing/)                          │
│ - Orchestration, media processing, enhancement          │
│ - MAY call: core/, utils/                               │
│ - Entry point: orchestrator.py                          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Core Layer (core/)                                      │
│ - Face detection, model management, GPU sessions        │
│ - MAY call: utils/ only                                 │
│ - MUST NOT call: ui/, processing/                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Utils Layer (utils/)                                    │
│ - Config, logging, validation, memory, errors           │
│ - MUST NOT call: ui/, processing/, core/                │
│ - Pure utilities with no upward dependencies            │
└─────────────────────────────────────────────────────────┘
```

---

## Key Modules Reference

| Module | Purpose | Lines |
|--------|---------|-------|
| `main.py` | Entry point, signal handling, Gradio launch | ~150 |
| `ui/app.py` | Gradio UI composition | ~665 |
| `processing/orchestrator.py` | Unified processing entry | ~246 |
| `processing/streaming_media.py` | Chunked video/GIF pipeline | ~390 |
| `processing/in_memory_enhancement.py` | Multi-GPU-safe in-RAM enhance | ~180 |
| `core/model_pool.py` | Per-GPU ONNX session management | ~326 |
| `core/media_processor.py` | InsightFace initialization | ~422 |
| `utils/config_manager.py` | Singleton config (74 properties) | ~448 |
| `utils/error_handler.py` | Exception hierarchy | ~488 |

---

## Non-Goals (Explicitly Out of Scope)

1. **Web deployment** - No containerization, Kubernetes, cloud scaling
2. **User authentication** - No accounts, sessions, permissions
3. **Database storage** - Files only, no DB
4. **Model training** - Use pre-trained models only
5. **Real-time webcam** - Batch processing only
6. **Mobile apps** - Desktop/server only
7. **API endpoints** - Gradio UI is the interface

---

## Configuration Philosophy

All tunables live in `config.yaml` with:
- Sensible defaults that work on 8GB VRAM
- Schema validation on load (`utils/config_schema.py`)
- Property-based access (`config.batch_size`)
- Hierarchical get (`config.get('gpu', 'batch_size', default=4)`)

No runtime config changes persist to disk (intentional safety).

---

## Quality Standards

| Area | Standard |
|------|----------|
| Test coverage | 80% minimum (excluding UI callbacks) |
| Error handling | All exceptions → `ErrorHandler` → user-friendly message |
| Logging | `logging.getLogger("FaceOff")` everywhere |
| Type hints | Required in core/, utils/; optional in UI callbacks |
| Thread safety | Locks required for all shared state |

---

## Historical Note

`BLUEPRINT.md` contains the original implementation plan (January 2026). It is now **historical reference only**. This document (`PROJECT_INTENT.md`) is the authoritative specification for project scope and goals.
