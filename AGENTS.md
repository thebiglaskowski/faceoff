# FaceOff — AGENTS.md

> Instructions for OpenCode sessions working in this repo. See `CLAUDE.md`
> for detailed coding conventions and `PROJECT_INTENT.md` for scope/spec.

---

## Environment

| Item | Detail |
|---|---|
| Python | 3.12 |
| Dependency manager | **`uv`** (conda is obsolete — see below) |
| Venv | `.venv/` (created by `uv venv --python 3.12`) |
| Lockfile | `uv.lock` |

```bash
# Sync env
uv sync

# Run
python main.py

# Tests
pytest tests/ -v

# Coverage
pytest tests/ --cov=. --cov-report=term-missing
```

**conda is dead** — `environment.yml` was deleted; `main.py` no longer uses conda paths.
Do not suggest `conda activate` or `environment.yml`. Use `uv sync` instead.

---

## Architecture (import boundaries — non-negotiable)

```
ui/       → MAY call → processing/ , utils/
processing/ → MAY call → core/ , utils/
core/     → MAY call → utils/
utils/    → MAY NOT call anything above
```

UI must never import `core/` directly; it goes through `processing/orchestrator.py`.

---

## Entry Points

| File | Role |
|---|---|
| `main.py` | Server entry, signal handlers, `LRUModelCache` shim for torchvision compatibility |
| `processing/orchestrator.py` | **`process_media(opts)`** — single entry for all media processing. Accepts a `ProcessOptions` dataclass (24 fields). |
| `core/model_pool.py` | Per-GPU ONNX session isolation — use `get_instance()` / `cleanup()` |
| `processing/streaming_media.py` | Chunked video/GIF pipeline (decode → swap → enhance → encode) |
| `processing/in_memory_enhancement.py` | In-memory enhancement; HAT/SwinIR single-GPU for CUDA stability |

---

## Environment Quirks

1. **torchvision compatibility shim** in `main.py` redirects `rgb_to_grayscale` from the removed `torchvision.transforms.functional_tensor` to `torchvision.transforms.functional`. **Do not remove this** — `basicsr` (pulled by GFPGAN/RealESRGAN) depends on the old path.

2. **FFmpeg is required** and must be on `$PATH`. Video/GIF decoding was rewritten from `moviepy` to direct FFmpeg subprocess calls (see `processing/video_processing.py`, `processing/gif_processing.py`). Moviepy is fully removed and `utils/video_io.pyi` provides type stubs for pyi compatibility.

3. **Multi-GPU face swap** uses VRAM-aware scheduling (`processing/gpu_scheduler.py`). **HAT/SwinIR enhancement** always runs on a single GPU (most free VRAM) to avoid WSL2 CUDA allocator races.

4. **Thread safety:** all shared mutable state must use `threading.Lock()`. The `model_pool.py` singleton is the main shared resource.

---

## Testing

- Unit tests: `pytest tests/unit/ -v`
- Integration tests: `pytest tests/ --integration` — may require GPU
- GPU tests: `pytest tests/ --gpu`
- Fixtures in `tests/conftest.py`. GPU operations are mocked.
- Test imports must be lazy (inside test methods) to avoid collection crashes.

---

## Key Constraints

- **Local only** — no cloud, no SaaS, no auth, no REST API, no DB, no training.
- **GPU required** for production (`CUDAExecutionProvider`). CPU fallback is a degraded path.
- **All tunables in `config.yaml`** — no hardcoded magic numbers.
- **No unbounded caches** — use `LRUModelCache` with explicit cleanup functions.

---

## Pending Work

- **Scene-aware face mapping** — keyframe timeline for multi-scene GIF/video (`.planning/designs/scene-aware-face-mapping.md`)
- **Wave 5+:** PyNvCodec zero-copy decode when package installed
- **ReSwapper** (deferred): Diffusion-based quality mode; needs dual-engine Fast/Quality toggle
- **Deferred UI:** Runtime config editor, model management UI (see `BLUEPRINT.md`)

## Completed (recent)

- Streaming pipeline replaced `async_pipeline.py`
- Batch face swap ONNX (`swap_face_batch`) — Wave 2
- HAT multi-GPU stability (serialized load, single-GPU inference)
- Face detection pool rebind after VRAM release
- Wave 3 phase 1: chunk GPU upload + swap IoBinding (`gpu.frame_retention_enabled`)
- Wave 3 phase 2: GPU paste-back + single D2H per chunk (`gpu.paste_on_gpu`, `core/face_paste_gpu.py`)
- Wave 3 phase 3: GPU detection (`gpu.detection_on_gpu`), NVDEC+pinned decode (`streaming.zero_copy_enabled`), GPU HAT chain (`gpu.enhancement_chain_enabled`)
- Wave 4: auto workload profiles (`gpu.auto_workload_tune`, `processing/workload_profile.py`), RealESRGAN GPU chain (`core/gpu_realesrgan.py`)
- FFmpeg stderr drain fix (`utils/video_io.py`) — prevents streaming encode/decode hangs

---

## Common Task Patterns

### Add a config option
1. `config.yaml` with default
2. Property in `utils/config_manager.py` if used frequently
3. No manual schema — `config.yaml` *is* the schema (validated at load)

### Add an enhancement model
1. Register in `processing/enhancement.py` or a new `*_enhancement.py`
2. Add `LRUModelCache` with cleanup
3. Add UI option in `ui/app.py` + handler

### Fix OOM
1. Check `utils/memory_manager.py` thresholds
2. Lower `gpu.batch_size` or `gpu.onnx_mem_limit_mb` in config
