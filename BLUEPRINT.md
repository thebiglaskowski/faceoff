# FaceOff Improvement Blueprint

> Generated from comprehensive code review on 2026-01-25
> Overall Grade: **8.5/10** (Strong)

---

## Table of Contents

1. [Current Status](#current-status)
2. [Priority 1: Critical](#priority-1-critical)
3. [Priority 2: High](#priority-2-high)
4. [Priority 3: Medium](#priority-3-medium)
5. [Code Quality Issues](#code-quality-issues)
6. [Performance Optimizations](#performance-optimizations)
7. [Specific Code Locations](#specific-code-locations)
8. [Architecture Notes](#architecture-notes)

---

## Current Status

| Area | Score | Status |
|------|-------|--------|
| Architecture | 9/10 | Excellent |
| GPU Optimization | 9/10 | Excellent |
| Configuration | 9/10 | Excellent |
| Error Handling | 8.5/10 | Good |
| Documentation | 8/10 | Good |
| Production Readiness | 8/10 | Good |
| Code Quality | 7.5/10 | Needs Work |
| Performance | 7.5/10 | Needs Work |
| UI Code | 6/10 | Needs Refactor |
| Multi-GPU Support | 6/10 | Partially Broken |
| Testing | 5/10 | Poor |

### What's Working Well
- Modular, well-organized codebase
- TensorRT compilation and caching (2-3x faster)
- Intelligent memory management with OOM recovery
- 3-stage async pipeline (detection → swap → enhancement)
- Comprehensive YAML configuration (60+ properties)
- User-friendly error messages with suggestions
- Logging with rotation

---

## Priority 1: Critical

### [ ] 1.1 Add Comprehensive Unit Tests

**Problem:** Only 5 basic test files exist, no unit tests for core logic.

**Current test files:**
- `tests/onnx_gpu_test.py` - Just 9 lines, basic CUDA check
- `tests/test_config.py` - Basic property checks
- `tests/test_improvements.py` - Feature validation
- `tests/test_progress.py` - Progress tracking
- `tests/test_terminal.py` - Terminal output

**Action:** Create proper pytest structure:

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_face_processor.py     # Face detection/tracking tests
│   ├── test_memory_manager.py     # Memory management tests
│   ├── test_config_manager.py     # Configuration tests
│   ├── test_error_handler.py      # Error handling tests
│   └── test_validation.py         # Input validation tests
├── integration/
│   ├── test_image_processing.py   # End-to-end image tests
│   ├── test_video_processing.py   # End-to-end video tests
│   ├── test_gif_processing.py     # End-to-end GIF tests
│   └── test_async_pipeline.py     # Pipeline integration tests
└── performance/
    ├── test_memory_usage.py       # Memory leak detection
    └── test_processing_speed.py   # Performance benchmarks
```

**Tests to write:**
- [ ] Face detection accuracy with known images
- [ ] Face tracking IoU calculations
- [ ] OOM recovery behavior
- [ ] Configuration loading/defaults
- [ ] Error message generation
- [ ] Input validation (file size, duration, resolution)
- [ ] Async pipeline frame ordering

---

### [ ] 1.2 Refactor ui/app.py (966 lines)

**Problem:** Massive file with 75-line event chains that are impossible to debug or test.

**Current structure:**
```python
# ui/app.py - 966 lines
# - 50+ event handlers
# - 75-line preset loading chain with .then().then().then()...
# - Duplicated logic across tabs
# - Deep function nesting
```

**Action:** Split into logical modules:

```
ui/
├── app.py                    # Main entry (~200 lines)
├── event_handlers/
│   ├── __init__.py
│   ├── preset_handlers.py    # Preset load/save logic
│   ├── processing_handlers.py # Process button handlers
│   └── gallery_handlers.py   # Gallery management
├── factories/
│   ├── __init__.py
│   └── tab_factory.py        # Factory for creating tabs (reduce duplication)
└── styles.py                 # CSS styles (extract from app.py)
```

**Refactor the preset loading chain:**

Before (bad):
```python
load_preset_btn.click(
    fn=lambda preset: toggle_enhancement_controls(...),
    inputs=[preset_dropdown],
    outputs=[img_components["model_row"], ...]
).then(
    fn=lambda preset: toggle_restoration_controls(...),
    ...
).then(
    fn=load_preset_settings,
    ...
).then(  # ... 70 more lines ...
)
```

After (good):
```python
# ui/event_handlers/preset_handlers.py
class PresetHandlers:
    def __init__(self, img_comp, gif_comp, vid_comp):
        self.components = [img_comp, gif_comp, vid_comp]

    def load_preset(self, preset_name: str) -> dict:
        """Load preset and return all updated values"""
        settings = preset_manager.load_preset(preset_name)
        return self._build_output_dict(settings)

    def _build_output_dict(self, settings: dict) -> dict:
        """Build Gradio output dictionary for all components"""
        outputs = {}
        for comp in self.components:
            outputs.update(self._apply_to_component(comp, settings))
        return outputs

# ui/app.py
preset_handlers = PresetHandlers(img_comp, gif_comp, vid_comp)
load_preset_btn.click(
    fn=preset_handlers.load_preset,
    inputs=[preset_dropdown],
    outputs=[...all outputs...]
)
```

---

### [ ] 1.3 Fix Multi-GPU Face Swapping

**Problem:** Multi-GPU is disabled for videos/GIFs due to ONNX Runtime threading issues.

**Location:** `processing/orchestrator.py`
```python
# TEMPORARY: Force single-GPU for videos/GIFs due to inswapper model threading issues
if len(device_ids) > 1 and media_type in ["video", "gif"]:
    device_ids = [device_ids[0]]
```

**Root cause:** ONNX Runtime is not thread-safe for simultaneous model loads.

**Potential solutions:**
- [ ] Investigate frame buffering instead of model sharing
- [ ] Create separate ONNX sessions per GPU
- [ ] Consider PyTorch alternative to InSwapper model
- [ ] Document limitations clearly if unfixable

---

## Priority 2: High

### [ ] 2.1 Bound Async Pipeline Queues

**Problem:** Unbounded queues can consume all memory with large frame counts.

**Location:** `processing/async_pipeline.py:45-47`
```python
self.detection_queue = queue.Queue(maxsize=0)  # Unbounded!
self.swapping_queue = queue.Queue(maxsize=0)   # Unbounded!
self.output_queue = queue.Queue(maxsize=0)     # Unbounded!
```

**Fix:**
```python
# Add backpressure to prevent memory explosion
self.detection_queue = queue.Queue(maxsize=10)
self.swapping_queue = queue.Queue(maxsize=10)
self.output_queue = queue.Queue(maxsize=10)
```

**Also add:**
- [ ] Timeout on thread joins (currently can hang indefinitely)
- [ ] Better error propagation between threads
- [ ] Logging of queue sizes for diagnostics

---

### [ ] 2.2 Add Runtime Configuration UI

**Problem:** Changing settings requires editing config.yaml and restarting.

**Action:** Add "Advanced Settings" tab in Gradio UI:

```python
with gr.Tab("Settings"):
    with gr.Group():
        gr.Markdown("### GPU Settings")
        batch_size = gr.Slider(1, 16, value=config.batch_size, label="Batch Size")
        tensorrt_enabled = gr.Checkbox(value=config.tensorrt_enabled, label="TensorRT")

    with gr.Group():
        gr.Markdown("### Memory Settings")
        cache_threshold = gr.Slider(512, 4096, value=config.clear_cache_threshold_mb)
        auto_clear = gr.Checkbox(value=config.auto_clear_cache)

    with gr.Group():
        gr.Markdown("### Face Detection")
        confidence = gr.Slider(0.1, 0.9, value=config.face_confidence_threshold)
        adaptive_detection = gr.Checkbox(value=config.adaptive_detection)

    save_config_btn = gr.Button("Save Configuration")
    reset_config_btn = gr.Button("Reset to Defaults")
```

---

### [ ] 2.3 Move Magic Numbers to Configuration

**Locations and values:**

| File | Line | Value | Suggested Config Key |
|------|------|-------|---------------------|
| `core/face_processor.py` | 97 | `iou_threshold=0.3` | `face_detection.iou_threshold` |
| `utils/memory_manager.py` | 108 | `mb_per_batch=500` | `memory.mb_per_batch_estimate` |
| `ui/helpers/face_detection.py` | ~50 | `padding=20` | `ui.face_padding_pixels` |
| `ui/helpers/face_detection.py` | ~80 | `font_size=max(24,...)` | `ui.min_font_size` |

**Add to config.yaml:**
```yaml
face_detection:
  iou_threshold: 0.3  # IoU threshold for face tracking

memory:
  mb_per_batch_estimate: 500  # Estimated VRAM per batch item

ui:
  face_padding_pixels: 20
  min_font_size: 24
```

---

### [ ] 2.4 Add Structured Logging Option

**Problem:** Current logging is string-based, hard to parse in log aggregation systems.

**Location:** `utils/logging_setup.py`

**Add JSON formatter option:**
```python
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# In config.yaml:
# logging:
#   format: "json"  # or "text"
```

---

### [ ] 2.5 Improve Thread Safety in Async Pipeline

**Problem:** Error in one thread doesn't immediately stop others.

**Location:** `processing/async_pipeline.py`

**Issues to fix:**
- [ ] Add timeout to thread.join() calls
- [ ] Propagate errors between threads immediately
- [ ] Add thread pool instead of manual threads
- [ ] Log when threads are killed due to timeout

```python
# Current (problematic):
for thread in self.threads:
    thread.join(timeout=2.0)  # No error if timeout reached

# Better:
for thread in self.threads:
    thread.join(timeout=5.0)
    if thread.is_alive():
        logger.error(f"Thread {thread.name} did not terminate, forcing stop")
        self.stop_event.set()
```

---

## Priority 3: Medium

### [ ] 3.1 Create Tab Factory to Reduce Duplication

**Problem:** Image/GIF/Video tabs are 70-80% identical code.

**Action:** Create factory function:

```python
# ui/factories/tab_factory.py
def create_processing_tab(
    tab_name: str,
    media_type: str,
    accepted_formats: list,
    extra_options: dict = None
) -> dict:
    """Factory function to create processing tabs with common structure"""

    with gr.Tab(tab_name) as tab:
        with gr.Row():
            source_image = gr.Image(label="Source Face", type="filepath")
            target_input = gr.File(label=f"Target {tab_name}", file_types=accepted_formats)

        # Common enhancement controls
        enhance_toggle = gr.Checkbox(label="Enable Enhancement")
        model_dropdown = gr.Dropdown(choices=ENHANCEMENT_MODELS)

        # Common face restoration controls
        restore_toggle = gr.Checkbox(label="Enable Face Restoration")

        # Media-specific options
        if extra_options:
            # Add GIF-specific or video-specific controls
            pass

        process_btn = gr.Button(f"Process {tab_name}")
        output = gr.File(label="Output")

    return {
        "tab": tab,
        "source": source_image,
        "target": target_input,
        "enhance": enhance_toggle,
        "model": model_dropdown,
        "restore": restore_toggle,
        "process_btn": process_btn,
        "output": output
    }

# Usage:
img_components = create_processing_tab("Image", "image", [".jpg", ".png", ".webp"])
gif_components = create_processing_tab("GIF", "gif", [".gif"], extra_options={"optimize": True})
vid_components = create_processing_tab("Video", "video", [".mp4", ".avi", ".mov"])
```

---

### [ ] 3.2 Add Model Management UI

**Features to add:**
- [ ] Show installed models and their sizes
- [ ] Download models from UI (with progress)
- [ ] View TensorRT cache size
- [ ] Clear cache button
- [ ] Verify model integrity

```python
with gr.Tab("Models"):
    gr.Markdown("### Installed Models")
    model_table = gr.Dataframe(
        headers=["Model", "Size", "Status"],
        value=get_model_status()
    )

    gr.Markdown("### TensorRT Cache")
    cache_size = gr.Textbox(label="Cache Size", value=get_cache_size())
    clear_cache_btn = gr.Button("Clear TensorRT Cache")

    gr.Markdown("### Download Models")
    download_dropdown = gr.Dropdown(choices=AVAILABLE_MODELS)
    download_btn = gr.Button("Download Selected Model")
    download_progress = gr.Progress()
```

---

### [ ] 3.3 Add Configuration Validation

**Problem:** Invalid config values aren't caught until runtime.

**Action:** Add validation on config load:

```python
# utils/config_manager.py
def _validate_config(self) -> list[str]:
    """Validate configuration values, return list of warnings"""
    warnings = []

    # Validate ranges
    if not (1 <= self.batch_size <= 32):
        warnings.append(f"batch_size {self.batch_size} outside valid range [1, 32]")

    if not (0.1 <= self.face_confidence_threshold <= 0.9):
        warnings.append(f"face_confidence_threshold {self.face_confidence_threshold} outside valid range")

    # Validate paths exist
    if not Path(self.output_directory).exists():
        warnings.append(f"output_directory '{self.output_directory}' does not exist")

    # Validate model files
    for model in self.enhancement_models:
        if not Path(model['path']).exists():
            warnings.append(f"Enhancement model not found: {model['path']}")

    return warnings
```

---

### [ ] 3.4 Add Performance Profiling

**Action:** Add profiling decorator for slow functions:

```python
# utils/profiling.py
import functools
import time
import logging

logger = logging.getLogger(__name__)

def profile(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

# Usage:
@profile
def process_video(input_path, ...):
    ...
```

**Also add:**
- [ ] Memory profiling with `tracemalloc`
- [ ] GPU memory tracking per operation
- [ ] Bottleneck identification report

---

### [ ] 3.5 Add Cleanup for Old Uploads

**Problem:** Uploaded files go to `inputs/` without cleanup schedule.

**Action:** Add scheduled cleanup:

```python
# utils/cleanup_manager.py
class CleanupManager:
    def __init__(self, max_age_hours: int = 24):
        self.max_age = timedelta(hours=max_age_hours)

    def cleanup_old_files(self, directory: Path) -> int:
        """Remove files older than max_age, return count deleted"""
        deleted = 0
        cutoff = datetime.now() - self.max_age

        for file in directory.iterdir():
            if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                file.unlink()
                deleted += 1

        return deleted

# Run on startup or periodically
cleanup = CleanupManager(max_age_hours=config.cleanup_max_age_hours)
cleanup.cleanup_old_files(Path("inputs"))
cleanup.cleanup_old_files(Path("temp"))
```

---

## Code Quality Issues

### [ ] Incomplete Type Hints

**Files needing type hints:**
- `ui/app.py` - Many functions lack parameter types
- `processing/orchestrator.py` - Return types missing
- `ui/helpers/*.py` - Mixed typing coverage

**Example fix:**
```python
# Before:
def process_input(source_image, target_image_path=None, ...):

# After:
def process_input(
    source_image: str,
    target_image_path: Optional[str] = None,
    enhance: bool = False,
    model_name: str = "realesr-general-x4v3",
    ...
) -> Tuple[Optional[str], str]:
```

---

### [ ] Inconsistent Error Handling Patterns

**Current patterns (inconsistent):**
```python
# Pattern 1: Re-raise FriendlyError
except FriendlyError:
    raise

# Pattern 2: Convert to gr.Error
except gr.Error:
    raise

# Pattern 3: Wrap and re-raise
except Exception as e:
    friendly_error = ErrorHandler.handle_error(e, context)
    logger.error(..., exc_info=True)
    raise gr.Error(friendly_error.format_message())
```

**Standardize to:**
```python
# Always use this pattern in UI layer:
try:
    result = process_media(...)
except Exception as e:
    friendly = ErrorHandler.handle_error(e, get_context())
    logger.error(f"Processing failed: {e}", exc_info=True)
    raise gr.Error(friendly.format_message()) from e
```

---

## Performance Optimizations

### [ ] Add Frame Prefetching

**Problem:** Frames loaded one at a time, GPU may idle waiting for I/O.

**Location:** `processing/async_pipeline.py`

**Solution:** Prefetch next frame while processing current:

```python
from concurrent.futures import ThreadPoolExecutor

class AsyncPipeline:
    def __init__(self, ...):
        self.frame_prefetch_executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_buffer = {}

    def _prefetch_frame(self, frame_idx: int, frames: list):
        """Load frame into memory asynchronously"""
        if frame_idx < len(frames):
            self.prefetch_buffer[frame_idx] = frames[frame_idx].copy()

    def process_frames(self, frames):
        # Submit prefetch for first few frames
        for i in range(min(3, len(frames))):
            self.frame_prefetch_executor.submit(self._prefetch_frame, i, frames)

        # Process with prefetching
        for idx, frame in enumerate(frames):
            # Prefetch upcoming frame
            if idx + 3 < len(frames):
                self.frame_prefetch_executor.submit(self._prefetch_frame, idx + 3, frames)

            # Use prefetched frame if available
            if idx in self.prefetch_buffer:
                frame = self.prefetch_buffer.pop(idx)

            # Process frame...
```

---

### [ ] Cache Memory Stats

**Problem:** `get_memory_stats()` called frequently, adds overhead.

**Location:** `utils/memory_manager.py`

**Solution:** Cache stats for short window:

```python
import time

class MemoryManager:
    def __init__(self, ...):
        self._stats_cache = None
        self._stats_cache_time = 0
        self._stats_cache_ttl = 0.1  # 100ms cache

    def get_memory_stats(self) -> dict:
        now = time.time()
        if self._stats_cache and (now - self._stats_cache_time) < self._stats_cache_ttl:
            return self._stats_cache

        # Compute fresh stats
        stats = self._compute_memory_stats()
        self._stats_cache = stats
        self._stats_cache_time = now
        return stats
```

---

### [ ] Reduce Enhancement Subprocess Overhead

**Problem:** Real-ESRGAN uses subprocess, has creation overhead.

**Location:** `processing/enhancement.py`

**Options:**
- [ ] Keep subprocess warm between calls
- [ ] Use Python API directly instead of CLI
- [ ] Batch multiple frames in single subprocess call

---

## Specific Code Locations

Quick reference for files that need changes:

| Task | File | Lines |
|------|------|-------|
| Refactor UI | `ui/app.py` | All (966 lines) |
| Bound queues | `processing/async_pipeline.py` | 45-47 |
| Multi-GPU fix | `processing/orchestrator.py` | ~150 |
| Magic numbers | `core/face_processor.py` | 97 |
| Magic numbers | `utils/memory_manager.py` | 108 |
| Type hints | `ui/app.py` | Throughout |
| Error handling | `ui/app.py` | ~400-500 |
| Thread safety | `processing/async_pipeline.py` | 267-273 |
| Logging format | `utils/logging_setup.py` | Throughout |
| Config validation | `utils/config_manager.py` | Add new method |

---

## Architecture Notes

### Current Data Flow
```
User Input (Gradio UI)
       ↓
   Orchestrator (processing/orchestrator.py)
       ↓
   ┌───────────────────────────────────────┐
   │         Media-Specific Processor       │
   │  (image_processing / video_processing  │
   │   / gif_processing)                    │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │         Async Pipeline                 │
   │  Detection → Swapping → Output         │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │         Enhancement (optional)         │
   │  Real-ESRGAN + GFPGAN                  │
   └───────────────────────────────────────┘
       ↓
   Output File
```

### Key Design Decisions (ADRs)

1. **Why YAML for config?** Human-readable, version-controllable, supports complex nesting
2. **Why Gradio?** Rapid UI development, built-in file handling, easy deployment
3. **Why async pipeline?** Overlaps I/O and GPU work, improves throughput 30-50%
4. **Why TensorRT caching?** Eliminates 60s first-run compilation delay
5. **Why subprocess for Real-ESRGAN?** Isolation, memory management, existing CLI tool

---

## Completion Tracking

### Phase 1: Foundation (Critical)
- [ ] 1.1 Unit tests for core modules
- [ ] 1.2 Refactor ui/app.py
- [ ] 1.3 Fix multi-GPU support

### Phase 2: Stability (High)
- [ ] 2.1 Bound async queues
- [ ] 2.2 Runtime config UI
- [ ] 2.3 Move magic numbers to config
- [ ] 2.4 Structured logging
- [ ] 2.5 Thread safety improvements

### Phase 3: Polish (Medium)
- [ ] 3.1 Tab factory
- [ ] 3.2 Model management UI
- [ ] 3.3 Config validation
- [ ] 3.4 Performance profiling
- [ ] 3.5 Cleanup manager

### Phase 4: Code Quality
- [ ] Complete type hints
- [ ] Standardize error handling
- [ ] Performance optimizations

---

*This blueprint will evolve as work progresses. Check off items as completed.*
