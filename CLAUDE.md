# CLAUDE.md - FaceOff Development Guidelines

> Guidelines for Claude Code when working on this project.

---

## Project Overview

FaceOff is a local, GPU-accelerated face swapping application. See `PROJECT_INTENT.md` for full specification.

**Key characteristics:**
- Local-only (no cloud/SaaS)
- Gradio web UI
- Multi-GPU support via isolated ONNX sessions
- Video/GIF/Image processing with optional enhancement

---

## Build & Test Commands

```bash
# Sync env (conda is obsolete — use uv only)
uv sync

# Run application
uv run python main.py

# Run all tests
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_streaming_pipeline.py -v

# Run lint (isort check, black check)
uv run isort --check --diff .
uv run black --check .
```

---

## Code Architecture

```
ui/                     # Gradio interface (MAY call processing/, utils/)
processing/             # Orchestration, pipelines (MAY call core/, utils/)
core/                   # Face detection, model pool (MAY call utils/ only)
utils/                  # Config, logging, validation (no upward deps)
tests/                  # Unit and integration tests
```

**Import rules:**
- `utils/` → No imports from `ui/`, `processing/`, `core/`
- `core/` → No imports from `ui/`, `processing/`
- `processing/` → May import from `core/`, `utils/`
- `ui/` → May import from `processing/`, `utils/` (NOT `core/` directly)

---

## Key Files

| File | Purpose | Change With Caution |
|------|---------|---------------------|
| `config.yaml` | All configurable settings | Yes - affects all users |
| `main.py` | Entry point, signal handlers | Yes - startup/shutdown |
| `core/model_pool.py` | GPU session isolation | Yes - thread safety critical |
| `processing/streaming_media.py` | Chunked video/GIF pipeline | Yes - concurrency |
| `processing/in_memory_enhancement.py` | In-memory enhancement | Yes - multi-GPU CUDA |
| `utils/config_manager.py` | Singleton config access | Yes - used everywhere |

---

## Coding Conventions

### Error Handling

```python
# Use the error handler for user-facing errors
from utils.error_handler import ErrorHandler

try:
    result = risky_operation()
except Exception as e:
    return ErrorHandler.handle_error(e, context="processing image")
```

### Logging

```python
import logging
logger = logging.getLogger("FaceOff")

# Use appropriate levels
logger.debug("Detailed info for debugging")
logger.info("Normal operation info")
logger.warning("Potential issue")
logger.error("Error occurred: %s", error, exc_info=True)
```

### Configuration Access

```python
from utils.config_manager import config

# Property access (preferred for common settings)
batch_size = config.batch_size

# Hierarchical get (for nested/optional settings)
chunk_size = config.streaming_chunk_size
```

### Thread Safety

All shared state MUST be protected:

```python
import threading

class SharedResource:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
```

### Model Caching

Use `LRUModelCache` for bounded caching:

```python
from utils.lru_cache import LRUModelCache

def cleanup_fn(model):
    del model
    torch.cuda.empty_cache()

cache = LRUModelCache("MyModel", cleanup_fn=cleanup_fn)

# Use get/put, not dict-style access
cached = cache.get(key)
if cached is None:
    model = create_model()
    cache.put(key, model)
    cached = model
```

---

## Testing Guidelines

### Unit Tests

- Location: `tests/unit/`
- Name pattern: `test_<module>.py`
- Use fixtures from `tests/conftest.py`
- Mock GPU operations with `mock_gpu` fixture

```python
def test_something(mock_gpu, temp_config):
    """Test description."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result.status == "success"
```

### Integration Tests

- Location: `tests/integration/`
- Mark with `@pytest.mark.integration`
- May require actual GPU (`@pytest.mark.gpu`)

---

## Common Tasks

### Adding a New Config Option

1. Add to `config.yaml` with sensible default
2. Add property to `utils/config_manager.py` if frequently used
3. Add to `utils/config_schema.py` validation (if exists)
4. Document in `config.yaml` comments

### Adding a New Enhancement Model

1. Add model info to appropriate module (`processing/enhancement.py` or similar)
2. Add cleanup function for LRU cache eviction
3. Add to model cache with `LRUModelCache`
4. Add UI option if user-selectable

### Fixing OOM Issues

1. Check `utils/memory_manager.py` thresholds
2. Reduce `gpu.batch_size` in config
3. Reduce `gpu.onnx_mem_limit_mb` if BFC arena errors
4. Add `torch.cuda.empty_cache()` after heavy operations

---

## What NOT to Do

- **Don't add cloud/SaaS features** - Local only
- **Don't add user authentication** - No accounts needed
- **Don't add REST API** - Gradio is the interface
- **Don't add database storage** - Files only
- **Don't add model training** - Inference only
- **Don't use unbounded caches** - Use `LRUModelCache`
- **Don't skip thread locks** - Concurrency is critical
- **Don't hardcode magic numbers** - Use `config.yaml`

---

## Before Committing

1. Run `isort --check --diff .` and `black --check .` — fix any drift
2. Run `pytest tests/ -v` — all tests must pass
3. Check logs: no new warnings, no OOM
4. If touching processing code, test with real media files

---

## Reference Documentation

- `PROJECT_INTENT.md` - Authoritative project specification
- `config.yaml` - All configurable settings with comments
- `BLUEPRINT.md` - Historical implementation plan (reference only)
