# FaceOff Test Suite

This directory contains the pytest test suite for FaceOff.

## Directory Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Unit tests
│   ├── test_config_manager.py   # Configuration system tests
│   ├── test_error_handler.py    # Error handling tests
│   ├── test_face_processor.py   # Face detection/tracking tests
│   ├── test_memory_manager.py   # Memory management tests
│   └── test_validation.py       # Input validation tests
└── README.md                # This file
```

## Running Tests

From the project root:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_config_manager.py -v

# Run tests matching a pattern
pytest tests/ -k "test_oom" -v
```

## Test Coverage

The test suite covers:

- **Configuration Management**: Loading, defaults, property access, validation
- **Error Handling**: User-friendly error conversion, suggestions, error hierarchy
- **Face Processing**: Face detection, IoU calculation, face tracking, mappings
- **Memory Management**: VRAM monitoring, cache clearing, OOM recovery
- **Input Validation**: File size, resolution, duration, format validation

## Writing Tests

Use the fixtures defined in `conftest.py`:

```python
def test_example(mock_gpu, temp_config, sample_image):
    """Example test using fixtures."""
    # mock_gpu - Mocks CUDA availability
    # temp_config - Temporary config.yaml for isolated testing
    # sample_image - Creates a small test image
    pass
```

## Demo Scripts

Utility scripts for manual testing are located in `scripts/`:

- `scripts/demo_config.py` - Test configuration system
- `scripts/demo_progress.py` - Demo progress tracking
- `scripts/demo_terminal.py` - Test terminal output
- `scripts/test_onnx_gpu.py` - Test ONNX GPU compatibility
