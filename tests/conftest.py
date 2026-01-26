"""
Shared pytest fixtures for FaceOff tests.

This module provides fixtures for:
- Mock GPU availability
- Temporary directories
- Config reset between tests
- Sample test images/data
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def reset_config():
    """Reset config singleton between tests to ensure isolation."""
    from utils.config_manager import Config

    # Clear the singleton instance
    Config._instance = None
    Config._config = {}

    yield

    # Reset again after test
    Config._instance = None
    Config._config = {}


@pytest.fixture
def temp_config(tmp_path, reset_config):
    """
    Create a temporary config.yaml for isolated testing.

    Returns:
        Path to temporary config file
    """
    config_content = """
limits:
  max_file_size_mb: 100
  max_video_duration_sec: 60
  max_image_pixels: 4194304
  max_gif_frames: 100

gpu:
  batch_size: 2
  max_batch_size: 8
  workers_per_gpu: 2
  tensorrt_enabled: false
  tensorrt_fp16: false
  tensorrt_workspace_mb: 1024

face_detection:
  inswapper_model_path: "test_inswapper.onnx"
  buffalo_model_path: "models/buffalo_l"
  face_analysis_name: "buffalo_l"
  face_analysis_det_size: [320, 320]
  confidence_threshold: 0.5
  adaptive_enabled: true
  detection_scale: 0.5
  min_resolution: 320

enhancement:
  default_model: "RealESRGAN_x4plus"
  defaults:
    tile_size: 128
    outscale: 2
    pre_pad: 0
    use_fp32: false
    denoise_strength: 0.5

face_restoration:
  enabled_by_default: false
  model_version: "1.3"
  default_weight: 0.5

async_pipeline:
  enabled: true
  min_frames_threshold: 5
  queue_size: 16

logging:
  log_file: "test.log"
  max_file_size_mb: 5
  backup_count: 2
  console_level: "WARNING"
  file_level: "DEBUG"

memory:
  auto_clear_cache: false
  clear_cache_threshold_mb: 512
  reduce_batch_on_oom: true
  min_batch_size: 1
  mb_per_batch_estimate: 250

file_formats:
  images: [".jpg", ".png"]
  videos: [".mp4"]
  gifs: [".gif"]

directories:
  temp_gif_frames: "temp/test_frames"
  temp_gif_enhanced: "temp/test_enhanced"
  output: "test_outputs"
  models: "test_models"
  cache: "test_cache"

ui:
  server_name: "127.0.0.1"
  server_port: 7861
  share: false
  theme: "default"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    # Patch config path
    from utils import config_manager
    original_path = config_manager.Config._config_path
    config_manager.Config._config_path = config_path

    yield config_path

    # Restore original path
    config_manager.Config._config_path = original_path


# =============================================================================
# GPU Mocking Fixtures
# =============================================================================

@pytest.fixture
def mock_cuda_available():
    """Mock torch.cuda.is_available() to return True."""
    with patch('torch.cuda.is_available', return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock torch.cuda.is_available() to return False."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def mock_gpu():
    """
    Comprehensive GPU mock for testing without actual CUDA.

    Mocks:
    - torch.cuda.is_available() -> True
    - torch.cuda.device_count() -> 1
    - torch.cuda.memory_allocated() -> 1GB
    - torch.cuda.memory_reserved() -> 2GB
    - torch.cuda.get_device_properties() -> Mock with 8GB total
    - torch.cuda.empty_cache() -> No-op
    - torch.cuda.synchronize() -> No-op
    """
    mock_device_props = MagicMock()
    mock_device_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
    mock_device_props.name = "Mock GPU"

    with patch.multiple(
        'torch.cuda',
        is_available=MagicMock(return_value=True),
        device_count=MagicMock(return_value=1),
        memory_allocated=MagicMock(return_value=1 * 1024 * 1024 * 1024),  # 1GB
        memory_reserved=MagicMock(return_value=2 * 1024 * 1024 * 1024),   # 2GB
        get_device_properties=MagicMock(return_value=mock_device_props),
        empty_cache=MagicMock(),
        synchronize=MagicMock(),
        OutOfMemoryError=RuntimeError,  # Map OOM to RuntimeError for testing
    ):
        yield


@pytest.fixture
def mock_multi_gpu():
    """Mock multiple GPUs for multi-GPU testing."""
    mock_device_props = MagicMock()
    mock_device_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
    mock_device_props.name = "Mock GPU"

    with patch.multiple(
        'torch.cuda',
        is_available=MagicMock(return_value=True),
        device_count=MagicMock(return_value=2),
        memory_allocated=MagicMock(return_value=1 * 1024 * 1024 * 1024),
        memory_reserved=MagicMock(return_value=2 * 1024 * 1024 * 1024),
        get_device_properties=MagicMock(return_value=mock_device_props),
        empty_cache=MagicMock(),
        synchronize=MagicMock(),
    ):
        yield


# =============================================================================
# Sample Image/Media Fixtures
# =============================================================================

@pytest.fixture
def sample_image(tmp_path):
    """
    Create a small test image with a simple face-like pattern.

    Returns:
        Path to test image file
    """
    # Create a 200x200 RGB image
    img_array = np.zeros((200, 200, 3), dtype=np.uint8)

    # Add a simple face-like circle pattern
    # Background: light skin tone
    img_array[:, :] = [180, 160, 140]

    # Draw a simple "face" - oval shape
    center_y, center_x = 100, 100
    for y in range(200):
        for x in range(200):
            # Oval face shape
            if ((x - center_x) / 60) ** 2 + ((y - center_y) / 80) ** 2 <= 1:
                img_array[y, x] = [220, 180, 160]
            # Eyes (dark circles)
            if ((x - 75) ** 2 + (y - 80) ** 2 <= 100) or \
               ((x - 125) ** 2 + (y - 80) ** 2 <= 100):
                img_array[y, x] = [50, 50, 50]
            # Nose (small triangle area)
            if 95 <= x <= 105 and 85 <= y <= 115:
                img_array[y, x] = [200, 160, 140]
            # Mouth (red line)
            if 85 <= x <= 115 and 130 <= y <= 135:
                img_array[y, x] = [60, 60, 100]

    img = Image.fromarray(img_array)
    img_path = tmp_path / "test_face.png"
    img.save(img_path)

    return img_path


@pytest.fixture
def sample_image_no_face(tmp_path):
    """
    Create a test image with no face content.

    Returns:
        Path to test image file
    """
    # Create a simple gradient image with no face
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img_array[i, :, 0] = i * 2  # Red gradient
        img_array[:, i, 2] = i * 2  # Blue gradient

    img = Image.fromarray(img_array)
    img_path = tmp_path / "test_no_face.png"
    img.save(img_path)

    return img_path


@pytest.fixture
def sample_gif(tmp_path):
    """
    Create a simple test GIF with a few frames.

    Returns:
        Path to test GIF file
    """
    frames = []
    for i in range(5):
        # Create frames with slight variations
        img_array = np.zeros((50, 50, 3), dtype=np.uint8)
        img_array[:, :, 0] = (i * 50) % 255  # Red changes per frame
        img_array[:, :, 1] = 100
        img_array[:, :, 2] = 100
        frames.append(Image.fromarray(img_array))

    gif_path = tmp_path / "test.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )

    return gif_path


@pytest.fixture
def large_image(tmp_path):
    """
    Create a large test image that may exceed limits.

    Returns:
        Path to large test image
    """
    # 5000x5000 = 25 million pixels (exceeds typical 4K limit)
    img = Image.new('RGB', (5000, 5000), color='red')
    img_path = tmp_path / "large_image.png"
    img.save(img_path)

    return img_path


# =============================================================================
# Mock Face Detection Fixtures
# =============================================================================

@pytest.fixture
def mock_face():
    """Create a mock face object for testing."""
    mock_face = MagicMock()
    mock_face.bbox = np.array([50, 50, 150, 150])  # x1, y1, x2, y2
    mock_face.det_score = 0.95
    mock_face.age = 30
    mock_face.gender = 1  # Male
    mock_face.embedding = np.random.rand(512).astype(np.float32)
    return mock_face


@pytest.fixture
def mock_faces():
    """Create multiple mock face objects for testing."""
    faces = []
    positions = [
        (50, 50, 150, 150),    # Left face
        (200, 50, 300, 150),   # Right face
        (125, 200, 225, 300),  # Bottom center face
    ]

    for i, (x1, y1, x2, y2) in enumerate(positions):
        face = MagicMock()
        face.bbox = np.array([x1, y1, x2, y2])
        face.det_score = 0.9 - (i * 0.1)  # Decreasing confidence
        face.age = 25 + (i * 5)
        face.gender = i % 2
        face.embedding = np.random.rand(512).astype(np.float32)
        faces.append(face)

    return faces


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def capture_logs(caplog):
    """
    Fixture to capture log output during tests.

    Usage:
        def test_something(capture_logs):
            # do something that logs
            assert "expected message" in capture_logs.text
    """
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture(autouse=True)
def isolate_tests(tmp_path, monkeypatch):
    """
    Automatically isolate tests from each other.

    - Sets working directory to temp path
    - Prevents actual file system pollution
    """
    # Store original cwd
    original_cwd = os.getcwd()

    # Don't change cwd, but set env vars for temp paths
    monkeypatch.setenv('FACEOFF_TEMP_DIR', str(tmp_path))

    yield

    # Cleanup happens automatically via tmp_path


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip GPU tests if CUDA is not available.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "gpu" in item.keywords and not cuda_available:
            item.add_marker(skip_gpu)
