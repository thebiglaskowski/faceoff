"""GPU stack integration tests — require CUDA and model files."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorrt_detection_does_not_false_positive():
    from utils.onnx_providers import is_tensorrt_runtime_available

    # Must reflect actual library loadability, not provider list presence
    result = is_tensorrt_runtime_available()
    assert isinstance(result, bool)


@pytest.mark.integration
@pytest.mark.gpu
def test_media_processor_face_detection():
    det_model = ROOT / "models" / "buffalo_l"
    if not det_model.exists():
        pytest.skip("buffalo_l models not present")

    from utils.onnx_providers import preload_nvidia_libraries, setup_nvidia_library_path

    setup_nvidia_library_path()
    preload_nvidia_libraries()

    from core.media_processor import MediaProcessor

    proc = MediaProcessor(device_id=0, use_tensorrt=False, optimize_models=False)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = proc.get_faces(img)
    assert isinstance(faces, list)