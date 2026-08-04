"""GPU stack integration tests — require CUDA and model files."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _probe_in_subprocess(preload: bool) -> str:
    """Run the TensorRT probe in a clean interpreter.

    A subprocess is required: dlopen cannot be undone, so once any test in this
    process has preloaded the NVIDIA wheels the 'cold' path is unreachable.
    """
    import subprocess

    body = (
        "import sys; sys.path.insert(0, %r)\n" % str(ROOT)
        + (
            "from utils.onnx_providers import preload_nvidia_libraries, "
            "setup_nvidia_library_path\n"
            "setup_nvidia_library_path(); preload_nvidia_libraries()\n"
            if preload
            else ""
        )
        + "from utils.onnx_providers import is_tensorrt_runtime_available\n"
        "print(is_tensorrt_runtime_available())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
    )
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "ERROR"


@pytest.mark.integration
@pytest.mark.gpu
def test_tensorrt_detection_is_not_preload_order_dependent():
    """TensorRT availability must not depend on the caller preloading first.

    is_tensorrt_runtime_available() is lru_cached, so a single early call made
    before preload_nvidia_libraries() would otherwise pin TensorRT to
    'unavailable' for the life of the process — silently downgrading every
    session to CUDA.
    """
    det_model = ROOT / "models" / "buffalo_l" / "det_10g.onnx"
    if not det_model.is_file():
        pytest.skip("buffalo_l det_10g.onnx not present — probe defers to True")

    cold = _probe_in_subprocess(preload=False)
    warm = _probe_in_subprocess(preload=True)

    assert cold in ("True", "False"), f"probe did not report cleanly: {cold!r}"
    assert cold == warm, (
        f"TensorRT detection is order-dependent: without preload={cold}, "
        f"with preload={warm}. The probe must preload the NVIDIA wheels itself."
    )


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
