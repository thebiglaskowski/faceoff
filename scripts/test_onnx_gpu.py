#!/usr/bin/env python3
"""Quick ONNX CUDA provider smoke test."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.onnx_providers import preload_nvidia_libraries, setup_nvidia_library_path

setup_nvidia_library_path()
preload_nvidia_libraries()

import onnxruntime as ort  # noqa: E402

model = ROOT / "models" / "buffalo_l" / "1k3d68.onnx"
print("Available execution providers:", ort.get_available_providers())
try:
    sess = ort.InferenceSession(
        str(model),
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ],
    )
    print("Active providers:", sess.get_providers())
    if "CUDAExecutionProvider" not in sess.get_providers():
        print("WARNING: Session fell back away from CUDA")
    else:
        print("CUDAExecutionProvider successfully initialized.")
except Exception as e:
    print("CUDAExecutionProvider FAILED:", e)
    raise SystemExit(1) from e