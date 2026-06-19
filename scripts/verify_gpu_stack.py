#!/usr/bin/env python3
"""Smoke-test the GPU inference stack before running the Gradio UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.onnx_providers import (  # noqa: E402
    is_tensorrt_runtime_available,
    preload_nvidia_libraries,
    setup_nvidia_library_path,
)

setup_nvidia_library_path()
preload_nvidia_libraries()


def main() -> int:
    import numpy as np
    import onnxruntime as ort
    import torch

    print("=== FaceOff GPU Stack Verification ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Providers: {ort.get_available_providers()}")
    trt_ok = is_tensorrt_runtime_available()
    print(f"TensorRT runtime + ORT EP loadable: {trt_ok}")

    det_model = ROOT / "models" / "buffalo_l" / "det_10g.onnx"
    if not det_model.exists():
        print(f"SKIP inference: model not found at {det_model}")
        return 0

    if trt_ok:
        from utils.onnx_providers import build_face_analysis_providers

        trt_sess = ort.InferenceSession(
            str(det_model),
            providers=build_face_analysis_providers(0, use_tensorrt=True),
        )
        print(f"TensorRT session providers: {trt_sess.get_providers()}")

    sess = ort.InferenceSession(
        str(det_model),
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ],
    )
    active = sess.get_providers()
    print(f"Session providers: {active}")

    inp = sess.get_inputs()[0]
    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    out = sess.run(None, {inp.name: data})
    print(f"Inference OK — output shapes: {[o.shape for o in out]}")

    from core.media_processor import MediaProcessor  # noqa: E402

    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    if trt_ok:
        proc_trt = MediaProcessor(device_id=0, use_tensorrt=True, optimize_models=False)
        faces_trt = proc_trt.get_faces(blank)
        det_providers = proc_trt._gpu._det_providers()
        print(
            f"MediaProcessor (TensorRT) OK — {len(faces_trt)} face(s), "
            f"detection providers: {det_providers}"
        )
        if det_providers and "TensorrtExecutionProvider" not in det_providers:
            print(
                "WARNING: Face detection did not activate TensorRT EP",
                file=sys.stderr,
            )
            return 1
    else:
        print("SKIP MediaProcessor TensorRT path — TRT not loadable")

    proc = MediaProcessor(device_id=0, use_tensorrt=False, optimize_models=False)
    faces = proc.get_faces(blank)
    print(f"MediaProcessor (CUDA) OK — {len(faces)} face(s) on blank image")
    print("=== All checks passed ===")
    print(
        "Note: WeightsContext.cpp warnings from buffalo_l ONNX are benign TensorRT noise."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc