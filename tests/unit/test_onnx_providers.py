"""Unit tests for ONNX provider configuration."""

import threading
import time
from pathlib import Path
from unittest.mock import patch


def test_tensorrt_load_candidates_include_versioned_pip_libs(tmp_path, monkeypatch):
    from utils import onnx_providers

    trt_dir = tmp_path / "tensorrt_libs"
    trt_dir.mkdir()
    (trt_dir / "libnvinfer.so.11").touch()

    monkeypatch.setattr(onnx_providers, "_site_packages", lambda: tmp_path)
    candidates = onnx_providers._tensorrt_load_candidates()
    assert trt_dir / "libnvinfer.so.11" in candidates


def test_build_face_analysis_uses_tensorrt_when_runtime_available():
    from utils.onnx_providers import build_face_analysis_providers

    with patch("utils.onnx_providers.is_tensorrt_runtime_available", return_value=True):
        providers = build_face_analysis_providers(0, use_tensorrt=True)
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    assert names[0] == "TensorrtExecutionProvider"
    assert "CUDAExecutionProvider" in names


def test_tensorrt_compile_guard_serializes_workers():
    from utils.onnx_providers import tensorrt_compile_guard

    order: list[str] = []

    def worker(tag: str) -> None:
        with tensorrt_compile_guard():
            order.append(f"{tag}-start")
            time.sleep(0.05)
            order.append(f"{tag}-end")

    threads = [
        threading.Thread(target=worker, args=("a",)),
        threading.Thread(target=worker, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    a_start, a_end = order.index("a-start"), order.index("a-end")
    b_start, b_end = order.index("b-start"), order.index("b-end")
    assert (a_end < b_start and a_start < b_start) or (b_end < a_start and b_start < a_start)


def test_tensorrt_unavailable_when_libs_missing():
    from utils import onnx_providers

    onnx_providers.is_tensorrt_runtime_available.cache_clear()
    with patch("ctypes.CDLL", side_effect=OSError("missing")):
        assert onnx_providers.is_tensorrt_runtime_available() is False
    onnx_providers.is_tensorrt_runtime_available.cache_clear()


def test_build_swapper_providers_cuda_only():
    from utils.onnx_providers import build_swapper_providers

    providers = build_swapper_providers(0)
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    assert names[0] == "CUDAExecutionProvider"
    assert "TensorrtExecutionProvider" not in names
    assert names[-1] == "CPUExecutionProvider"


def test_build_face_analysis_heuristic_cudnn():
    from utils.onnx_providers import build_face_analysis_providers

    with patch("utils.onnx_providers.is_tensorrt_runtime_available", return_value=False):
        providers = build_face_analysis_providers(0, use_tensorrt=True)
    cuda_opts = providers[0][1]
    assert cuda_opts["cudnn_conv_algo_search"] == "HEURISTIC"