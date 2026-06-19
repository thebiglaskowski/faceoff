"""Centralized ONNX Runtime execution provider configuration."""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Tuple, Union

from utils.config_manager import config

logger = logging.getLogger("FaceOff")

ProviderEntry = Union[str, Tuple[str, dict]]

_tensorrt_compile_lock = threading.Lock()


@contextmanager
def tensorrt_compile_guard() -> Iterator[None]:
    """Serialize TensorRT engine builds — Myelin is not safe across concurrent GPUs."""
    with _tensorrt_compile_lock:
        yield


def _site_packages() -> Path:
    import sys as _sys

    major, minor = _sys.version_info.major, _sys.version_info.minor
    return Path(_sys.prefix) / "lib" / f"python{major}.{minor}" / "site-packages"


def _tensorrt_lib_dirs() -> list[Path]:
    site = _site_packages()
    trt_libs = site / "tensorrt_libs"
    return [trt_libs] if trt_libs.is_dir() else []


def _tensorrt_load_candidates() -> list[Path | str]:
    """Ordered TensorRT library paths/names to try (pip wheel first, then system)."""
    candidates: list[Path | str] = []

    if sys.platform == "win32":
        pip_names = ("nvinfer_10.dll", "nvinfer.dll")
        for directory in _tensorrt_lib_dirs():
            for path in sorted(directory.glob("nvinfer*.dll"), reverse=True):
                if path.is_file():
                    candidates.append(path)
            for name in pip_names:
                path = directory / name
                if path.is_file():
                    candidates.append(path)
        candidates.extend(pip_names)
        return _dedupe_load_candidates(candidates)

    for directory in _tensorrt_lib_dirs():
        for path in sorted(directory.glob("libnvinfer.so*"), reverse=True):
            if path.is_file() and path.name.startswith("libnvinfer.so"):
                candidates.append(path)
    candidates.extend(("libnvinfer.so.10", "libnvinfer.so"))
    return _dedupe_load_candidates(candidates)


def _dedupe_load_candidates(candidates: list[Path | str]) -> list[Path | str]:
    seen: set[str] = set()
    ordered: list[Path | str] = []
    for item in candidates:
        key = str(item)
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def _try_load_tensorrt_library() -> str | None:
    for candidate in _tensorrt_load_candidates():
        try:
            ctypes.CDLL(str(candidate))
            logger.debug("TensorRT library loadable: %s", candidate)
            return str(candidate)
        except OSError:
            continue
    return None


def _is_ort_tensorrt_provider_loadable() -> bool:
    """Probe ORT TensorRT EP with a real session (provider .so cannot be dlopen'd alone)."""
    det_model = Path(config.buffalo_model_path) / "det_10g.onnx"
    if not det_model.is_file():
        logger.debug(
            "TensorRT EP probe deferred — %s not found (will verify on first session)",
            det_model,
        )
        return True

    try:
        import onnxruntime as ort
    except ImportError as exc:
        logger.warning("onnxruntime not importable for TensorRT probe: %s", exc)
        return False

    cache = Path(config.tensorrt_cache_dir) / "gpu_0"
    cache.mkdir(parents=True, exist_ok=True)
    probe_providers: List[ProviderEntry] = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_max_workspace_size": config.tensorrt_workspace_mb * 1024 * 1024,
                "trt_fp16_enable": config.tensorrt_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
            },
        ),
        ("CUDAExecutionProvider", _cuda_options(0)),
        "CPUExecutionProvider",
    ]

    try:
        session = ort.InferenceSession(str(det_model), providers=probe_providers)
        active = session.get_providers()
        if "TensorrtExecutionProvider" not in active:
            logger.warning("TensorRT EP probe session fell back to: %s", active)
            return False
        logger.debug("TensorRT EP probe session active: %s", active)
        return True
    except Exception as exc:
        logger.warning(
            "TensorRT EP probe failed — ensure tensorrt>=10.9,<11 is installed: %s",
            exc,
        )
        return False


@lru_cache(maxsize=1)
def is_tensorrt_runtime_available() -> bool:
    """Return True only when TensorRT libs and ORT's TensorRT EP both load."""
    if _try_load_tensorrt_library() is None:
        logger.info("TensorRT runtime libraries not found; CUDA-only inference")
        return False
    if not _is_ort_tensorrt_provider_loadable():
        logger.info(
            "TensorRT present but ONNX Runtime TensorRT EP unavailable; CUDA-only inference"
        )
        return False
    return True


def setup_nvidia_library_path() -> None:
    """
    Prepend NVIDIA pip package lib dirs to LD_LIBRARY_PATH.

    cuDNN must come first to avoid SUBLIBRARY_VERSION_MISMATCH from mixed loads.
    """
    import os

    site = _site_packages()
    ordered = [
        site / "nvidia" / "cudnn" / "lib",
        site / "tensorrt_libs",
    ]
    for lib_dir in ordered:
        if lib_dir.is_dir():
            os.environ["LD_LIBRARY_PATH"] = (
                str(lib_dir) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            )


def _cdll_global(path: Path) -> None:
    """dlopen a shared library into the global symbol namespace for ORT EP plugins."""
    mode = getattr(ctypes, "RTLD_GLOBAL", 0x00100)
    ctypes.CDLL(str(path), mode=mode)


def _preload_tensorrt_wheel_libs(trt_dir: Path) -> None:
    """Preload TensorRT libs ORT's TensorRT EP links against (libnvinfer.so.10, etc.)."""
    ordered_names = (
        "libnvinfer.so.10",
        "libnvinfer.so",
        "libnvonnxparser.so.10",
        "libnvinfer_plugin.so.10",
        "libnvinfer.so.11",
        "libnvonnxparser.so.11",
        "libnvinfer_plugin.so.11",
    )
    loaded: set[str] = set()
    for name in ordered_names:
        path = trt_dir / name
        if path.is_file() and name not in loaded:
            try:
                _cdll_global(path)
                loaded.add(name)
                logger.debug("Preloaded TensorRT library: %s", path)
            except OSError as exc:
                logger.debug("TensorRT preload skipped %s: %s", name, exc)


def preload_nvidia_libraries() -> None:
    """
    Preload cuDNN + TensorRT wheels before onnxruntime import.

    Changing LD_LIBRARY_PATH after process start does not help dlopen; explicit
    RTLD_GLOBAL preload is required for ORT's TensorRT execution provider.
    """
    if sys.platform == "win32":
        return

    site = _site_packages()
    cudnn = site / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9"
    if cudnn.exists():
        try:
            _cdll_global(cudnn)
            logger.debug("Preloaded cuDNN: %s", cudnn)
        except OSError as exc:
            logger.debug("cuDNN preload skipped: %s", exc)

    for trt_dir in _tensorrt_lib_dirs():
        _preload_tensorrt_wheel_libs(trt_dir)


def _cuda_options(device_id: int) -> dict:
    gpu_mem_limit_mb = config.get("gpu", "onnx_mem_limit_mb", default=2048)
    algo = config.get("gpu", "cudnn_conv_algo_search", default="HEURISTIC")
    return {
        "device_id": device_id,
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": gpu_mem_limit_mb * 1024 * 1024,
        "cudnn_conv_algo_search": algo,
        "do_copy_in_default_stream": True,
    }


def build_face_analysis_providers(
    device_id: int,
    *,
    use_tensorrt: bool | None = None,
    tensorrt_fp16: bool | None = None,
) -> List[ProviderEntry]:
    """Providers for InsightFace FaceAnalysis (detection / landmarks)."""
    if use_tensorrt is None:
        use_tensorrt = config.tensorrt_enabled
    if tensorrt_fp16 is None:
        tensorrt_fp16 = config.tensorrt_fp16

    providers: List[ProviderEntry] = []
    if use_tensorrt and is_tensorrt_runtime_available():
        cache = Path(config.tensorrt_cache_dir) / f"gpu_{device_id}"
        cache.mkdir(parents=True, exist_ok=True)
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": device_id,
                    "trt_max_workspace_size": config.tensorrt_workspace_mb * 1024 * 1024,
                    "trt_fp16_enable": tensorrt_fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                },
            )
        )
    providers.append(("CUDAExecutionProvider", _cuda_options(device_id)))
    providers.append("CPUExecutionProvider")
    return providers


def build_swapper_providers(device_id: int) -> List[ProviderEntry]:
    """Providers for inswapper (CUDA only — TensorRT incompatible with this model)."""
    return [
        ("CUDAExecutionProvider", _cuda_options(device_id)),
        "CPUExecutionProvider",
    ]


def build_onnx_providers(
    device_id: int = 0,
    use_tensorrt: bool = True,
    use_fp16: bool = True,
    workspace_mb: int = 2048,
) -> List[ProviderEntry]:
    """Legacy helper used by tensorrt_utils.get_onnx_providers."""
    if use_tensorrt and is_tensorrt_runtime_available():
        return build_face_analysis_providers(
            device_id, use_tensorrt=True, tensorrt_fp16=use_fp16
        )
    return [
        ("CUDAExecutionProvider", _cuda_options(device_id)),
        "CPUExecutionProvider",
    ]