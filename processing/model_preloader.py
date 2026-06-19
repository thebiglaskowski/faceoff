"""
Model preloading functionality for FaceOff.

This module handles preloading models at startup to reduce first-run delay.
Lives in processing layer since it needs to import from core.

Architecture (from CLAUDE.md):
- processing/ may import from core/, utils/
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.gpu_manager import GPUManager
from core.model_pool import get_model_pool
from utils.config_manager import config
from utils.onnx_providers import is_tensorrt_runtime_available

logger = logging.getLogger("FaceOff")


def _resolve_preload_device_ids(device_ids: Optional[List[int]] = None) -> List[int]:
    if device_ids is not None:
        return device_ids
    if not GPUManager.is_available():
        return []
    return list(range(GPUManager.get_device_count()))


def preload_models(device_ids: Optional[List[int]] = None) -> None:
    """
    Preload pooled face-detection and swapper models at startup.

    When TensorRT is enabled, engine builds run sequentially per GPU via the
    global compile lock so multi-GPU warmup is safe before the first request.

    Args:
        device_ids: GPUs to warm. Defaults to every available CUDA device.
    """
    if not config.preload_on_startup:
        logger.info("Model preloading disabled in config")
        return

    device_ids = _resolve_preload_device_ids(device_ids)
    if not device_ids:
        logger.info("No CUDA GPUs available — skipping model preload")
        return

    use_tensorrt = bool(config.tensorrt_enabled and is_tensorrt_runtime_available())

    pool = get_model_pool()
    pool.set_model_path(config.inswapper_model_path)

    logger.info(
        "Preloading models on %d GPU(s) %s (TensorRT=%s)...",
        len(device_ids),
        device_ids,
        use_tensorrt,
    )

    primary_gpu = device_ids[0]
    warmed = 0
    for device_id in device_ids:
        try:
            trt_for_gpu = use_tensorrt and device_id == primary_gpu
            logger.info(
                "Preloading face detection + swapper on GPU %d (TensorRT=%s)...",
                device_id,
                trt_for_gpu,
            )
            instance = pool.get_instance(device_id, use_tensorrt=trt_for_gpu)
            if not trt_for_gpu:
                from core.model_pool import _trt_warmup_images

                instance.get_faces(_trt_warmup_images()[0])
            warmed += 1
            logger.info("GPU %d preload complete", device_id)
        except Exception as exc:
            logger.warning(
                "Model preloading failed on GPU %d: %s (will compile on first use)",
                device_id,
                exc,
            )

    if warmed:
        logger.info(
            "Model preloading finished — %d/%d GPU(s) ready",
            warmed,
            len(device_ids),
        )
    else:
        logger.warning("Model preloading did not warm any GPU")

    if config.get("model_cache", "preload_enhancement_on_startup", default=False):
        _preload_default_enhancement(primary_gpu)


def _preload_default_enhancement(gpu_id: int) -> None:
    """Optionally warm the default Real-ESRGAN upsampler on the primary GPU."""
    try:
        from processing.enhancement import _get_upsampler

        model_name = config.default_enhancement_model
        tile_size = config.default_tile_size
        _get_upsampler(
            model_name=model_name,
            gpu_id=gpu_id,
            tile_size=tile_size,
            pre_pad=config.default_pre_pad,
            use_fp32=config.default_use_fp32,
            denoise_strength=config.default_denoise_strength,
        )
        logger.info("Enhancement model preloaded on GPU %d (%s)", gpu_id, model_name)
    except Exception as exc:
        logger.warning("Enhancement preload skipped: %s", exc)