"""OSEDiff one-step diffusion restoration (image-only quality mode).

Runs AFTER face restoration (CodeFormer/GFPGAN owns identity) as a final
generative cleanup pass for still images. One diffusion step takes seconds;
it is deliberately NOT offered for GIF/video (per-frame diffusion flickers
and multiplies runtime by frame count).

First-run checklist (author could not verify from a network-isolated box):
  1. ``uv sync`` installs diffusers/accelerate/peft (new pyproject deps).
  2. Weights snapshot from ``osediff.repo_id`` — if filenames differ from
     ``lora_file``/``vae_file``, correct config.yaml and report back.
  3. Smoke: driver.py img→img with ``--enhance`` on the image tab.

Follows the HAT module pattern: LRUModelCache per GPU, serialized loads,
lazy heavy imports with a friendly error when deps are missing.
"""

from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from utils.error_handler import FriendlyError
from utils.lru_cache import LRUModelCache
from utils.memory_manager import refresh_gpu_memory

logger = logging.getLogger("FaceOff")

OSEDIFF_VARIANT = "OSEDiff_SD21 (Quality — slow, image only)"


def _opts():
    from utils.config_manager import config

    return {
        "enabled": bool(config.get("osediff", "enabled", default=True)),
        "repo_id": str(config.get("osediff", "repo_id", default="csguoh/OSEDiff")),
        "lora_file": str(
            config.get("osediff", "lora_file", default="osediff_lora.safetensors")
        ),
        "vae_file": str(
            config.get(
                "osediff",
                "vae_file",
                default="vae-ft-mse-840000-ema-pruned.safetensors",
            )
        ),
        "base_model": str(
            config.get(
                "osediff",
                "base_model",
                default="stabilityai/stable-diffusion-2-1-base",
            )
        ),
        "local_dir": str(config.get("osediff", "local_dir", default="models/osediff")),
        "steps": int(config.get("osediff", "num_inference_steps", default=1)),
        "strength": float(config.get("osediff", "strength", default=0.75)),
        "max_side_px": int(config.get("osediff", "max_side_px", default=1024)),
        "prompt": str(
            config.get(
                "osediff",
                "prompt",
                default="clean, high-resolution, detailed photo",
            )
        ),
    }


def _require_diffusers():
    """Import diffusers stack or raise a user-actionable error."""
    try:
        import diffusers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as exc:
        raise FriendlyError(
            "OSEDiff dependencies missing",
            "OSEDiff needs the diffusion stack (diffusers/accelerate/peft).",
            ["Install it with: uv sync"],
            technical_details=str(exc),
        ) from exc


def ensure_osediff_weights() -> Path:
    """Snapshot LoRA + VAE weights into models/osediff (unless present)."""
    opts = _opts()
    dest = Path(opts["local_dir"])
    have_lora = (dest / opts["lora_file"]).exists()
    have_vae = (dest / opts["vae_file"]).exists()
    if have_lora and have_vae:
        return dest
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FriendlyError(
            "OSEDiff weight download unavailable",
            "Fetching OSEDiff weights needs huggingface_hub.",
            ["Install it with: uv sync"],
            technical_details=str(exc),
        ) from exc
    logger.info("Downloading OSEDiff weights (%s) ...", opts["repo_id"])
    try:
        snapshot_download(
            repo_id=opts["repo_id"],
            local_dir=str(dest),
            allow_patterns=[opts["lora_file"], opts["vae_file"]],
        )
    except Exception as exc:
        raise FriendlyError(
            "OSEDiff weight download failed",
            f"Could not fetch OSEDiff weights from '{opts['repo_id']}'. "
            "Check osediff.repo_id / filenames in config.yaml.",
            ["Verify the repo exists; use --osediff-repo to override."],
            technical_details=str(exc),
        ) from exc
    return dest


def _osediff_cleanup(pipe) -> None:
    """Drop a diffusion pipeline and reclaim VRAM."""
    try:
        import torch

        try:
            pipe.to("cpu")
        except Exception:
            pass
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


_osediff_cache = LRUModelCache("OSEDiff", cleanup_fn=_osediff_cleanup)
_osediff_load_lock = threading.Lock()
_osediff_gpu_locks: Dict[int, threading.Lock] = {}
_osediff_gpu_locks_guard = threading.Lock()


def _osediff_gpu_lock(gpu_id: int) -> threading.Lock:
    with _osediff_gpu_locks_guard:
        if gpu_id not in _osediff_gpu_locks:
            _osediff_gpu_locks[gpu_id] = threading.Lock()
        return _osediff_gpu_locks[gpu_id]


def _build_pipeline(gpu_id: int):
    """Assemble SD2.1 img2img + OSEDiff LoRA + fine-tuned VAE.

    VERIFY on first GPU run: exact LoRA/scheduler call shapes against the
    installed diffusers version and the snapshot layout. Everything
    version-sensitive lives in this one function.
    """
    _require_diffusers()
    import torch
    from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline

    opts = _opts()
    dest = ensure_osediff_weights()
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    logger.info("Loading OSEDiff pipeline on %s ...", device)
    vae = AutoencoderKL.from_single_file(
        str(dest / opts["vae_file"]), torch_dtype=dtype
    )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        opts["base_model"], vae=vae, torch_dtype=dtype
    )
    lora_path = dest / opts["lora_file"]
    if lora_path.exists():
        pipe.load_lora_weights(str(dest), weight_name=opts["lora_file"])
    else:
        logger.warning("LoRA file %s missing — running base SD2.1 one-step", lora_path)
    try:
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    except Exception:
        pipe.to(device)
    return pipe


def _get_pipeline(gpu_id: int):
    """Cached pipeline per GPU; loads serialized like HAT (WSL2-safe)."""
    cached = _osediff_cache.get(gpu_id)
    if cached is not None:
        return cached
    with _osediff_load_lock:
        cached = _osediff_cache.get(gpu_id)
        if cached is not None:
            return cached
        pipe = _build_pipeline(gpu_id)
        _osediff_cache.put(gpu_id, pipe)
        return pipe


def clear_osediff_cache() -> None:
    """Release OSEDiff pipelines (also called on app shutdown paths)."""
    with _osediff_load_lock:
        _osediff_cache.clear()
    gc.collect()
    logger.info("OSEDiff cache cleared")


def _fit_to_vram_side(
    bgr: np.ndarray, max_side: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Downscale longest side to max_side; return (image, original_hw)."""
    import cv2

    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr, (h, w)
    scale = max_side / longest
    small = cv2.resize(
        bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    return small, (h, w)


def enhance_image_osediff(
    bgr: np.ndarray,
    gpu_id: int = 0,
    model_name: Optional[str] = None,  # accepted for dispatch symmetry; single variant
) -> np.ndarray:
    """One-step diffusion cleanup of a BGR image; returns same-size BGR uint8."""
    import cv2
    import torch
    from PIL import Image

    _require_diffusers()
    opts = _opts()
    small, (orig_h, orig_w) = _fit_to_vram_side(bgr, opts["max_side_px"])
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_in = Image.fromarray(rgb)

    with _osediff_gpu_lock(gpu_id):
        pipe = _get_pipeline(gpu_id)
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        gen_device = device if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(42)
        with torch.no_grad():
            out = pipe(
                prompt=opts["prompt"],
                image=pil_in,
                strength=opts["strength"],
                num_inference_steps=opts["steps"],
                generator=generator,
            ).images[0]

    result_rgb = np.asarray(out)
    if (result_rgb.shape[0], result_rgb.shape[1]) != (orig_h, orig_w):
        result_rgb = cv2.resize(
            result_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
        )
    result_bgr = cv2.cvtColor(result_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    try:
        refresh_gpu_memory(gpu_id)
    except Exception as exc:
        logger.debug("OSEDiff post-run GPU refresh: %s", exc)
    return result_bgr
