"""Unit tests for OSEDiff integration (no GPU, no diffusers, no weights)."""

import importlib.util

import numpy as np
import pytest

from processing.in_memory_enhancement import InMemoryEnhancer

_HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None
needs_no_diffusers = pytest.mark.skipif(
    _HAS_DIFFUSERS, reason="expects diffusers to be absent"
)


def _load():
    from processing import osediff_enhancement

    return osediff_enhancement


def test_opts_defaults_keep_single_step():
    od = _load()
    opts = od._opts()
    assert opts["steps"] == 1
    assert opts["max_side_px"] >= 512
    assert 0.0 < opts["strength"] <= 1.0
    assert opts["local_dir"] == "models/osediff"


@needs_no_diffusers
def test_missing_diffusers_is_actionable():
    od = _load()
    from utils.error_handler import FriendlyError

    with pytest.raises(FriendlyError, match="diffusion stack") as exc_info:
        od._require_diffusers()
    assert any("uv sync" in s for s in exc_info.value.suggestions)


@needs_no_diffusers
def test_enhance_without_deps_raises_before_touching_weights():
    od = _load()
    from utils.error_handler import FriendlyError

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(FriendlyError):
        od.enhance_image_osediff(frame, gpu_id=0)


def test_dispatch_routes_osediff_frames():
    from unittest.mock import patch

    from processing.osediff_enhancement import OSEDIFF_VARIANT

    enhancer = InMemoryEnhancer("OSEDiff", OSEDIFF_VARIANT, [0])
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
    sentinel = np.ones((16, 16, 3), dtype=np.uint8) * 7
    with patch(
        "processing.osediff_enhancement.enhance_image_osediff",
        return_value=sentinel,
    ) as mock_enh:
        out = enhancer._enhance_bgr_batch(frames, 0)
    assert mock_enh.call_count == 2
    assert all(np.array_equal(f, sentinel) for f in out)


def test_dispatch_keeps_original_on_frame_error():
    from unittest.mock import patch

    from processing.osediff_enhancement import OSEDIFF_VARIANT

    enhancer = InMemoryEnhancer("OSEDiff", OSEDIFF_VARIANT, [0])
    frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
    with patch(
        "processing.osediff_enhancement.enhance_image_osediff",
        side_effect=RuntimeError("boom"),
    ):
        assert enhancer._enhance_bgr_batch(frames, 0) == frames


def test_fit_to_vram_side_downscales_long_side_only():
    od = _load()
    wide = np.zeros((400, 1600, 3), dtype=np.uint8)
    small, orig = od._fit_to_vram_side(wide, 800)
    assert orig == (400, 1600)
    assert max(small.shape[:2]) == 800
    tiny = np.zeros((64, 64, 3), dtype=np.uint8)
    same, _ = od._fit_to_vram_side(tiny, 800)
    assert same.shape == tiny.shape


def test_fetch_osediff_skips_when_present(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    import download_models

    dest = tmp_path / "models" / "osediff"
    dest.mkdir(parents=True)
    (dest / "osediff_lora.safetensors").write_bytes(b"x")
    (dest / "vae-ft-mse-840000-ema-pruned.safetensors").write_bytes(b"y")
    assert download_models.fetch_osediff(tmp_path, "whatever/repo") == dest
