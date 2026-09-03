"""Unit tests for shared-palette GIF quantization (no GPU, no models)."""

import numpy as np


def _load():
    from utils import video_io

    return video_io


def _gradient_frame(seed=0, size=96):
    """Smooth skin-like gradient plus a touch of noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, size, dtype=np.float32)
    grad = np.tile(x, (size, 1))
    rgb = np.stack(
        [
            grad,
            np.clip(grad * 0.82 + 18, 0, 255),
            np.clip(grad * 0.68 + 12, 0, 255),
        ],
        axis=-1,
    )
    noise = rng.integers(0, 3, rgb.shape).astype(np.float32)
    return np.clip(rgb + noise, 0, 255).astype(np.uint8)


def test_global_palette_beats_per_frame_dither_on_flicker():
    """Near-identical frames must decode near-identically (no static crawl)."""
    from PIL import Image

    vio = _load()
    base = _gradient_frame(seed=1)
    near = _gradient_frame(seed=1)
    near[40:56, 40:56] = np.clip(
        near[40:56, 40:56].astype(np.int16) + 2, 0, 255
    ).astype(np.uint8)
    frames = [Image.fromarray(base), Image.fromarray(near)]

    shared = vio.quantize_frames_global_palette(frames)
    dithered = [
        f.convert("RGB").quantize(
            colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG
        )
        for f in frames
    ]

    def flicker(pair):
        a = np.asarray(pair[0].convert("RGB")).astype(np.int16)
        b = np.asarray(pair[1].convert("RGB")).astype(np.int16)
        return float(np.abs(a - b).mean())

    assert flicker(shared) < flicker(dithered)


def test_write_gif_roundtrip_frame_count(tmp_path):
    vio = _load()
    from PIL import Image

    out = tmp_path / "out.gif"
    frames = [_gradient_frame(seed=i) for i in range(4)]
    assert vio.write_gif(frames, out, fps=10) is True
    with Image.open(out) as im:
        assert im.format == "GIF"
        assert getattr(im, "n_frames", 1) == 4


def test_gif_encode_options_safe_defaults():
    vio = _load()
    colors, dither = vio._gif_encode_options()
    assert 2 <= colors <= 256
    assert isinstance(dither, bool)
