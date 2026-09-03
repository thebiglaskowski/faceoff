"""Unit tests for UI preview helpers (no GPU, no ffmpeg)."""

import subprocess
from unittest.mock import patch


def test_first_frame_extract_timeout_returns_false(tmp_path):
    from ui.helpers.preview import _extract_first_frame_png

    out = tmp_path / "preview.png"
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60),
    ):
        assert _extract_first_frame_png("whatever.mp4", out) is False


def test_show_gif_preview_none_hides():
    from ui.helpers.preview import show_gif_preview

    update = show_gif_preview(None)
    assert update["visible"] is False
    assert update["value"] is None
