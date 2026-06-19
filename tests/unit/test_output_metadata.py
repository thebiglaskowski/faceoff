"""Tests for output metadata sidecar files."""

import json
from pathlib import Path

from processing.orchestrator import ProcessOptions
from utils.output_metadata import (
    build_gallery_caption,
    format_settings_detail,
    format_settings_summary,
    load_output_metadata,
    metadata_path_for,
    save_output_metadata,
)


def test_save_and_load_metadata(tmp_path):
    out = tmp_path / "swapped_123.png"
    out.write_bytes(b"png")

    opts = ProcessOptions(
        source_image=__import__("numpy").zeros((1, 1, 3)),
        dest_path="target.png",
        media_type="image",
        output_dir=str(tmp_path),
        enhance=True,
        model_display="RealESRGAN_x4plus (General)",
        gpu_selection="GPU 0: RTX 3060 Ti",
    )

    meta_path = save_output_metadata(out, opts)
    assert meta_path == metadata_path_for(out)
    assert meta_path.exists()

    loaded = load_output_metadata(out)
    assert loaded is not None
    assert loaded["settings"]["enhance"] is True
    assert loaded["settings"]["model_display"] == "RealESRGAN_x4plus (General)"
    assert "source_image" not in loaded["settings"]


def test_format_settings_summary():
    summary = format_settings_summary(
        {
            "enhance": True,
            "enhancement_model": "RealESRGAN",
            "model_display": "General x4",
            "outscale": 4,
            "restore_faces": False,
            "gpu_selection": "GPU 0: RTX 3060 Ti",
        }
    )
    assert "RealESRGAN" in summary
    assert "Restore: off" in summary
    assert "GPU 0" in summary


def test_build_gallery_caption_with_metadata(tmp_path):
    from datetime import datetime

    out = tmp_path / "test.gif"
    out.write_bytes(b"GIF89a")
    meta = {
        "settings": {
            "enhance": False,
            "restore_faces": False,
            "gpu_selection": "All GPUs: 2 devices",
        }
    }
    metadata_path_for(out).write_text(json.dumps(meta), encoding="utf-8")

    caption = build_gallery_caption(out, datetime(2026, 6, 19, 10, 0, 0))
    lines = caption.split("\n")
    assert lines[0] == "test.gif"
    assert "Enhance: off" in lines[2]


def test_format_settings_detail_missing():
    detail = format_settings_detail(None, "/outputs/image/old.png")
    assert "No metadata found" in detail