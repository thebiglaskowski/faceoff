"""Stub file for utils.video_io — mypy interface."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, overload

# Guard for mypy when the module itself isn't importable (e.g., no ffmpeg on PATH).


@dataclass(frozen=True)
class FrameResult:
    path: Path
    frame_index: int
    pts_sec: float


def probe_video(video_path: str) -> dict[str, str | int | float | bool]:
    """Probe video metadata using ffprobe --show_entries."""


def probe_gif(gif_path: str) -> dict[str, str | int | float | bool]:
    """Same interface as probe_video but works for GIF files."""


def extract_video_frames(
    video_path: str,
    output_dir: Union[str, Path],
    start_time: float = 0.0,
    duration: Optional[float] = None,
    fps: Optional[float] = None,
) -> List[FrameResult]:
    """Extract frames from video using FFmpeg. Returns sorted FrameResult list."""


def extract_video_frames_raw(
    video_path: str,
    output_dir: Union[str, Path],
    start_time: float = 0.0,
    duration: Optional[float] = None,
    fmt: str = "yuv420p",
) -> Path:
    """Extract raw YUV frame sequence into a single binary file."""


def write_video_from_pil_frames(
    frames,
    output_path: Union[str, Path],
    fps: float,
    audio_path: Optional[Union[str, Path]] = None,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
) -> bool:
    """Write video from PIL Image frames to a temp dir, then run FFmpeg."""


def write_video_from_frames(
    frames,
    output_path: Union[str, Path],
    fps: float,
    audio_path: Optional[Union[str, Path]] = None,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
) -> bool:
    """Write video from numpy frames (RGB arrays) or PIL Images."""


def extract_audio(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    codec: str = "aac",
    bitrate: str = "192k",
) -> Optional[Path]:
    """Extract audio track from video as AAC. Returns path or None."""


def extract_gif_frame_durations(gif_path: str) -> List[int]:
    """Extract frame durations from a GIF (in ms) using PIL/ffprobe."""


def write_gif(
    frames,
    output_path: Union[str, Path],
    durations: Optional[List[int]] = None,
    loop: int = 0,
    fps: Optional[float] = None,
) -> bool:
    """Write a GIF from PIL Images or numpy arrays, preserving per-frame durations."""


def get_video_info(video_path: str) -> dict[str, str | int | float | bool]:
    """Convenience function. Returns same dict as probe_video."""

# Re-export for backwards compatibility.
