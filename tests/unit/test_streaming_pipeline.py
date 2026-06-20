"""Unit tests for streaming video/GIF pipeline components."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from processing.gpu_scheduler import assign_frames_to_gpus, frame_pixel_weights
from utils import video_io


class TestGpuScheduler:
    def test_single_gpu_gets_all_frames(self):
        result = assign_frames_to_gpus(10, [0])
        assert result == {0: list(range(10))}

    def test_multi_gpu_splits_all_frames(self):
        with patch("processing.gpu_scheduler._gpu_free_vram_mb", return_value=1000.0):
            result = assign_frames_to_gpus(10, [0, 1])
        assert sum(len(v) for v in result.values()) == 10
        assert set(i for indices in result.values() for i in indices) == set(range(10))

    def test_frame_pixel_weights(self):
        frames = [np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((50, 50, 3), dtype=np.uint8)]
        weights = frame_pixel_weights(frames)
        assert weights[0] == 20000.0
        assert weights[1] == 2500.0


class TestStreamingVideoIO:
    def test_ffmpeg_stderr_drain_starts_for_writer(self, tmp_path):
        out = tmp_path / "drain_test.mp4"
        writer = video_io.StreamingVideoWriter(out, 8, 8, 2.0, use_nvenc=False)
        try:
            assert writer._stderr_drain._thread is not None
            assert writer._stderr_drain._thread.is_alive()
        finally:
            writer.finalize()

    def test_nvenc_available_does_not_crash(self):
        video_io._NVENC_AVAILABLE = None
        assert isinstance(video_io.nvenc_available(), bool)

    def test_cuda_hwaccel_available_does_not_crash(self):
        video_io._CUDA_HWACCEL_AVAILABLE = None
        assert isinstance(video_io.cuda_hwaccel_available(), bool)

    def test_zero_copy_decode_filter_includes_hwdownload(self):
        vf = video_io._decode_video_filter(30.0, zero_copy=True)
        assert "hwdownload" in vf
        assert "fps=30.0" in vf

    def test_gif_zero_copy_cuda_decode_fails(self, tmp_path):
        """GIF paletted decode cannot use CUDA hwaccel_output_format (regression guard)."""
        import subprocess
        from PIL import Image

        gif_path = tmp_path / "probe.gif"
        frames = [
            Image.fromarray(np.full((16, 16, 3), fill, dtype=np.uint8))
            for fill in (40, 80, 120)
        ]
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            str(gif_path),
            "-vf",
            video_io._decode_video_filter(10.0, zero_copy=True),
            "-pix_fmt",
            "rgb24",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0

    def test_streaming_roundtrip_small_video(self, tmp_path):
        """Write then read back a few frames via raw pipe."""
        out = tmp_path / "out.mp4"
        w, h, fps = 16, 16, 2.0
        frames = [
            np.full((h, w, 3), fill, dtype=np.uint8) for fill in (255, 128, 64)
        ]

        # Create source video first
        src = tmp_path / "src.mp4"
        with video_io.StreamingVideoWriter(
            src, w, h, fps, use_nvenc=False
        ) as writer:
            writer.write_frames(frames)

        with video_io.StreamingFrameReader(str(src), hwaccel=False) as reader:
            read_back = reader.read_chunk(10)

        assert len(read_back) == 3
        assert read_back[0].shape == (h, w, 3)

    def test_write_gif_from_numpy(self, tmp_path):
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        out = tmp_path / "test.gif"
        assert video_io.write_gif(frames, out, durations=[100, 100, 100])
        assert out.exists()


class TestFrameBatch:
    def test_process_frames_batch_default_mapping(self):
        from processing.frame_batch import process_frames_batch

        processor = MagicMock()
        face = MagicMock()
        processor.get_faces.return_value = [face]
        processor.swap_face_batch.return_value = np.ones((4, 4, 3), dtype=np.uint8)

        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        src_faces = [MagicMock()]

        with patch("processing.frame_batch.filter_faces_by_confidence", side_effect=lambda f, _: f):
            result = process_frames_batch(
                frames,
                src_faces,
                face_confidence=0.5,
                processor=processor,
            )

        assert len(result) == 1
        processor.swap_face_batch.assert_called_once()