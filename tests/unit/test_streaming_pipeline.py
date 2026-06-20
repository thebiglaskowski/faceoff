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

    def test_multi_gpu_collects_swapped_frames_without_gpu_paste(self, mock_gpu):
        """Multi-GPU must not use GPU paste (shared buffer); every frame is returned."""
        from processing.frame_batch import process_chunk_multi_gpu
        from processing.workload_profile import WorkloadProfile

        frames = [
            np.full((4, 4, 3), fill, dtype=np.uint8) for fill in (10, 20, 30, 40)
        ]
        face = MagicMock()
        profile = WorkloadProfile(
            name="stream_swap_only",
            frame_retention=True,
            paste_on_gpu=True,
            detection_on_gpu=False,
            enhancement_chain=False,
            zero_copy=False,
            pinned_encode=False,
            defer_download=False,
        )

        def _make_gpu(dev_id):
            inst = MagicMock()
            inst.device_id = dev_id
            inst.get_faces.return_value = [face]
            inst.swap_face_batch.side_effect = (
                lambda img, dst, src, **kw: np.full(img.shape, dev_id * 100 + 7, np.uint8)
            )
            return inst

        gpu_instances = [_make_gpu(0), _make_gpu(1)]

        with patch("processing.frame_batch.assign_frames_to_gpus") as mock_assign, patch(
            "processing.frame_batch.filter_faces_by_confidence",
            side_effect=lambda f, _: f,
        ), patch("processing.frame_batch.config") as cfg:
            mock_assign.return_value = {0: [0, 1], 1: [2, 3]}
            cfg.workers_per_gpu = 1
            cfg.iou_threshold = 0.5
            cfg.gpu_frame_retention_enabled = True
            cfg.gpu_paste_on_gpu = True
            cfg.gpu_detection_on_gpu = False

            result = process_chunk_multi_gpu(
                frames,
                src_faces=[MagicMock()],
                device_ids=[0, 1],
                gpu_instances=gpu_instances,
                face_confidence=0.5,
                batch_size=8,
                profile=profile,
            )

        assert len(result) == 4
        assert result[0][0, 0, 0] == 7
        assert result[1][0, 0, 0] == 7
        assert result[2][0, 0, 0] == 107
        assert result[3][0, 0, 0] == 107
        for inst in gpu_instances:
            for call in inst.swap_face_batch.call_args_list:
                assert call.kwargs.get("paste_on_gpu") is False


class TestEnhanceSwapVram:
    def test_postprocess_releases_swap_before_enhance(self, mock_gpu):
        from processing.streaming_media import _postprocess_chunk
        from unittest.mock import MagicMock, patch

        enhancer = MagicMock()
        enhancer.device_ids = [0, 1]
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        with patch(
            "processing.streaming_media.prepare_for_enhancement"
        ) as mock_prepare, patch(
            "processing.streaming_media.refresh_gpu_memory"
        ) as mock_refresh, patch(
            "processing.streaming_media.config"
        ) as cfg:
            cfg.release_swap_models_before_enhance = True
            enhancer.enhance_rgb_frames.return_value = frames
            result = _postprocess_chunk(
                frames,
                restoration_session=None,
                enhancer=enhancer,
                maintain_dimensions=True,
                original_size=(4, 4),
            )

        mock_prepare.assert_called_once()
        mock_refresh.assert_called_once_with(device_ids=[0, 1], force=True)
        assert result == frames


class TestMultiGpuWorkloadTrim:
    def test_streaming_trims_profile_for_multi_gpu(self, mock_gpu):
        from processing.streaming_media import process_streaming
        from processing.workload_profile import WorkloadProfile
        from unittest.mock import MagicMock, patch

        profile = WorkloadProfile(
            name="stream_swap_only",
            frame_retention=True,
            paste_on_gpu=True,
            detection_on_gpu=True,
            enhancement_chain=False,
            zero_copy=True,
            pinned_encode=True,
            defer_download=False,
            chunk_size=32,
        )

        with patch("processing.streaming_media.resolve_workload_profile", return_value=profile), patch(
            "processing.streaming_media._detect_source_faces",
            return_value=[MagicMock()],
        ), patch(
            "processing.streaming_media.video_io.open_streaming_reader"
        ) as mock_reader, patch(
            "processing.streaming_media.log_workload_profile"
        ) as mock_log, patch(
            "processing.streaming_media.get_progress_tracker"
        ) as mock_progress, patch(
            "processing.streaming_media.get_model_pool"
        ) as mock_pool, patch(
            "processing.streaming_media._process_chunk_with_oom_fallback",
            return_value=([], None),
        ), patch(
            "processing.streaming_media._postprocess_chunk",
            return_value=[],
        ), patch(
            "processing.streaming_media._write_chunk_frames"
        ), patch(
            "processing.streaming_media._effective_chunk_size",
            return_value=32,
        ), patch(
            "processing.streaming_media.MemoryManager"
        ):
            reader = MagicMock()
            reader.__enter__.return_value = reader
            reader.read_chunk.side_effect = [[np.zeros((4, 4, 3), dtype=np.uint8)], []]
            reader.width = 1920
            reader.height = 1080
            mock_reader.return_value = reader
            mock_progress.return_value = MagicMock()
            mock_pool.return_value.get_instances.return_value = [MagicMock(), MagicMock()]

            ctx = MagicMock()
            ctx.media_type = "video"
            ctx.dest_path = "test.mp4"
            ctx.output_dir = MagicMock()
            ctx.output_dir.resolve.return_value = MagicMock()
            ctx.fps = 30.0
            ctx.width = 1920
            ctx.height = 1080
            ctx.audio_path = None

            writer = MagicMock()
            writer.__enter__.return_value = writer
            writer.finalize.return_value = True
            writer.frames_written = 1

            with patch("processing.streaming_media.video_io.StreamingVideoWriter", return_value=writer):
                process_streaming(
                    MagicMock(),
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    ctx,
                    device_ids=[0, 1],
                    workload_profile=profile,
                )

        logged = mock_log.call_args[0][0]
        assert logged.name == "stream_swap_only_multi_gpu"
        assert logged.frame_retention is False
        assert logged.detection_on_gpu is False


class TestPinnedEncode:
    def test_stack_rgb_frames_rejects_scalars(self):
        from processing.streaming_media import _stack_rgb_frames

        with pytest.raises(ValueError, match="expected HxWx3"):
            _stack_rgb_frames([np.uint8(7)] * 16)

    def test_stack_rgb_frames_contiguous_batch(self):
        from processing.streaming_media import _stack_rgb_frames

        frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in range(3)]
        batch = _stack_rgb_frames(frames)
        assert batch.shape == (3, 4, 6, 3)
        assert batch.flags["C_CONTIGUOUS"]

    def test_write_chunk_frames_skips_pinned_when_disabled(self, mock_gpu):
        from processing.streaming_media import _write_chunk_frames

        writer = MagicMock()
        frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(4)]
        _write_chunk_frames(
            writer,
            frames,
            use_pinned_encode=False,
            frame_buffer=None,
            gpu_paste_active=False,
        )
        writer.write_frames_pinned.assert_not_called()
        writer.write_frames.assert_called_once_with(frames)

    def test_write_chunk_frames_pinned_rejects_scalar_batch(self, mock_gpu):
        from processing.streaming_media import _write_chunk_frames

        writer = MagicMock()
        _write_chunk_frames(
            writer,
            [np.uint8(i) for i in range(16)],
            use_pinned_encode=True,
            frame_buffer=None,
            gpu_paste_active=False,
        )
        writer.write_frames_pinned.assert_not_called()
        writer.write_frames.assert_called_once()