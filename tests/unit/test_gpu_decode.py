"""Tests for PyNvVideoCodec NVDEC decode (Wave 5)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestNvcodecProbe:
    def test_available_when_pynvvideocodec_imports(self):
        import utils.gpu_decode as gd

        gd._NVcodec_AVAILABLE = None
        gd._LEGACY_NVcodec_AVAILABLE = None
        with patch.dict("sys.modules", {"PyNvVideoCodec": MagicMock()}):
            assert gd.nvcodec_decode_available() is True

    def test_unavailable_when_no_package(self):
        import utils.gpu_decode as gd

        gd._NVcodec_AVAILABLE = False
        gd._LEGACY_NVcodec_AVAILABLE = False
        assert gd.nvcodec_decode_available() is False


class TestOpenStreamingReader:
    def test_uses_nvcodec_reader_when_requested(self, tmp_path):
        from utils import video_io

        fake_reader = MagicMock()
        with patch(
            "utils.gpu_decode.nvcodec_decode_available", return_value=True
        ), patch(
            "utils.gpu_decode.NvCodecFrameReader", return_value=fake_reader
        ) as mock_nv:
            reader = video_io.open_streaming_reader(
                str(tmp_path / "clip.mp4"),
                use_nvcodec=True,
            )

        assert reader is fake_reader
        mock_nv.assert_called_once()

    def test_falls_back_to_ffmpeg_on_nvcodec_failure(self, tmp_path):
        from utils import video_io

        with patch(
            "utils.gpu_decode.nvcodec_decode_available", return_value=True
        ), patch(
            "utils.gpu_decode.NvCodecFrameReader",
            side_effect=RuntimeError("no gpu"),
        ), patch.object(
            video_io, "StreamingFrameReader"
        ) as mock_ff:
            mock_ff.return_value = MagicMock()
            reader = video_io.open_streaming_reader(
                str(tmp_path / "clip.mp4"),
                use_nvcodec=True,
                hwaccel=True,
            )

        assert reader is mock_ff.return_value
        mock_ff.assert_called_once()

    def test_skips_nvcodec_when_not_installed(self, tmp_path):
        from utils import video_io

        with patch(
            "utils.gpu_decode.nvcodec_decode_available", return_value=False
        ), patch.object(
            video_io, "StreamingFrameReader"
        ) as mock_ff:
            mock_ff.return_value = MagicMock()
            video_io.open_streaming_reader(
                str(tmp_path / "clip.mp4"),
                use_nvcodec=True,
            )

        mock_ff.assert_called_once()


class TestNvCodecFrameReader:
    def test_read_chunk_copies_frames(self):
        from utils.gpu_decode import NvCodecFrameReader

        decoded = MagicMock()
        decoded.shape = (2, 2, 3)
        view = MagicMock()
        view.dataptr = 0
        decoded.cuda.return_value = [view]

        meta = MagicMock()
        meta.width = 2
        meta.height = 2
        meta.average_fps = 30.0
        meta.num_frames = 4

        decoder = MagicMock()
        decoder.get_stream_metadata.return_value = meta
        decoder.get_batch_frames.side_effect = [[decoded], []]

        fake_rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        with patch(
            "utils.gpu_decode._decoded_frame_to_numpy", return_value=fake_rgb
        ), patch(
            "PyNvVideoCodec.CreateSimpleDecoder", return_value=decoder
        ):
            reader = NvCodecFrameReader("clip.mp4")
            chunk = reader.read_chunk(1)

        assert len(chunk) == 1
        assert chunk[0].shape == (2, 2, 3)
        assert reader.frames_read == 1
        assert reader.read_chunk(1) == []