"""Tests for GPU face paste (Wave 3 phase 2)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestGpuPasteWiring:
    def test_process_frames_batch_downloads_chunk_when_gpu_paste(self, mock_gpu):
        import torch
        from core.gpu_frame import ChunkFrameBuffer
        from processing.frame_batch import process_frames_batch

        frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        buf = ChunkFrameBuffer(frames, device_id=0)
        face = MagicMock()

        gpu_inst = MagicMock()
        gpu_inst.get_faces.return_value = [face]
        gpu_inst.swap_face_batch.return_value = torch.ones(
            (8, 8, 3), dtype=torch.uint8, device="cuda:0"
        ) * 42

        with patch("processing.frame_batch.config") as cfg:
            cfg.gpu_frame_retention_enabled = True
            cfg.gpu_paste_on_gpu = True
            cfg.workers_per_gpu = 1
            with patch(
                "processing.frame_batch.filter_faces_by_confidence",
                side_effect=lambda f, _: f,
            ):
                result = process_frames_batch(
                    frames,
                    [MagicMock()],
                    face_confidence=0.5,
                    gpu_instance=gpu_inst,
                    frame_buffer=buf,
                )

        assert len(result) == 1
        assert result[0].shape == (8, 8, 3)
        assert result[0][0, 0, 0] == 42
        gpu_inst.swap_face_batch.assert_called_once()
        call_kw = gpu_inst.swap_face_batch.call_args.kwargs
        assert call_kw.get("paste_on_gpu") is True
        assert call_kw.get("frame_gpu") is not None