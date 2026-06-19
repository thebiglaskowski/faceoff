"""Tests for Wave 3 GPU frame retention."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestChunkFrameBuffer:
    def test_upload_creates_gpu_batch(self, mock_gpu):
        import torch
        from core.gpu_frame import ChunkFrameBuffer

        frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8) * 255,
        ]
        buf = ChunkFrameBuffer(frames, device_id=0)
        batch = buf.upload()
        assert batch.shape == (2, 8, 8, 3)
        assert batch.device.type == "cuda"
        assert buf.frame_numpy(0) is frames[0]

    def test_replace_from_numpy_clears_gpu_cache(self, mock_gpu):
        from core.gpu_frame import ChunkFrameBuffer

        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        buf = ChunkFrameBuffer(frames, device_id=0)
        buf.upload()
        new_frames = [np.ones((4, 4, 3), dtype=np.uint8)]
        buf.replace_from_numpy(new_frames)
        assert buf._gpu_batch is None
        assert buf.frames is new_frames

    def test_update_frame_gpu_and_download_all(self, mock_gpu):
        import torch
        from core.gpu_frame import ChunkFrameBuffer

        frames = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
        ]
        buf = ChunkFrameBuffer(frames, device_id=0)
        buf.upload()
        updated = torch.ones((4, 4, 3), dtype=torch.uint8, device="cuda:0") * 200
        buf.update_frame_gpu(1, updated)
        assert buf.has_gpu_batch()
        out = buf.download_all()
        assert len(out) == 2
        assert out[0].shape == (4, 4, 3)
        assert out[1][0, 0, 0] == 200


class TestSwapperIoBinding:
    def test_run_swapper_batch_falls_back_to_numpy(self, mock_gpu):
        from core.model_pool import GPUModelInstance

        swapper = MagicMock()
        swapper.input_names = ["target", "source"]
        swapper.output_names = ["output"]
        swapper.session.run.return_value = [np.zeros((1, 3, 128, 128), dtype=np.float32)]

        inst = GPUModelInstance.__new__(GPUModelInstance)
        inst.device_id = 0
        inst._lock = __import__("threading").Lock()
        inst.swapper = swapper

        blobs = np.zeros((1, 3, 128, 128), dtype=np.float32)
        src = np.zeros((1, 512), dtype=np.float32)
        out = inst.run_swapper_batch(blobs, src, use_iobinding=False)
        assert out.shape == (1, 3, 128, 128)
        swapper.session.run.assert_called_once()