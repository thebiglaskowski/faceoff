"""Tests for Wave 3 phase 3 GPU detection."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestGpuDetection:
    def test_downscale_frame_gpu(self, mock_gpu):
        import torch
        from core.gpu_detection import downscale_frame_gpu

        frame = torch.zeros((200, 400, 3), dtype=torch.uint8, device="cuda:0")
        scaled, factor = downscale_frame_gpu(frame, scale=0.5, min_resolution=64)
        assert scaled.shape[0] == 100
        assert scaled.shape[1] == 200
        assert factor == pytest.approx(0.5, rel=0.05)

    def test_detect_faces_from_gpu_uses_instance(self, mock_gpu):
        import torch
        from core.gpu_detection import detect_faces_from_gpu

        gpu_inst = MagicMock()
        face = MagicMock()
        gpu_inst.get_faces.return_value = [face]
        frame = torch.zeros((64, 64, 3), dtype=torch.uint8, device="cuda:0")

        faces = detect_faces_from_gpu(gpu_inst, frame, adaptive_processor=None)
        assert faces == [face]
        gpu_inst.get_faces.assert_called_once()