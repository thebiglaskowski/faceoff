"""Tests for MediaProcessor pool re-binding after VRAM release."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_model_pool():
    from core import model_pool as mp_module

    original = mp_module.ModelPool._instance
    mp_module.ModelPool._instance = None
    yield
    mp_module.ModelPool._instance = original


class TestMediaProcessorRebind:
    def test_get_faces_rebinds_after_release(self, tmp_path):
        """UI/processing must recover when pool cleanup deleted face_app."""
        inswapper = tmp_path / "inswapper.onnx"
        buffalo = tmp_path / "buffalo"
        inswapper.write_text("x")
        buffalo.mkdir()

        def _fresh_gpu():
            gpu = MagicMock()
            gpu.models_ready.return_value = True
            gpu.face_app = MagicMock()
            gpu.face_app.get.return_value = []
            gpu.swapper = MagicMock()
            return gpu

        mock_gpu = _fresh_gpu()
        mock_gpu_rebound = _fresh_gpu()

        with patch("core.media_processor.config") as mock_config, patch(
            "core.media_processor.get_model_pool"
        ) as mock_get_pool, patch(
            "core.media_processor.is_tensorrt_runtime_available", return_value=False
        ):
            mock_config.inswapper_model_path = str(inswapper)
            mock_config.buffalo_model_path = str(buffalo)

            mock_pool = MagicMock()
            mock_pool.get_instance.side_effect = [mock_gpu, mock_gpu_rebound]
            mock_get_pool.return_value = mock_pool

            from core.media_processor import MediaProcessor

            processor = MediaProcessor(device_id=0, use_tensorrt=False)
            processor.get_faces(np.zeros((64, 64, 3), dtype=np.uint8))
            assert mock_pool.get_instance.call_count == 1

            processor.release_gpu_models()
            mock_gpu.models_ready.return_value = False

            processor.get_faces(np.zeros((64, 64, 3), dtype=np.uint8))
            assert mock_pool.get_instance.call_count == 2
            assert processor._gpu is mock_gpu_rebound