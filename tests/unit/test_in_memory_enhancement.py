"""Tests for in-memory enhancement and compression skip."""

import numpy as np
from unittest.mock import patch

from processing.in_memory_enhancement import InMemoryEnhancer, resolve_enhancement_device_ids
from processing.restoration_session import RestorationSession
from utils.compression import should_skip_compression


class TestResolveEnhancementDevices:
    def test_hat_uses_single_gpu(self):
        with patch(
            "processing.in_memory_enhancement.select_enhancement_gpu", return_value=1
        ):
            assert resolve_enhancement_device_ids([0, 1], "HAT") == [1]

    def test_realesrgan_respects_multi_gpu_config(self, temp_config):
        with patch("processing.in_memory_enhancement.config") as mock_cfg, patch(
            "processing.in_memory_enhancement.select_enhancement_gpu", return_value=0
        ):
            mock_cfg.enhancement_multi_gpu_enabled = False
            assert resolve_enhancement_device_ids([0, 1], "RealESRGAN") == [0]


class TestInMemoryEnhancer:
    def test_face_enhance_off_by_default(self):
        enhancer = InMemoryEnhancer("RealESRGAN", "RealESRGAN_x4plus", [0])
        assert enhancer.face_enhance is False

    def test_multi_gpu_splits_frames(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]
        enhancer = InMemoryEnhancer("RealESRGAN", "RealESRGAN_x4plus", [0, 1])

        with patch.object(enhancer, "_enhance_bgr_batch", side_effect=lambda bgr, gid: bgr):
            result = enhancer.enhance_rgb_frames(frames)

        assert len(result) == 4

    def test_hat_preloads_single_gpu_when_multi_available(self):
        with patch(
            "processing.in_memory_enhancement.select_enhancement_gpu", return_value=1
        ), patch("processing.hat_enhancement.preload_hat_models") as mock_preload:
            InMemoryEnhancer("HAT", "HAT_Base_4x_ImageNet", [0, 1])
        mock_preload.assert_called_once_with([1], "HAT_Base_4x_ImageNet")

    def test_hat_never_shards_frames_across_gpus(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]
        with patch(
            "processing.in_memory_enhancement.select_enhancement_gpu", return_value=0
        ), patch.object(
            InMemoryEnhancer, "_enhance_bgr_batch", side_effect=lambda bgr, gid: bgr
        ) as mock_batch:
            enhancer = InMemoryEnhancer("HAT", "HAT_Base_4x_ImageNet", [0, 1])
            result = enhancer.enhance_rgb_frames(frames)

        assert len(result) == 4
        assert enhancer.device_ids == [0]
        mock_batch.assert_called_once()
        assert mock_batch.call_args[0][1] == 0


class TestCompressionSkip:
    def test_skip_small_image(self, tmp_path):
        img = tmp_path / "small.png"
        img.write_bytes(b"x" * 100)
        assert should_skip_compression(str(img), "image") is True