"""Tests for automatic workload profiling (Wave 4)."""

import pytest
from unittest.mock import patch


class TestWorkloadProfile:
    def test_swap_only_video_profile(self, temp_config, reset_config):
        from processing.workload_profile import resolve_workload_profile

        with patch("processing.workload_profile._cuda_ready", return_value=True):
            with patch(
                "utils.gpu_decode.nvcodec_decode_available", return_value=False
            ):
                profile = resolve_workload_profile(
                    media_type="video",
                    enhance=False,
                    enhancement_model="RealESRGAN",
                    restore_faces=False,
                    face_mappings=None,
                    width=1920,
                    height=1080,
                )

        assert profile is not None
        assert profile.name == "stream_swap_only"
        assert profile.paste_on_gpu is True
        assert profile.zero_copy is True
        assert profile.enhancement_chain is False
        assert profile.defer_download is False
        assert profile.chunk_size is not None

    def test_hat_enhance_chain_profile(self, temp_config, reset_config):
        from processing.workload_profile import resolve_workload_profile

        with patch("processing.workload_profile._cuda_ready", return_value=True):
            with patch(
                "utils.gpu_decode.nvcodec_decode_available", return_value=False
            ):
                profile = resolve_workload_profile(
                    media_type="video",
                    enhance=True,
                    enhancement_model="HAT",
                    restore_faces=False,
                    face_mappings=None,
                    width=1920,
                    height=1080,
                    outscale=4,
                )

        assert profile.name == "stream_hat_chain"
        assert profile.enhancement_chain is True
        assert profile.defer_download is True

    def test_face_mappings_disable_paste(self, temp_config, reset_config):
        from processing.workload_profile import resolve_workload_profile

        with patch("processing.workload_profile._cuda_ready", return_value=True):
            with patch(
                "utils.gpu_decode.nvcodec_decode_available", return_value=False
            ):
                profile = resolve_workload_profile(
                    media_type="video",
                    enhance=False,
                    enhancement_model="RealESRGAN",
                    restore_faces=False,
                    face_mappings=[(0, 0)],
                    width=1280,
                    height=720,
                )

        assert profile.paste_on_gpu is False

    def test_video_profile_enables_nvcodec_when_available(
        self, temp_config, reset_config
    ):
        from processing.workload_profile import resolve_workload_profile

        with patch("processing.workload_profile._cuda_ready", return_value=True):
            with patch(
                "utils.gpu_decode.nvcodec_decode_available", return_value=True
            ):
                profile = resolve_workload_profile(
                    media_type="video",
                    enhance=False,
                    enhancement_model="RealESRGAN",
                    restore_faces=False,
                    face_mappings=None,
                    width=1920,
                    height=1080,
                )

        assert profile is not None
        assert profile.use_nvcodec_decode is True

    def test_gif_profile_never_uses_nvcodec(self, temp_config, reset_config):
        from processing.workload_profile import resolve_workload_profile

        with patch("processing.workload_profile._cuda_ready", return_value=True):
            with patch(
                "utils.gpu_decode.nvcodec_decode_available", return_value=True
            ):
                profile = resolve_workload_profile(
                    media_type="gif",
                    enhance=False,
                    enhancement_model="RealESRGAN",
                    restore_faces=False,
                    face_mappings=None,
                    width=640,
                    height=480,
                )

        assert profile is not None
        assert profile.use_nvcodec_decode is False

    def test_auto_tune_disabled_returns_none(self, temp_config, reset_config):
        from utils.config_manager import config
        from processing.workload_profile import resolve_workload_profile

        config._config["gpu"]["auto_workload_tune"] = False
        profile = resolve_workload_profile(
            media_type="video",
            enhance=False,
            enhancement_model="RealESRGAN",
            restore_faces=False,
            face_mappings=None,
        )
        assert profile is None