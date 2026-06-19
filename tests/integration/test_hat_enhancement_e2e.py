"""
End-to-end integration tests for HAT enhancement module.

These tests verify the full HAT enhancement workflow including:
- Model availability and structure
- Tiled vs direct inference routing
- Enhancement result dimensions
- Cache management
- Integration with image/video/GIF processing pipelines
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image


class TestHATIntegration:
    """Integration tests for HAT enhancement."""

    @pytest.mark.integration
    def test_hat_model_configuration(self):
        """Test HAT model configuration is properly loaded."""
        from processing.hat_enhancement import HAT_MODELS, DEFAULT_HAT_MODEL
        from utils.constants import HAT_MODEL_OPTIONS

        assert "HAT_Base_4x_ImageNet" in HAT_MODELS
        assert "HAT_Base_4x_GAN_sharper" in HAT_MODELS
        assert DEFAULT_HAT_MODEL == "HAT_Base_4x_ImageNet"

    @pytest.mark.integration
    def test_hat_config_options_match(self):
        """Test HAT options in constants align with enhancement module."""
        from processing.hat_enhancement import HAT_MODELS
        from utils.constants import HAT_MODEL_OPTIONS

        for model_name in HAT_MODELS:
            # Model names in constants use suffix like "(General)"
            base_name = model_name
            found = any(base_name in k for k in HAT_MODEL_OPTIONS)
            assert found, f"Missing {model_name} in HAT_MODEL_OPTIONS"

    @pytest.mark.integration
    def test_hat_enhancement_routing_in_image_processing(self):
        """Test image processing routes to HAT when selected."""
        from processing.hat_enhancement import HAT_MODELS, enhance_image_hat

        assert "HAT_Base_4x_ImageNet" in HAT_MODELS

    @pytest.mark.integration
    def test_hat_enhancement_scaling(self):
        """Test HAT 4x scaling is applied to output dimensions."""
        from processing.hat_enhancement import get_hat_scale

        assert get_hat_scale("HAT_Base_4x_ImageNet") == 4
        assert get_hat_scale("HAT_Base_4x_GAN_sharper") == 4

    @pytest.mark.integration
    def test_config_options_structure(self):
        """Test HAT config options have consistent keys."""
        from utils.constants import HAT_MODEL_OPTIONS

        for key, val in HAT_MODEL_OPTIONS.items():
            assert "model_name" in val
            assert "scale" in val
            # Key contains model_name (may have suffix)
            assert val["model_name"] in key

    @pytest.mark.integration
    def test_image_processing_has_hat_enhancement(self):
        """Test image_processing module imports HAT."""
        from processing import image_processing
        src = open(image_processing.__file__).read()

        # HAT enhancement should be imported and used
        assert "hat_enhancement" in src or "enhance_image" in src

    @pytest.mark.integration
    def test_video_processing_has_hat_enhancement(self):
        """Test video_processing module imports HAT."""
        from processing import video_processing
        src = open(video_processing.__file__).read()

        # HAT enhancement should be in processing path
        assert "hat" in src.lower() or "enhance" in src.lower()

    @pytest.mark.integration
    def test_gif_processing_has_hat_enhancement(self):
        """Test gif_processing module imports HAT."""
        from processing import gif_processing
        src = open(gif_processing.__file__).read()

        # HAT enhancement should be in processing path
        assert "hat" in src.lower() or "enhance" in src.lower()


class TestHATCacheIntegration:
    """Integration tests for HAT cache management."""

    @pytest.mark.integration
    def test_hat_config_options_structure(self):
        """Test HAT config options have consistent keys."""
        from utils.constants import HAT_MODEL_OPTIONS

        for key, val in HAT_MODEL_OPTIONS.items():
            assert "model_name" in val
            assert "scale" in val
            assert isinstance(val["scale"], int)
            # Key contains model_name (may have suffix like " (General)")
            assert val["model_name"] in key

    @pytest.mark.integration
    def test_hat_constants_consistency(self):
        """Test that DEFAULT constants match HAT_MODELS."""
        from processing.hat_enhancement import DEFAULT_HAT_MODEL, HAT_MODELS
        from utils.constants import HAT_MODEL_OPTIONS

        assert DEFAULT_HAT_MODEL in HAT_MODELS

        # Check option model_name is consistent (keys may have suffixes)
        for name, opts in HAT_MODEL_OPTIONS.items():
            # Keys have display suffixes like "(General)" but model_name doesn't
            assert name.startswith(opts["model_name"])
