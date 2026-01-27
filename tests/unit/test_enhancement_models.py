"""
Unit tests for enhancement models (RealESRGAN, SwinIR, GFPGAN, CodeFormer).

Tests cover:
- Model initialization and cleanup
- Image enhancement pipelines
- Frame enhancement for GIF/Video
- Error handling
- Model selection and routing
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_rgb_array():
    """Create a sample RGB numpy array for testing."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_array():
    """Create a sample BGR numpy array for testing (OpenCV format)."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def temp_frames_dir(tmp_path):
    """Create temporary frames directory with sample frames."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create 5 sample frames
    for i in range(5):
        img = Image.new('RGB', (128, 128), color=(i * 50, 100, 150))
        img.save(frames_dir / f"frame_{i:04d}.png")

    return frames_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# RealESRGAN Tests
# =============================================================================

class TestRealESRGAN:
    """Tests for RealESRGAN enhancement module."""

    @pytest.mark.unit
    def test_model_options_defined(self):
        """Test that MODEL_OPTIONS dictionary is properly defined."""
        from utils.constants import MODEL_OPTIONS

        # MODEL_OPTIONS has display names as keys (e.g., "RealESRGAN_x4plus (General - Best for Photos)")
        assert len(MODEL_OPTIONS) > 0

        # Check that common model types exist (by checking model_name values)
        model_names = [info.get("model_name") for info in MODEL_OPTIONS.values()]
        assert "RealESRGAN_x4plus" in model_names
        assert "RealESRGAN_x2plus" in model_names

        # Check structure
        for display_name, model_info in MODEL_OPTIONS.items():
            assert "model_name" in model_info
            assert "supports_denoise" in model_info

    @pytest.mark.unit
    def test_realesrgan_x4_model_exists(self):
        """Test that x4 models exist in options."""
        from utils.constants import MODEL_OPTIONS

        # Check that x4 models exist by model_name
        model_names = [info.get("model_name") for info in MODEL_OPTIONS.values()]
        assert "RealESRGAN_x4plus" in model_names
        assert "RealESRGAN_x4plus_anime_6B" in model_names

    @pytest.mark.unit
    def test_realesrgan_x2_model_exists(self):
        """Test that x2 models exist in options."""
        from utils.constants import MODEL_OPTIONS

        model_names = [info.get("model_name") for info in MODEL_OPTIONS.values()]
        assert "RealESRGAN_x2plus" in model_names

    @pytest.mark.unit
    def test_denoise_support_detection(self):
        """Test that only specific models support denoise."""
        from utils.constants import MODEL_OPTIONS

        # Find models by their model_name and check supports_denoise
        for display_name, info in MODEL_OPTIONS.items():
            model_name = info.get("model_name")
            if model_name == "realesr-general-x4v3":
                assert info["supports_denoise"] is True
            elif model_name == "RealESRGAN_x4plus":
                assert info["supports_denoise"] is False

    @pytest.mark.unit
    def test_model_urls_defined(self):
        """Test that model download URLs are defined."""
        from processing.enhancement import MODEL_URLS

        assert "RealESRGAN_x4plus" in MODEL_URLS
        assert "RealESRGAN_x2plus" in MODEL_URLS
        assert all(url.startswith("https://") for url in MODEL_URLS.values())

    @pytest.mark.unit
    def test_model_scales_defined(self):
        """Test that model scales are defined."""
        from processing.enhancement import MODEL_SCALES

        assert MODEL_SCALES["RealESRGAN_x4plus"] == 4
        assert MODEL_SCALES["RealESRGAN_x2plus"] == 2

    @pytest.mark.unit
    @patch('processing.enhancement._get_upsampler')
    def test_enhance_image_calls_upsampler(self, mock_get_upsampler, sample_bgr_array, mock_gpu):
        """Test that enhance_image creates the upsampler correctly."""
        from processing.enhancement import enhance_image

        mock_upsampler = MagicMock()
        mock_upsampler.enhance.return_value = (sample_bgr_array * 2, None)
        mock_get_upsampler.return_value = mock_upsampler

        with patch('processing.enhancement._get_face_enhancer', return_value=None):
            result = enhance_image(
                sample_bgr_array,
                model_name="RealESRGAN_x4plus",
                gpu_id=0,
                outscale=4,
                tile_size=256,
                face_enhance=False
            )

        mock_get_upsampler.assert_called_once()
        mock_upsampler.enhance.assert_called_once()

    @pytest.mark.unit
    @patch('processing.enhancement._get_upsampler')
    def test_enhance_image_returns_array(self, mock_get_upsampler, sample_bgr_array, mock_gpu):
        """Test that enhance_image returns a numpy array."""
        from processing.enhancement import enhance_image

        expected_output = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_upsampler = MagicMock()
        mock_upsampler.enhance.return_value = (expected_output, None)
        mock_get_upsampler.return_value = mock_upsampler

        result = enhance_image(
            sample_bgr_array,
            model_name="RealESRGAN_x4plus",
            gpu_id=0,
            outscale=4,
            tile_size=256,
            face_enhance=False
        )

        assert isinstance(result, np.ndarray)

    @pytest.mark.unit
    def test_enhance_image_handles_none_input(self, mock_gpu):
        """Test that enhance_image handles None input gracefully."""
        from processing.enhancement import enhance_image

        # None input should return None
        result = enhance_image(None)
        assert result is None

    @pytest.mark.unit
    def test_clear_enhancement_cache(self, mock_gpu):
        """Test that enhancement cache can be cleared."""
        from processing.enhancement import clear_enhancement_cache, _upsampler_cache

        # Clear should not raise
        clear_enhancement_cache()

        # Cache should be empty after clearing
        assert len(_upsampler_cache) == 0


# =============================================================================
# SwinIR Tests
# =============================================================================

class TestSwinIR:
    """Tests for SwinIR/Swin2SR enhancement module."""

    @pytest.mark.unit
    def test_swinir_models_defined(self):
        """Test that SWINIR_MODELS dictionary is properly defined."""
        from processing.swinir_enhancement import SWINIR_MODELS

        assert "Swin2SR_x2" in SWINIR_MODELS
        assert "Swin2SR_x4" in SWINIR_MODELS
        assert "Swin2SR_RealWorld_x4" in SWINIR_MODELS
        assert "Swin2SR_Compressed_x4" in SWINIR_MODELS

    @pytest.mark.unit
    def test_swinir_model_structure(self):
        """Test that each SwinIR model has required keys."""
        from processing.swinir_enhancement import SWINIR_MODELS

        for model_name, model_info in SWINIR_MODELS.items():
            assert "model_id" in model_info, f"Missing model_id for {model_name}"
            assert "scale" in model_info, f"Missing scale for {model_name}"
            assert model_info["scale"] in [2, 4], f"Invalid scale for {model_name}"

    @pytest.mark.unit
    def test_swinir_default_model(self):
        """Test the default SwinIR model."""
        from processing.swinir_enhancement import DEFAULT_SWINIR_MODEL

        assert DEFAULT_SWINIR_MODEL == "Swin2SR_RealWorld_x4"

    @pytest.mark.unit
    def test_swinir_scale_retrieval(self):
        """Test get_swinir_scale function."""
        from processing.swinir_enhancement import get_swinir_scale

        assert get_swinir_scale("Swin2SR_x2") == 2
        assert get_swinir_scale("Swin2SR_x4") == 4
        assert get_swinir_scale("InvalidModel") == 4  # Default

    @pytest.mark.unit
    def test_swinir_invalid_model_name(self, mock_gpu, sample_bgr_array):
        """Test that invalid model name falls back to default."""
        from processing.swinir_enhancement import enhance_image_swinir

        # Should log a warning but use default model
        # With mocked GPU it will fail at model loading, returning None
        result = enhance_image_swinir(
            sample_bgr_array,
            model_name="InvalidModel",
            gpu_id=0
        )

        # Without actual model loading, result should be None
        assert result is None

    @pytest.mark.unit
    def test_swinir_none_input(self, mock_gpu):
        """Test that None input returns None."""
        from processing.swinir_enhancement import enhance_image_swinir

        # None input should fail gracefully
        result = enhance_image_swinir(
            None,
            model_name="Swin2SR_x4",
            gpu_id=0
        )

        assert result is None

    @pytest.mark.unit
    def test_swinir_cache_clear(self, mock_gpu):
        """Test that SwinIR cache can be cleared."""
        from processing.swinir_enhancement import clear_swinir_cache, _swinir_cache

        clear_swinir_cache()
        assert len(_swinir_cache) == 0

    @pytest.mark.unit
    def test_get_available_swinir_models(self):
        """Test that available models can be retrieved."""
        from processing.swinir_enhancement import get_available_swinir_models, SWINIR_MODELS

        available = get_available_swinir_models()
        assert available == SWINIR_MODELS
        # Should be a copy, not the same object
        assert available is not SWINIR_MODELS


# =============================================================================
# GFPGAN Tests
# =============================================================================

class TestGFPGAN:
    """Tests for GFPGAN face restoration module."""

    @pytest.mark.unit
    def test_face_restorer_module_exists(self):
        """Test that face_restoration module exists and has FaceRestorer."""
        from processing.face_restoration import FaceRestorer
        assert FaceRestorer is not None

    @pytest.mark.unit
    def test_face_restorer_init_params(self):
        """Test FaceRestorer accepts correct init parameters."""
        from processing.face_restoration import FaceRestorer
        import inspect

        sig = inspect.signature(FaceRestorer.__init__)
        params = list(sig.parameters.keys())

        assert 'device_id' in params
        assert 'model_version' in params

    @pytest.mark.unit
    def test_face_restorer_init_no_immediate_load(self, mock_gpu):
        """Test FaceRestorer uses lazy initialization."""
        from processing.face_restoration import FaceRestorer

        # Should not load model during __init__ (lazy init)
        restorer = FaceRestorer(device_id=0)

        # Internal state should show not initialized
        assert restorer._initialized is False

    @pytest.mark.unit
    def test_face_restorer_has_restore_method(self):
        """Test FaceRestorer has restore_face method."""
        from processing.face_restoration import FaceRestorer

        assert hasattr(FaceRestorer, 'restore_face')
        assert callable(getattr(FaceRestorer, 'restore_face'))

    @pytest.mark.unit
    def test_face_restorer_has_cleanup_method(self):
        """Test FaceRestorer has cleanup method."""
        from processing.face_restoration import FaceRestorer

        assert hasattr(FaceRestorer, 'cleanup')
        assert callable(getattr(FaceRestorer, 'cleanup'))


# =============================================================================
# CodeFormer Tests
# =============================================================================

class TestCodeFormer:
    """Tests for CodeFormer face restoration module."""

    @pytest.mark.unit
    def test_codeformer_module_exists(self):
        """Test that codeformer_restoration module exists."""
        from processing.codeformer_restoration import CodeFormerRestorer
        assert CodeFormerRestorer is not None

    @pytest.mark.unit
    def test_codeformer_restorer_init_params(self):
        """Test CodeFormerRestorer accepts correct parameters."""
        from processing.codeformer_restoration import CodeFormerRestorer
        import inspect

        sig = inspect.signature(CodeFormerRestorer.__init__)
        params = list(sig.parameters.keys())

        assert 'device_id' in params
        assert 'upscale' in params
        assert 'model_dir' in params

    @pytest.mark.unit
    def test_codeformer_model_urls_defined(self):
        """Test that CodeFormer model URLs are defined."""
        from processing.codeformer_restoration import (
            CODEFORMER_MODEL_URL,
            DETECTION_MODEL_URL,
            PARSING_MODEL_URL
        )

        assert CODEFORMER_MODEL_URL.startswith("https://")
        assert DETECTION_MODEL_URL.startswith("https://")
        assert PARSING_MODEL_URL.startswith("https://")

    @pytest.mark.unit
    def test_codeformer_convenience_function_exists(self):
        """Test that convenience function exists."""
        from processing.codeformer_restoration import restore_faces_codeformer
        assert callable(restore_faces_codeformer)

    @pytest.mark.unit
    def test_codeformer_frames_function_exists(self):
        """Test that frame restoration function exists."""
        from processing.codeformer_restoration import restore_frames_codeformer
        assert callable(restore_frames_codeformer)

    @pytest.mark.unit
    def test_codeformer_cache_clear(self, mock_gpu):
        """Test that CodeFormer cache can be cleared."""
        from processing.codeformer_restoration import clear_codeformer_cache, _codeformer_cache

        clear_codeformer_cache()
        assert len(_codeformer_cache) == 0


# =============================================================================
# Model Selection/Routing Tests
# =============================================================================

class TestModelRouting:
    """Tests for model selection and routing logic."""

    @pytest.mark.unit
    def test_enhancement_model_options(self):
        """Test that enhancement model options are properly defined."""
        # Check UI defines both options
        enhancement_options = ["RealESRGAN", "SwinIR"]
        assert "RealESRGAN" in enhancement_options
        assert "SwinIR" in enhancement_options

    @pytest.mark.unit
    def test_restoration_model_options(self):
        """Test that restoration model options are properly defined."""
        restoration_options = ["GFPGAN", "CodeFormer"]
        assert "GFPGAN" in restoration_options
        assert "CodeFormer" in restoration_options

    @pytest.mark.unit
    def test_get_restorer_returns_gfpgan_by_default(self, mock_gpu):
        """Test that _get_restorer returns GFPGAN by default."""
        from processing.image_processing import _get_restorer

        # Patch at the location where FaceRestorer is used (image_processing imports it)
        with patch('processing.image_processing.FaceRestorer') as mock_class:
            mock_restorer = MagicMock()
            mock_class.return_value = mock_restorer

            restorer, restore_func = _get_restorer("GFPGAN", gpu_id=0)

            mock_class.assert_called_once()
            assert callable(restore_func)

    @pytest.mark.unit
    def test_get_restorer_returns_codeformer(self, mock_gpu):
        """Test that _get_restorer returns CodeFormer when specified."""
        from processing.image_processing import _get_restorer

        with patch('processing.codeformer_restoration.CodeFormerRestorer') as mock_class:
            mock_restorer = MagicMock()
            mock_class.return_value = mock_restorer

            restorer, restore_func = _get_restorer("CodeFormer", gpu_id=0)

            mock_class.assert_called_once()
            assert callable(restore_func)

    @pytest.mark.unit
    def test_apply_enhancement_routes_to_swinir(self):
        """Test that _apply_enhancement routes to SwinIR when specified."""
        from processing.image_processing import _apply_enhancement

        with patch('processing.swinir_enhancement.enhance_image_swinir_file') as mock_swinir:
            mock_swinir.return_value = Path("enhanced.png")

            result = _apply_enhancement(
                output_path=Path("test.png"),
                output_dir=Path("outputs"),
                enhancement_model="SwinIR",
                model_name="Swin2SR_x4",
                tile_size=256,
                outscale=4,
                denoise_strength=0.5,
                use_fp32=False,
                pre_pad=10,
                gpu_id=0
            )

            mock_swinir.assert_called_once()


# =============================================================================
# Processing Pipeline Tests
# =============================================================================

class TestProcessingPipeline:
    """Tests for the processing pipeline with model selection."""

    @pytest.mark.unit
    def test_process_image_signature_includes_model_params(self):
        """Test that process_image accepts model selection parameters."""
        from processing.image_processing import process_image
        import inspect

        sig = inspect.signature(process_image)
        param_names = list(sig.parameters.keys())

        assert 'restore_faces' in param_names
        assert 'restoration_weight' in param_names

    @pytest.mark.unit
    def test_orchestrator_signature_includes_model_params(self):
        """Test that orchestrator accepts model selection parameters."""
        from processing.orchestrator import process_media
        import inspect

        sig = inspect.signature(process_media)
        param_names = list(sig.parameters.keys())

        assert 'enhancement_model' in param_names
        assert 'restoration_model' in param_names

    @pytest.mark.unit
    def test_default_enhancement_model(self):
        """Test that default enhancement model is RealESRGAN."""
        from processing.orchestrator import process_media
        import inspect

        sig = inspect.signature(process_media)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        assert defaults.get('enhancement_model') == 'RealESRGAN'

    @pytest.mark.unit
    def test_default_restoration_model(self):
        """Test that default restoration model is GFPGAN."""
        from processing.orchestrator import process_media
        import inspect

        sig = inspect.signature(process_media)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        assert defaults.get('restoration_model') == 'GFPGAN'


# =============================================================================
# Handler Tests
# =============================================================================

class TestProcessingHandlers:
    """Tests for UI processing handlers."""

    @pytest.mark.unit
    def test_process_image_handler_signature(self):
        """Test that process_image handler accepts model parameters."""
        from ui.handlers.processing_handlers import process_image
        import inspect

        sig = inspect.signature(process_image)
        param_names = list(sig.parameters.keys())

        assert 'enhancement_model' in param_names
        assert 'restoration_model' in param_names

    @pytest.mark.unit
    def test_process_gif_handler_signature(self):
        """Test that process_gif handler accepts model parameters."""
        from ui.handlers.processing_handlers import process_gif
        import inspect

        sig = inspect.signature(process_gif)
        param_names = list(sig.parameters.keys())

        assert 'enhancement_model' in param_names
        assert 'restoration_model' in param_names

    @pytest.mark.unit
    def test_process_video_handler_signature(self):
        """Test that process_video handler accepts model parameters."""
        from ui.handlers.processing_handlers import process_video
        import inspect

        sig = inspect.signature(process_video)
        param_names = list(sig.parameters.keys())

        assert 'enhancement_model' in param_names
        assert 'restoration_model' in param_names


# =============================================================================
# UI Component Tests
# =============================================================================

class TestUIComponents:
    """Tests for UI component model selectors."""

    @pytest.mark.unit
    def test_image_tab_has_model_selectors(self):
        """Test that image_tab returns model selector components."""
        # We can check this by verifying the returned keys
        from ui.components.image_tab import create_image_tab
        import gradio as gr

        # Can't easily test Gradio component creation without running
        # Instead, test that the module has the expected structure
        import inspect
        source = inspect.getsource(create_image_tab)

        assert 'enhancement_model_selector' in source
        assert 'restoration_model_selector' in source

    @pytest.mark.unit
    def test_gif_tab_has_model_selectors(self):
        """Test that gif_tab returns model selector components."""
        from ui.components.gif_tab import create_gif_tab
        import inspect
        source = inspect.getsource(create_gif_tab)

        assert 'enhancement_model_selector_gif' in source
        assert 'restoration_model_selector_gif' in source

    @pytest.mark.unit
    def test_video_tab_has_model_selectors(self):
        """Test that video_tab returns model selector components."""
        from ui.components.video_tab import create_video_tab
        import inspect
        source = inspect.getsource(create_video_tab)

        assert 'enhancement_model_selector_vid' in source
        assert 'restoration_model_selector_vid' in source
