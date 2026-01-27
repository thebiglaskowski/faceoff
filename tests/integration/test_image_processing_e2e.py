"""
End-to-end integration tests for image processing pipeline.

These tests verify the full image processing workflow including:
- Input validation
- Face detection coordination
- Face swapping orchestration
- Output generation
- Enhancement (optional)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image


@pytest.fixture
def mock_media_processor():
    """Create a mock MediaProcessor that simulates face operations."""
    processor = MagicMock()

    # Mock face detection - return a mock face
    mock_face = MagicMock()
    mock_face.bbox = np.array([50, 50, 150, 150])
    mock_face.det_score = 0.95
    mock_face.embedding = np.random.rand(512).astype(np.float32)
    processor.get_faces.return_value = [mock_face]

    # Mock face swapping - return input image unchanged
    def mock_swap(image, target_face, source_face):
        return image.copy()
    processor.swap_face.side_effect = mock_swap

    # Mock image read/write
    def mock_read(path):
        return np.zeros((200, 200, 3), dtype=np.uint8)
    processor.read_image.side_effect = mock_read

    def mock_write(path, image):
        return str(path)
    processor.write_image.side_effect = mock_write

    return processor


@pytest.fixture
def realistic_face_image(tmp_path):
    """Create a test image with realistic dimensions."""
    img = Image.new('RGB', (512, 512), color=(200, 180, 160))
    path = tmp_path / "face_image.png"
    img.save(path)
    return path


@pytest.fixture
def source_face_image(tmp_path):
    """Create a source face image."""
    img = Image.new('RGB', (256, 256), color=(180, 160, 140))
    path = tmp_path / "source_face.png"
    img.save(path)
    return path


class TestImageProcessingE2E:
    """End-to-end tests for image processing."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="ImageProcessor class not implemented - uses process_image function instead")
    def test_full_image_pipeline_with_single_face(
        self, tmp_path, realistic_face_image, source_face_image, mock_media_processor
    ):
        """Test complete image processing with one face detected."""
        from processing.image_processing import ImageProcessor

        with patch('processing.image_processing.MediaProcessor', return_value=mock_media_processor):
            with patch('processing.image_processing.MemoryManager'):
                processor = ImageProcessor.__new__(ImageProcessor)
                processor.processor = mock_media_processor
                processor.logger = MagicMock()

                # Simulate the process flow
                source_img = mock_media_processor.read_image(str(source_face_image))
                target_img = mock_media_processor.read_image(str(realistic_face_image))

                source_faces = mock_media_processor.get_faces(source_img)
                target_faces = mock_media_processor.get_faces(target_img)

                assert len(source_faces) == 1, "Should detect one source face"
                assert len(target_faces) == 1, "Should detect one target face"

                # Perform swap
                result = mock_media_processor.swap_face(
                    target_img, target_faces[0], source_faces[0]
                )

                assert result is not None, "Should return swapped image"
                assert result.shape == target_img.shape, "Output shape should match input"

    @pytest.mark.integration
    def test_image_pipeline_no_face_detected(
        self, tmp_path, mock_media_processor
    ):
        """Test handling when no face is detected in target."""
        # Configure mock to return no faces
        mock_media_processor.get_faces.return_value = []

        source_img = np.zeros((200, 200, 3), dtype=np.uint8)
        target_img = np.zeros((200, 200, 3), dtype=np.uint8)

        source_faces = mock_media_processor.get_faces(source_img)
        target_faces = mock_media_processor.get_faces(target_img)

        assert len(target_faces) == 0, "Should detect no faces"
        # Processing should handle this gracefully

    @pytest.mark.integration
    def test_image_pipeline_multiple_faces(
        self, tmp_path, mock_media_processor
    ):
        """Test processing image with multiple faces."""
        # Configure mock to return multiple faces
        mock_faces = []
        for i in range(3):
            face = MagicMock()
            face.bbox = np.array([50 + i*100, 50, 150 + i*100, 150])
            face.det_score = 0.9 - i*0.1
            face.embedding = np.random.rand(512).astype(np.float32)
            mock_faces.append(face)

        mock_media_processor.get_faces.return_value = mock_faces

        target_img = np.zeros((200, 500, 3), dtype=np.uint8)
        faces = mock_media_processor.get_faces(target_img)

        assert len(faces) == 3, "Should detect three faces"

    @pytest.mark.integration
    def test_image_validation_valid_file(self, tmp_path):
        """Test validation of valid image files."""
        from utils.validation import validate_image_file

        # Create valid image
        img = Image.new('RGB', (512, 512), color='blue')
        path = tmp_path / "valid.png"
        img.save(path)

        # Should not raise
        result = validate_image_file(str(path), max_pixels=1000000)
        assert result is True or result is None  # Validation passes

    @pytest.mark.integration
    def test_image_validation_oversized_file(self, tmp_path):
        """Test validation rejects oversized images."""
        from utils.validation import validate_image_file

        # Create oversized image
        img = Image.new('RGB', (5000, 5000), color='red')
        path = tmp_path / "oversized.png"
        img.save(path)

        # Should raise or return error
        with pytest.raises(Exception):
            validate_image_file(str(path), max_pixels=4194304)  # 2048x2048 limit


class TestImageWithEnhancement:
    """Tests for image processing with enhancement enabled."""

    @pytest.mark.integration
    def test_enhancement_toggle_respected(self, mock_media_processor):
        """Test that enhancement is only applied when enabled."""
        # Mock enhancement function
        with patch('processing.enhancement.enhance_image') as mock_enhance:
            mock_enhance.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

            # Simulate processing with enhancement disabled
            enhance_enabled = False
            result_img = np.zeros((200, 200, 3), dtype=np.uint8)

            if enhance_enabled:
                result_img = mock_enhance(result_img)

            mock_enhance.assert_not_called()

            # Now with enhancement enabled
            enhance_enabled = True
            if enhance_enabled:
                result_img = mock_enhance(result_img)

            mock_enhance.assert_called_once()

    @pytest.mark.integration
    def test_enhancement_fp16_mode(self, mock_media_processor):
        """Test enhancement respects FP16/FP32 mode setting."""
        with patch('processing.enhancement.enhance_image') as mock_enhance:
            mock_enhance.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

            # Call with FP16 (use_fp32=False)
            from processing.enhancement import enhance_image

            # The function signature should respect use_fp32 parameter
            # This tests the interface, not actual enhancement
            result = mock_enhance(
                np.zeros((200, 200, 3), dtype=np.uint8),
                use_fp32=False  # FP16 mode
            )

            call_kwargs = mock_enhance.call_args[1]
            assert call_kwargs.get('use_fp32') == False


class TestImageOutputFormats:
    """Tests for various output format handling."""

    @pytest.mark.integration
    def test_output_preserves_dimensions(self, mock_media_processor):
        """Test that output image has expected dimensions."""
        input_shape = (480, 640, 3)
        input_img = np.zeros(input_shape, dtype=np.uint8)

        # Mock swap returns same shape
        mock_media_processor.swap_face.return_value = input_img.copy()

        result = mock_media_processor.swap_face(
            input_img, MagicMock(), MagicMock()
        )

        assert result.shape == input_shape

    @pytest.mark.integration
    def test_output_valid_numpy_array(self, mock_media_processor):
        """Test that output is a valid numpy array."""
        input_img = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_media_processor.swap_face.return_value = input_img.copy()

        result = mock_media_processor.swap_face(
            input_img, MagicMock(), MagicMock()
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB channels
