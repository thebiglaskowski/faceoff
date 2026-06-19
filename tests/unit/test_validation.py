"""
Unit tests for utils/validation.py

Tests:
- validate_file_size
- validate_image_resolution
- validate_video_duration
- validate_gif_frames
- validate_media_type
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np


class TestValidateFileSize:
    """Tests for validate_file_size function."""

    def test_valid_file_size(self, tmp_path):
        """Should pass for files under limit."""
        from utils.validation import validate_file_size

        # Create small test file
        test_file = tmp_path / "small.txt"
        test_file.write_text("x" * 1000)  # ~1KB

        # Should not raise
        validate_file_size(str(test_file), max_size_mb=1)

    def test_file_over_limit(self, tmp_path):
        """Should raise for files over limit."""
        from utils.validation import validate_file_size

        # Create file that's > 1KB but test with 0MB limit
        test_file = tmp_path / "large.txt"
        test_file.write_bytes(b"x" * 2000)

        with pytest.raises(ValueError) as exc_info:
            validate_file_size(str(test_file), max_size_mb=0.001)

        assert "too large" in str(exc_info.value).lower()

    def test_file_exactly_at_limit(self, tmp_path):
        """Should pass for files exactly at limit."""
        from utils.validation import validate_file_size

        # Create ~1MB file
        test_file = tmp_path / "exact.bin"
        test_file.write_bytes(b"x" * (1024 * 1024))

        # Should pass at 1MB limit
        validate_file_size(str(test_file), max_size_mb=1)

    def test_uses_config_default(self, tmp_path, temp_config, reset_config):
        """Should use config default when not specified."""
        from utils.validation import validate_file_size

        test_file = tmp_path / "test.txt"
        test_file.write_text("small file")

        # temp_config has max_file_size_mb: 100
        # Small file should pass
        validate_file_size(str(test_file))


class TestValidateImageResolution:
    """Tests for validate_image_resolution function."""

    def test_valid_resolution(self, sample_image):
        """Should pass for images under limit."""
        from utils.validation import validate_image_resolution

        # sample_image is 200x200 = 40,000 pixels
        validate_image_resolution(str(sample_image))

    def test_resolution_over_limit(self, large_image):
        """Should raise for images over limit."""
        from utils.validation import validate_image_resolution

        with pytest.raises(ValueError) as exc_info:
            validate_image_resolution(str(large_image))

        assert "resolution" in str(exc_info.value).lower()

    def test_exactly_at_limit(self, tmp_path):
        """Should pass for images exactly at limit."""
        from utils.validation import validate_image_resolution
        from utils.constants import MAX_IMAGE_PIXELS

        # Create image exactly at limit (2048x2048 for 4M limit)
        side = int(MAX_IMAGE_PIXELS ** 0.5)
        img = Image.new('RGB', (side, side))
        img_path = tmp_path / "exact.png"
        img.save(img_path)

        validate_image_resolution(str(img_path))


class TestValidateVideoDuration:
    """Tests for validate_video_duration function."""

    def test_valid_duration(self, tmp_path):
        """Should pass for videos under limit."""
        from utils.validation import validate_video_duration

        with patch('utils.video_io.probe_video') as mock_probe:
            mock_probe.return_value = {'duration': 30}

            validate_video_duration(str(tmp_path / "test.mp4"))

    def test_duration_over_limit(self, tmp_path):
        """Should raise for videos over limit."""
        from utils.validation import validate_video_duration

        with patch('utils.video_io.probe_video') as mock_probe:
            mock_probe.return_value = {'duration': 600}

            with pytest.raises(ValueError) as exc_info:
                validate_video_duration(str(tmp_path / "long.mp4"))

            assert "too long" in str(exc_info.value).lower()

    def test_invalid_video_file(self, tmp_path):
        """Should raise for invalid video files."""
        from utils.validation import validate_video_duration

        with patch('utils.video_io.probe_video') as mock_probe:
            mock_probe.side_effect = Exception("Cannot open file")

            with pytest.raises(ValueError) as exc_info:
                validate_video_duration(str(tmp_path / "invalid.mp4"))

            assert "invalid" in str(exc_info.value).lower()


class TestValidateGifFrames:
    """Tests for validate_gif_frames function."""

    def test_valid_frame_count(self, sample_gif):
        """Should pass for GIFs under frame limit."""
        from utils.validation import validate_gif_frames

        # sample_gif has 5 frames
        validate_gif_frames(str(sample_gif))

    def test_too_many_frames(self, tmp_path):
        """Should raise for GIFs with too many frames."""
        from utils.validation import validate_gif_frames
        from utils.constants import MAX_GIF_FRAMES

        # Create GIF with many frames
        frames = []
        for i in range(MAX_GIF_FRAMES + 10):
            img = Image.new('RGB', (10, 10), color=(i % 255, 0, 0))
            frames.append(img)

        gif_path = tmp_path / "many_frames.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0
        )

        with pytest.raises(ValueError) as exc_info:
            validate_gif_frames(str(gif_path))

        assert "too many frames" in str(exc_info.value).lower()

    def test_invalid_gif_file(self, tmp_path):
        """Should raise for invalid GIF files."""
        from utils.validation import validate_gif_frames

        # Create a non-GIF file
        not_gif = tmp_path / "fake.gif"
        not_gif.write_text("not a gif")

        with pytest.raises(ValueError) as exc_info:
            validate_gif_frames(str(not_gif))

        assert "invalid" in str(exc_info.value).lower()


class TestValidateMediaType:
    """Tests for validate_media_type function."""

    def test_detect_image(self, sample_image):
        """Should detect image media type."""
        from utils.validation import validate_media_type

        with patch('utils.validation.magic') as mock_magic:
            mock_magic.from_file.return_value = "image/png"

            result = validate_media_type(str(sample_image))
            assert result == "image"

    def test_detect_gif(self, sample_gif):
        """Should detect GIF media type."""
        from utils.validation import validate_media_type

        with patch('utils.validation.magic') as mock_magic:
            mock_magic.from_file.return_value = "image/gif"

            result = validate_media_type(str(sample_gif))
            assert result == "gif"

    def test_detect_video(self, tmp_path):
        """Should detect video media type."""
        from utils.validation import validate_media_type

        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch('utils.validation.magic') as mock_magic:
            mock_magic.from_file.return_value = "video/mp4"

            result = validate_media_type(str(video_file))
            assert result == "video"

    def test_unsupported_type(self, tmp_path):
        """Should raise for unsupported media types."""
        from utils.validation import validate_media_type

        text_file = tmp_path / "test.txt"
        text_file.write_text("hello")

        with patch('utils.validation.magic') as mock_magic:
            mock_magic.from_file.return_value = "text/plain"

            with pytest.raises(ValueError) as exc_info:
                validate_media_type(str(text_file))

            assert "unsupported" in str(exc_info.value).lower()

    def test_various_image_formats(self, tmp_path):
        """Should detect various image formats."""
        from utils.validation import validate_media_type

        formats = [
            ("test.jpg", "image/jpeg"),
            ("test.png", "image/png"),
            ("test.bmp", "image/bmp"),
            ("test.webp", "image/webp"),
        ]

        for filename, mime_type in formats:
            test_file = tmp_path / filename
            test_file.touch()

            with patch('utils.validation.magic') as mock_magic:
                mock_magic.from_file.return_value = mime_type

                result = validate_media_type(str(test_file))
                assert result == "image", f"Failed for {filename}"

    def test_various_video_formats(self, tmp_path):
        """Should detect various video formats."""
        from utils.validation import validate_media_type

        formats = [
            ("test.mp4", "video/mp4"),
            ("test.avi", "video/avi"),
            ("test.mov", "video/quicktime"),
            ("test.webm", "video/webm"),
        ]

        for filename, mime_type in formats:
            test_file = tmp_path / filename
            test_file.touch()

            with patch('utils.validation.magic') as mock_magic:
                mock_magic.from_file.return_value = mime_type

                result = validate_media_type(str(test_file))
                assert result == "video", f"Failed for {filename}"


class TestValidationIntegration:
    """Integration tests combining multiple validations."""

    def test_validate_complete_image(self, sample_image):
        """Should pass all validations for valid image."""
        from utils.validation import (
            validate_file_size,
            validate_image_resolution,
            validate_media_type
        )

        with patch('utils.validation.magic') as mock_magic:
            mock_magic.from_file.return_value = "image/png"

            # All should pass
            validate_file_size(str(sample_image))
            validate_image_resolution(str(sample_image))
            result = validate_media_type(str(sample_image))

            assert result == "image"

    def test_validate_complete_gif(self, sample_gif):
        """Should pass all validations for valid GIF."""
        from utils.validation import (
            validate_file_size,
            validate_gif_frames,
            validate_media_type
        )

        with patch('utils.validation.magic') as mock_magic:
            mock_magic.from_file.return_value = "image/gif"

            validate_file_size(str(sample_gif))
            validate_gif_frames(str(sample_gif))
            result = validate_media_type(str(sample_gif))

            assert result == "gif"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_file(self, tmp_path):
        """Should handle empty files."""
        from utils.validation import validate_file_size

        empty_file = tmp_path / "empty.txt"
        empty_file.touch()

        # Should pass size validation (0 bytes)
        validate_file_size(str(empty_file), max_size_mb=1)

    def test_nonexistent_file(self, tmp_path):
        """Should raise for nonexistent files."""
        from utils.validation import validate_file_size

        with pytest.raises((FileNotFoundError, OSError)):
            validate_file_size(str(tmp_path / "nonexistent.txt"))

    def test_unicode_filename(self, tmp_path):
        """Should handle unicode filenames."""
        from utils.validation import validate_file_size

        unicode_file = tmp_path / "테스트_文件.txt"
        unicode_file.write_text("test content")

        validate_file_size(str(unicode_file), max_size_mb=1)

    def test_special_characters_in_path(self, tmp_path):
        """Should handle special characters in path."""
        from utils.validation import validate_file_size

        special_dir = tmp_path / "path with spaces (1)"
        special_dir.mkdir()
        test_file = special_dir / "file [test].txt"
        test_file.write_text("test")

        validate_file_size(str(test_file), max_size_mb=1)
