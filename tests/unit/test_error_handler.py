"""
Unit tests for utils/error_handler.py

Tests:
- FriendlyError class
- ErrorHandler static methods
- wrap_error decorator
- Error message formatting
"""

import pytest
from unittest.mock import MagicMock, patch
import torch


class TestFriendlyError:
    """Tests for FriendlyError exception class."""

    def test_friendly_error_creation(self):
        """Should create FriendlyError with all fields."""
        error = FriendlyError(
            title="Test Error",
            message="This is a test",
            suggestions=["Try this", "Or this"],
            technical_details="Stack trace here"
        )

        assert error.title == "Test Error"
        assert error.message == "This is a test"
        assert len(error.suggestions) == 2
        assert error.technical_details == "Stack trace here"

    def test_friendly_error_format_message(self):
        """Should format error message correctly."""
        error = FriendlyError(
            title="Test Error",
            message="Something went wrong",
            suggestions=["Fix A", "Fix B"]
        )

        formatted = error.format_message()

        assert "Test Error" in formatted
        assert "Something went wrong" in formatted
        assert "Fix A" in formatted
        assert "Fix B" in formatted
        assert "Try these fixes" in formatted

    def test_friendly_error_no_suggestions(self):
        """Should format correctly with no suggestions."""
        error = FriendlyError(
            title="Error",
            message="Message",
            suggestions=[]
        )

        formatted = error.format_message()

        assert "Error" in formatted
        assert "Try these fixes" not in formatted

    def test_friendly_error_inherits_exception(self):
        """Should be raisable as exception."""
        error = FriendlyError(
            title="Test",
            message="Raisable",
            suggestions=[]
        )

        with pytest.raises(FriendlyError) as exc_info:
            raise error

        assert str(exc_info.value) == "Raisable"


# Import after test class to avoid import errors during collection
from utils.error_handler import FriendlyError, ErrorHandler, wrap_error


class TestErrorHandlerOOM:
    """Tests for OOM error handling."""

    def test_handle_oom_error(self):
        """Should handle OOM errors correctly."""
        oom_error = MemoryError("CUDA out of memory")

        context = {
            'tile_size': 512,
            'outscale': 4,
            'restore_faces': True
        }

        result = ErrorHandler.handle_error(oom_error, context)

        assert result.title == "Out of Memory"
        assert "tile" in result.message.lower()
        assert len(result.suggestions) > 0

    def test_oom_suggests_tile_reduction(self):
        """Should suggest reducing tile size."""
        oom_error = MemoryError("out of memory")

        context = {'tile_size': 512, 'outscale': 4}
        result = ErrorHandler.handle_error(oom_error, context)

        tile_suggestion = [s for s in result.suggestions if "tile" in s.lower()]
        assert len(tile_suggestion) > 0

    def test_oom_suggests_scale_reduction(self):
        """Should suggest reducing output scale."""
        oom_error = MemoryError("out of memory")

        context = {'tile_size': 256, 'outscale': 4}
        result = ErrorHandler.handle_error(oom_error, context)

        scale_suggestion = [s for s in result.suggestions if "scale" in s.lower()]
        assert len(scale_suggestion) > 0


class TestErrorHandlerNoFaces:
    """Tests for no faces detected error handling."""

    def test_handle_no_faces_error(self):
        """Should handle no faces errors correctly."""
        error = ValueError("No faces detected in image")

        context = {'face_confidence': 0.5}
        result = ErrorHandler.handle_error(error, context)

        assert result.title == "No Faces Detected"
        assert "confidence" in result.message.lower()

    def test_no_faces_suggests_lower_confidence(self):
        """Should suggest lowering confidence threshold."""
        error = ValueError("no face found")

        context = {'face_confidence': 0.7}
        result = ErrorHandler.handle_error(error, context)

        conf_suggestion = [s for s in result.suggestions if "confidence" in s.lower()]
        assert len(conf_suggestion) > 0


class TestErrorHandlerFileNotFound:
    """Tests for file not found error handling."""

    def test_handle_file_not_found(self):
        """Should handle file not found errors."""
        error = FileNotFoundError("No such file: test.jpg")

        context = {'file_path': '/path/to/test.jpg'}
        result = ErrorHandler.handle_error(error, context)

        assert result.title == "File Not Found"
        assert "/path/to/test.jpg" in result.message

    def test_file_not_found_suggestions(self):
        """Should provide helpful file suggestions."""
        error = FileNotFoundError("Missing file")

        result = ErrorHandler.handle_error(error, {})

        assert any("path" in s.lower() or "file" in s.lower()
                   for s in result.suggestions)


class TestErrorHandlerPermission:
    """Tests for permission error handling."""

    def test_handle_permission_error(self):
        """Should handle permission errors."""
        error = PermissionError("Access denied")

        result = ErrorHandler.handle_error(error, {})

        assert result.title == "Permission Denied"
        assert any("permission" in s.lower() for s in result.suggestions)


class TestErrorHandlerModel:
    """Tests for model error handling."""

    def test_handle_model_error(self):
        """Should handle model initialization errors."""
        error = RuntimeError("Model initialization failed")

        result = ErrorHandler.handle_error(error, {})

        assert result.title == "Model Initialization Failed"
        assert any("model" in s.lower() for s in result.suggestions)


class TestErrorHandlerInvalidMapping:
    """Tests for invalid face mapping error handling."""

    def test_handle_invalid_mapping(self):
        """Should handle invalid face mapping errors."""
        error = ValueError("Invalid mapping: index out of range")

        context = {'source_faces': 2, 'dest_faces': 3}
        result = ErrorHandler.handle_error(error, context)

        assert result.title == "Invalid Face Mapping"
        assert "0-1" in result.message  # source: 0-1
        assert "0-2" in result.message  # dest: 0-2


class TestErrorHandlerMedia:
    """Tests for media processing error handling."""

    def test_handle_video_error(self):
        """Should handle video processing errors."""
        error = RuntimeError("Video codec not supported")

        result = ErrorHandler.handle_error(error, {})

        assert result.title == "Media Processing Error"

    def test_handle_gif_error(self):
        """Should handle GIF processing errors."""
        error = RuntimeError("GIF frame extraction failed")

        result = ErrorHandler.handle_error(error, {})

        assert result.title == "Media Processing Error"


class TestErrorHandlerGeneric:
    """Tests for generic/unknown error handling."""

    def test_handle_unknown_error(self):
        """Should handle unknown errors gracefully."""
        error = KeyError("some unknown key")

        result = ErrorHandler.handle_error(error, {})

        assert result.title == "Unexpected Error"
        assert "KeyError" in result.message

    def test_generic_error_has_suggestions(self):
        """Generic errors should still have suggestions."""
        error = RuntimeError("Unknown error occurred")

        result = ErrorHandler.handle_error(error, {})

        assert len(result.suggestions) > 0

    def test_truncates_long_error_message(self):
        """Should truncate very long error messages."""
        long_message = "A" * 500
        error = RuntimeError(long_message)

        result = ErrorHandler.handle_error(error, {})

        # Message should be truncated
        assert len(result.message) < 500


class TestWrapErrorDecorator:
    """Tests for wrap_error decorator."""

    def test_wrap_error_returns_result(self):
        """Should return function result when no error."""
        @wrap_error
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_wrap_error_converts_exception(self):
        """Should convert exceptions to FriendlyError."""
        @wrap_error
        def failing_function():
            raise MemoryError("out of memory")

        with pytest.raises(FriendlyError) as exc_info:
            failing_function()

        assert exc_info.value.title == "Out of Memory"

    def test_wrap_error_preserves_friendly_error(self):
        """Should not re-wrap FriendlyError."""
        original = FriendlyError(
            title="Original",
            message="Original message",
            suggestions=["Original suggestion"]
        )

        @wrap_error
        def raises_friendly():
            raise original

        with pytest.raises(FriendlyError) as exc_info:
            raises_friendly()

        assert exc_info.value.title == "Original"

    def test_wrap_error_uses_context(self):
        """Should pass context to error handler."""
        @wrap_error
        def function_with_context(context=None):
            raise MemoryError("OOM")

        with pytest.raises(FriendlyError):
            function_with_context(context={'tile_size': 512})


class TestErrorContext:
    """Tests for context handling in errors."""

    def test_context_affects_suggestions(self):
        """Context should affect error suggestions."""
        error = MemoryError("OOM")

        # With restore_faces enabled
        context1 = {'restore_faces': True, 'tile_size': 256, 'outscale': 4}
        result1 = ErrorHandler.handle_error(error, context1)

        # Without restore_faces
        context2 = {'restore_faces': False, 'tile_size': 256, 'outscale': 4}
        result2 = ErrorHandler.handle_error(error, context2)

        # Should have different suggestions
        restore_mentions1 = sum(1 for s in result1.suggestions if "restor" in s.lower())
        restore_mentions2 = sum(1 for s in result2.suggestions if "restor" in s.lower())

        assert restore_mentions1 > restore_mentions2

    def test_empty_context_works(self):
        """Should work with empty context."""
        error = ValueError("No faces")

        result = ErrorHandler.handle_error(error, {})

        assert result is not None
        assert len(result.suggestions) > 0

    def test_none_context_works(self):
        """Should work with None context."""
        error = ValueError("No faces")

        result = ErrorHandler.handle_error(error, None)

        assert result is not None
