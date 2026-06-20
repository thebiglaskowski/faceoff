"""
User-friendly error handling for FaceOff.
Converts technical errors into helpful messages with solutions.

Exception Hierarchy:
    FaceOffError (base)
    ├── FriendlyError (user-facing with suggestions)
    ├── ProcessingError (face swap/enhancement failures)
    │   ├── FaceDetectionError
    │   ├── FaceSwapError
    │   └── EnhancementError
    ├── ResourceError (memory/GPU issues)
    │   ├── OutOfMemoryError
    │   └── GPUError
    ├── ValidationError (input validation)
    │   ├── FileValidationError
    │   └── ConfigValidationError
    └── ModelError (model loading/inference)
"""
import functools
import logging
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from pathlib import Path

from utils.memory_manager import is_memory_error

logger = logging.getLogger("FaceOff")

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Exception Hierarchy
# =============================================================================

class FaceOffError(Exception):
    """Base exception for all FaceOff errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


class ProcessingError(FaceOffError):
    """Base class for processing-related errors."""
    pass


class FaceDetectionError(ProcessingError):
    """Error during face detection."""
    pass


class FaceSwapError(ProcessingError):
    """Error during face swapping."""
    pass


class EnhancementError(ProcessingError):
    """Error during image enhancement."""
    pass


class ResourceError(FaceOffError):
    """Base class for resource-related errors."""
    pass


class OutOfMemoryError(ResourceError):
    """GPU or system ran out of memory."""

    def __init__(
        self,
        message: str = "Out of memory",
        device_id: int = 0,
        allocated_mb: float = 0,
        total_mb: float = 0
    ):
        self.device_id = device_id
        self.allocated_mb = allocated_mb
        self.total_mb = total_mb
        details = f"GPU {device_id}: {allocated_mb:.0f}/{total_mb:.0f} MB"
        super().__init__(message, details)


class GPUError(ResourceError):
    """GPU-related error (unavailable, driver issues, etc.)."""
    pass


class ValidationError(FaceOffError):
    """Base class for validation errors."""
    pass


class FileValidationError(ValidationError):
    """File validation failed (format, size, permissions)."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        super().__init__(message, file_path)


class ConfigValidationError(ValidationError):
    """Configuration validation failed."""

    def __init__(self, message: str, key: Optional[str] = None, value: Any = None):
        self.key = key
        self.value = value
        details = f"{key}={value}" if key else None
        super().__init__(message, details)


class ModelError(FaceOffError):
    """Model loading or inference error."""

    def __init__(self, message: str, model_name: Optional[str] = None):
        self.model_name = model_name
        super().__init__(message, model_name)


class FriendlyError(FaceOffError):
    """Exception with user-friendly message and suggestions."""

    def __init__(
        self,
        title: str,
        message: str,
        suggestions: List[str],
        technical_details: str = ""
    ):
        self.title = title
        self.suggestions = suggestions
        self.technical_details = technical_details
        super().__init__(message, technical_details)
    
    def format_message(self) -> str:
        """Format error message for display."""
        lines = [
            f"❌ {self.title}",
            "━" * 60,
            self.message,
            ""
        ]
        
        if self.suggestions:
            lines.append("💡 Try these fixes:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
        
        return "\n".join(lines)


class ErrorHandler:
    """Handles and translates technical errors into user-friendly messages."""
    
    @staticmethod
    def handle_error(error: Exception, context: dict = None) -> FriendlyError:
        """
        Convert technical error to friendly error.
        
        Args:
            error: Original exception
            context: Additional context (current settings, file info, etc.)
            
        Returns:
            FriendlyError with helpful message
        """
        context = context or {}
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Out of Memory errors
        if is_memory_error(error):
            return ErrorHandler._handle_oom_error(context)
        
        # No faces detected
        if "no faces detected" in error_str or "no face" in error_str:
            return ErrorHandler._handle_no_faces_error(context)
        
        # File not found
        if "filenotfounderror" in error_type.lower() or "no such file" in error_str:
            return ErrorHandler._handle_file_not_found_error(error, context)
        
        # Permission errors
        if "permissionerror" in error_type.lower() or "permission denied" in error_str:
            return ErrorHandler._handle_permission_error(context)
        
        # GPU / ONNX runtime errors (cuDNN, execution provider failures)
        if any(
            x in error_str
            for x in (
                "cudnn",
                "ep_fail",
                "execution provider",
                "sublibrary_version_mismatch",
                "libcudnn",
            )
        ) or error_type in ("EPFail", "RuntimeException"):
            return ErrorHandler._handle_gpu_runtime_error(error, context)

        # Model initialization errors
        if "model" in error_str and ("failed" in error_str or "initialization" in error_str):
            return ErrorHandler._handle_model_error(error, context)
        
        # Invalid mapping errors
        if (
            "invalid mapping" in error_str
            or "out of range" in error_str
            or "no valid face mappings" in error_str
            or "face mappings did not match" in error_str
        ):
            return ErrorHandler._handle_invalid_mapping_error(context)
        
        # Path validation (must precede media handler — paths may contain ".gif")
        if "path not allowed" in error_str or "path traversal" in error_str:
            return ErrorHandler._handle_path_validation_error(error, context)

        # Video/GIF processing errors
        if any(
            x in error_str
            for x in [
                "codec",
                "unsupported format",
                "failed to process frame",
                "frame extraction",
                "streaming encode",
                "streaming decode",
            ]
        ):
            return ErrorHandler._handle_media_error(error, context)
        if error_type in ("FileNotFoundError",) and context.get("media_type") in ("video", "gif"):
            return ErrorHandler._handle_media_error(error, context)
        
        # Generic fallback
        return ErrorHandler._handle_generic_error(error, context)
    
    @staticmethod
    def _handle_oom_error(context: dict) -> FriendlyError:
        """Handle out of memory errors."""
        tile_size = context.get('tile_size', 256)
        outscale = context.get('outscale', 4)
        restore = context.get('restore_faces', False)
        
        suggestions = [
            f"Reduce tile size: {tile_size} → {max(128, tile_size//2)}",
            f"Lower output scale: {outscale}x → {outscale//2}x" if outscale > 2 else "Lower output scale to 2x",
        ]
        
        if restore:
            suggestions.append("Disable face restoration temporarily")
        
        suggestions.extend([
            "Close other GPU-intensive applications",
            "Process smaller sections at a time",
            "Use Fast Preview preset to test settings"
        ])
        
        message = (
            f"Your GPU ran out of memory during processing.\n\n"
            f"Current settings: tile={tile_size}, scale={outscale}x, restore={restore}\n"
            f"Suggested: tile={max(128, tile_size//2)}, scale={outscale//2}x, restore=False"
        )
        
        return FriendlyError(
            title="Out of Memory",
            message=message,
            suggestions=suggestions,
            technical_details=f"OOM with settings: {context}"
        )
    
    @staticmethod
    def _handle_no_faces_error(context: dict) -> FriendlyError:
        """Handle no faces detected errors."""
        confidence = context.get('face_confidence', 0.5)
        is_source = context.get('is_source_image', False)
        
        image_type = "source" if is_source else "target"
        
        suggestions = [
            f"Lower face detection confidence: {confidence} → {max(0.3, confidence - 0.1)}",
            "Ensure faces are clearly visible and well-lit",
            "Try a different image with more prominent faces",
            "Check if the image is corrupted or too small",
            "Use images with faces facing forward (not profile)"
        ]
        
        message = (
            f"No faces were detected in the {image_type} image.\n\n"
            f"Current confidence threshold: {confidence}\n"
            f"This means faces must have {confidence*100:.0f}% detection confidence to be recognized."
        )
        
        return FriendlyError(
            title="No Faces Detected",
            message=message,
            suggestions=suggestions
        )
    
    @staticmethod
    def _handle_file_not_found_error(error: Exception, context: dict) -> FriendlyError:
        """Handle file not found errors."""
        file_path = context.get('file_path', 'unknown')
        
        suggestions = [
            "Verify the file path is correct",
            "Check if the file was moved or deleted",
            "Ensure you have permission to access the file",
            "Try uploading the file again",
            "Check for special characters in the filename"
        ]
        
        message = f"Could not find or access the file:\n{file_path}"
        
        return FriendlyError(
            title="File Not Found",
            message=message,
            suggestions=suggestions,
            technical_details=str(error)
        )
    
    @staticmethod
    def _handle_permission_error(context: dict) -> FriendlyError:
        """Handle permission errors."""
        suggestions = [
            "Check file/folder permissions",
            "Close the file if it's open in another program",
            "Run the application with appropriate permissions",
            "Try saving to a different location",
            "Check if the output directory is read-only"
        ]
        
        message = "Permission denied. Cannot read or write to the specified location."
        
        return FriendlyError(
            title="Permission Denied",
            message=message,
            suggestions=suggestions
        )
    
    @staticmethod
    def _handle_gpu_runtime_error(error: Exception, context: dict) -> FriendlyError:
        """Handle CUDA/cuDNN/ONNX Runtime execution provider failures."""
        suggestions = [
            "Set gpu.tensorrt_enabled: false in config.yaml if TensorRT is not installed",
            "Run: uv run python scripts/verify_gpu_stack.py",
            "Pin nvidia-cudnn-cu12 to 9.20.x to match onnxruntime-gpu (see pyproject.toml)",
            "Restart the application after changing dependencies (uv sync)",
            "Use a single GPU selection instead of 'All GPUs' while debugging",
        ]

        message = (
            "ONNX Runtime failed on the GPU. This is usually a CUDA/cuDNN or "
            "TensorRT version mismatch — not a face-swap logic error.\n\n"
            "Common causes:\n"
            "• cuDNN sub-library version mismatch (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH)\n"
            "• TensorRT / ONNX Runtime mismatch (pin tensorrt>=10.9,<11 for ORT 1.22+)\n"
            "• TensorRT libs not preloaded — restart via python main.py\n"
            "• Mixed NVIDIA library paths on LD_LIBRARY_PATH"
        )

        return FriendlyError(
            title="GPU Runtime Error",
            message=message,
            suggestions=suggestions,
            technical_details=str(error),
        )

    @staticmethod
    def _handle_model_error(error: Exception, context: dict) -> FriendlyError:
        """Handle model initialization errors."""
        suggestions = [
            "Verify model files exist in the models/ directory",
            "Re-download missing model files",
            "Check if another process is using the GPU",
            "Restart the application",
            "Update CUDA/ONNX Runtime if outdated"
        ]
        
        message = (
            "Failed to initialize AI models. This usually means:\n"
            "• Model files are missing or corrupted\n"
            "• GPU drivers need updating\n"
            "• CUDA/ONNX Runtime has an issue"
        )
        
        return FriendlyError(
            title="Model Initialization Failed",
            message=message,
            suggestions=suggestions,
            technical_details=str(error)
        )
    
    @staticmethod
    def _handle_invalid_mapping_error(context: dict) -> FriendlyError:
        """Handle invalid face mapping errors."""
        src_count = context.get('source_faces', 0)
        dst_count = context.get('dest_faces', 0)
        
        suggestions = [
            f"Ensure face indices are valid (source: 0-{src_count-1}, dest: 0-{dst_count-1})",
            "Use auto-detect instead of manual mapping",
            "Check if faces were detected correctly",
            "Try re-uploading images with clearer faces"
        ]
        
        message = (
            f"Invalid face mapping specified.\n\n"
            f"Detected faces: {src_count} source, {dst_count} destination\n"
            f"Face indices should be 0-{src_count-1} for source, 0-{dst_count-1} for destination."
        )
        
        return FriendlyError(
            title="Invalid Face Mapping",
            message=message,
            suggestions=suggestions
        )
    
    @staticmethod
    def _handle_path_validation_error(error: Exception, context: dict) -> FriendlyError:
        """Handle rejected or unsafe file paths."""
        file_path = context.get("file_path", str(error))
        suggestions = [
            "Re-upload the file using the Gradio file picker",
            "Avoid manually editing file paths",
            "Save the file under the project's inputs/ directory if using a custom workflow",
        ]
        message = (
            "The uploaded file path could not be accepted.\n\n"
            f"Path: {file_path}"
        )
        return FriendlyError(
            title="Invalid File Path",
            message=message,
            suggestions=suggestions,
            technical_details=str(error),
        )

    @staticmethod
    def _handle_media_error(error: Exception, context: dict) -> FriendlyError:
        """Handle video/GIF processing errors."""
        suggestions = [
            "Verify the file is a valid video/GIF format",
            "Try converting to a common format (MP4, GIF)",
            "Check if the file is corrupted",
            "Ensure FFmpeg is installed correctly",
            "Try a different video/GIF file"
        ]
        
        message = (
            "Failed to process video or GIF file. Common causes:\n"
            "• Unsupported codec or format\n"
            "• Corrupted file\n"
            "• Missing FFmpeg codecs"
        )
        
        return FriendlyError(
            title="Media Processing Error",
            message=message,
            suggestions=suggestions,
            technical_details=str(error)
        )
    
    @staticmethod
    def _handle_generic_error(error: Exception, context: dict) -> FriendlyError:
        """Handle unknown errors with generic helpful message."""
        suggestions = [
            "Try restarting the application",
            "Check application logs for details",
            "Verify all files and settings are correct",
            "Try with default/simpler settings",
            "Report this issue if it persists"
        ]
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        message = (
            f"An unexpected error occurred: {error_type}\n\n"
            f"Details: {error_msg[:200]}"
        )
        
        return FriendlyError(
            title="Unexpected Error",
            message=message,
            suggestions=suggestions,
            technical_details=f"{error_type}: {error_msg}"
        )


def wrap_error(func: F) -> F:
    """Decorator to automatically convert exceptions to friendly errors."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FriendlyError:
            # Already a friendly error, re-raise
            raise
        except FaceOffError as e:
            # Convert FaceOffError subclasses to friendly error
            context = kwargs.get('context', {})
            friendly_error = ErrorHandler.handle_error(e, context)
            logger.error("Error in %s: %s", func.__name__, e, exc_info=True)
            raise friendly_error from e
        except Exception as e:
            # Convert unknown exceptions to friendly error
            context = kwargs.get('context', {})
            friendly_error = ErrorHandler.handle_error(e, context)
            logger.error("Error in %s: %s", func.__name__, e, exc_info=True)
            raise friendly_error from e

    return wrapper  # type: ignore


def convert_to_faceoff_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> FaceOffError:
    """
    Convert a standard exception to an appropriate FaceOffError subclass.

    Args:
        error: Original exception
        context: Additional context about the operation

    Returns:
        Appropriate FaceOffError subclass
    """
    context = context or {}
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Already a FaceOffError
    if isinstance(error, FaceOffError):
        return error

    # Out of Memory (PyTorch, ONNX BFC arena, etc.)
    if is_memory_error(error):
        return OutOfMemoryError(
            str(error),
            device_id=context.get('device_id', 0),
            allocated_mb=context.get('allocated_mb', 0),
            total_mb=context.get('total_mb', 0)
        )

    # No faces detected
    if "no faces detected" in error_str or "no face" in error_str:
        return FaceDetectionError(str(error))

    # File not found or permission
    if "filenotfounderror" in error_type.lower() or "no such file" in error_str:
        return FileValidationError(str(error), context.get('file_path'))

    if "permissionerror" in error_type.lower() or "permission denied" in error_str:
        return FileValidationError(f"Permission denied: {error}", context.get('file_path'))

    # Model errors
    if "model" in error_str and ("failed" in error_str or "initialization" in error_str):
        return ModelError(str(error), context.get('model_name'))

    # GPU errors
    if "cuda" in error_str or "gpu" in error_str:
        return GPUError(str(error))

    # Default to base error
    return FaceOffError(str(error))
