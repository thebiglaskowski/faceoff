"""
User-friendly error handling for FaceOff.
Converts technical errors into helpful messages with solutions.
"""
import logging
import torch
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger("FaceOff")


class FriendlyError(Exception):
    """Exception with user-friendly message and suggestions."""
    
    def __init__(self, title: str, message: str, suggestions: list[str], technical_details: str = ""):
        self.title = title
        self.message = message
        self.suggestions = suggestions
        self.technical_details = technical_details
        super().__init__(message)
    
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
        if "out of memory" in error_str or isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
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
        
        # Model initialization errors
        if "model" in error_str and ("failed" in error_str or "initialization" in error_str):
            return ErrorHandler._handle_model_error(error, context)
        
        # Invalid mapping errors
        if "invalid mapping" in error_str or "out of range" in error_str:
            return ErrorHandler._handle_invalid_mapping_error(context)
        
        # Video/GIF processing errors
        if any(x in error_str for x in ["video", "gif", "frame", "codec"]):
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


def wrap_error(func):
    """Decorator to automatically convert exceptions to friendly errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FriendlyError:
            # Already a friendly error, re-raise
            raise
        except Exception as e:
            # Convert to friendly error
            context = kwargs.get('context', {})
            friendly_error = ErrorHandler.handle_error(e, context)
            logger.error("Error in %s: %s", func.__name__, e, exc_info=True)
            raise friendly_error from e
    
    return wrapper
