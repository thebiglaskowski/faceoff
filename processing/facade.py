"""
Processing layer facade for UI access to core functionality.

This module provides a clean API boundary between the UI and core layers.
UI modules should import from here rather than directly from core/.

Architecture (from CLAUDE.md):
- ui/ may import from processing/, utils/
- processing/ may import from core/, utils/
- core/ may import from utils/
- utils/ must NOT import from core/, processing/, ui/
"""

# Re-export face processor functionality
from core.face_processor import (
    FaceProcessor,
    FaceMappingManager,
    sort_faces_by_position,
    filter_faces_by_confidence,
)

# Re-export GPU management
from core.gpu_manager import GPUManager

# Re-export media processor for model preloading
from core.media_processor import MediaProcessor

__all__ = [
    # Face processing
    "FaceProcessor",
    "FaceMappingManager",
    "sort_faces_by_position",
    "filter_faces_by_confidence",
    # GPU management
    "GPUManager",
    # Media processing
    "MediaProcessor",
]
