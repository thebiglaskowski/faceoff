"""
Face mapping utilities for UI.
Helper functions for managing face index mappings.
"""
import logging
from typing import Tuple

logger = logging.getLogger("FaceOff")


def add_face_mapping(source_idx: str, target_idx: str, current_mappings_text: str, face_mapping_manager) -> Tuple[str, str]:
    """
    Add a face mapping.
    
    Args:
        source_idx: Source face selection (e.g., "Source Face 0")
        target_idx: Target face selection (e.g., "Target Face 1")
        current_mappings_text: Current mappings display text
        face_mapping_manager: FaceMappingManager instance
        
    Returns:
        Tuple of (status_message, updated_mappings_text)
    """
    if source_idx is None or target_idx is None:
        return "⚠️ Select both source and target faces", current_mappings_text
    
    src_idx = int(source_idx.split()[-1])
    tgt_idx = int(target_idx.split()[-1])
    
    face_mapping_manager.add(src_idx, tgt_idx)
    
    status = f"✅ Added mapping: Source {src_idx} → Target {tgt_idx}"
    mappings_text = face_mapping_manager.get_display_text()
    
    return status, mappings_text


def clear_face_mappings(face_mapping_manager) -> Tuple[str, str]:
    """
    Clear all face mappings.
    
    Args:
        face_mapping_manager: FaceMappingManager instance
        
    Returns:
        Tuple of (status_message, mappings_text)
    """
    face_mapping_manager.clear()
    return "Mappings cleared", "No mappings"
