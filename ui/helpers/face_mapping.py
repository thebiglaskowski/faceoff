"""
Face mapping utilities for UI.
Helper functions for managing face index mappings.
"""

import logging
from typing import Any, Callable, Tuple

import gradio as gr

logger = logging.getLogger("FaceOff")


def select_mapping_index(evt: gr.SelectData, prefix: str):
    """Click a face thumbnail to choose it in the mapping dropdown."""
    if evt.index is None:
        return gr.skip()
    return gr.update(value=f"{prefix} {evt.index}")


def _parse_face_choice(choice: Any) -> int | None:
    try:
        return int(str(choice).split()[-1])
    except (ValueError, IndexError, AttributeError):
        return None


def pick_target_and_map(
    evt: gr.SelectData,
    source_choice: Any,
    current_mappings_text: str,
    add_fn: Callable[[int, int, str], Tuple[Any, Any]],
    target_prefix: str,
):
    """Click a target face to choose it AND create the mapping at once."""
    if evt.index is None:
        return gr.skip(), gr.skip(), gr.skip()
    src_idx = _parse_face_choice(source_choice)
    if src_idx is None:
        return (
            gr.update(value=f"{target_prefix} {evt.index}"),
            "⚠️ Pick a source face first, then click a target face",
            current_mappings_text,
        )
    status, mappings = add_fn(src_idx, evt.index, current_mappings_text)
    return gr.update(value=f"{target_prefix} {evt.index}"), status, mappings


def auto_map_single_pair(
    src_gallery: Any,
    tgt_gallery: Any,
    current_mappings_text: str,
    add_fn: Callable[[int, int, str], Tuple[Any, Any]],
):
    """After detection, auto-map when exactly one face is on each side."""
    try:
        single = len(src_gallery or []) == 1 and len(tgt_gallery or []) == 1
    except TypeError:
        single = False
    if not single:
        return gr.skip(), gr.skip()
    return add_fn(0, 0, current_mappings_text)


def add_face_mapping(
    source_idx: str, target_idx: str, current_mappings_text: str, face_mapping_manager
) -> Tuple[str, str]:
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

    try:
        src_idx = int(str(source_idx).split()[-1])
        tgt_idx = int(str(target_idx).split()[-1])
    except (ValueError, IndexError, AttributeError):
        return (
            "⚠️ Could not read face selection — re-run Detect Faces first",
            current_mappings_text,
        )

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
