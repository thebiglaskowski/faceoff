"""
UI handler modules for FaceOff.

Contains:
- preset_handlers: Preset load/save/delete logic
- processing_handlers: Media processing handlers
"""

from ui.handlers.preset_handlers import (
    delete_selected_preset,
    get_default_preset,
    get_preset_choices,
    get_preset_info_text,
    load_preset_all_tabs,
    save_current_preset,
)
from ui.handlers.processing_handlers import (
    add_face_mapping_gif,
    add_face_mapping_image,
    add_face_mapping_video,
    add_face_mapping_wrapper,
    clear_face_mappings_gif,
    clear_face_mappings_image,
    clear_face_mappings_video,
    clear_face_mappings_wrapper,
    process_gif,
    process_image,
    process_video,
)

__all__ = [
    "load_preset_all_tabs",
    "save_current_preset",
    "delete_selected_preset",
    "get_preset_info_text",
    "get_preset_choices",
    "get_default_preset",
    "process_image",
    "process_gif",
    "process_video",
    "add_face_mapping_image",
    "add_face_mapping_gif",
    "add_face_mapping_video",
    "clear_face_mappings_image",
    "clear_face_mappings_gif",
    "clear_face_mappings_video",
    "add_face_mapping_wrapper",
    "clear_face_mappings_wrapper",
]
