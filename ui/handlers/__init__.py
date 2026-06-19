"""
UI handler modules for FaceOff.

Contains:
- preset_handlers: Preset load/save/delete logic
- processing_handlers: Media processing handlers
"""

from ui.handlers.preset_handlers import (
    load_preset_all_tabs,
    save_current_preset,
    delete_selected_preset,
    get_preset_info_text,
    get_preset_choices,
    get_default_preset,
)

from ui.handlers.processing_handlers import (
    process_image,
    process_gif,
    process_video,
    add_face_mapping_image,
    add_face_mapping_gif,
    add_face_mapping_video,
    clear_face_mappings_image,
    clear_face_mappings_gif,
    clear_face_mappings_video,
    add_face_mapping_wrapper,
    clear_face_mappings_wrapper,
)

__all__ = [
    'load_preset_all_tabs',
    'save_current_preset',
    'delete_selected_preset',
    'get_preset_info_text',
    'get_preset_choices',
    'get_default_preset',
    'process_image',
    'process_gif',
    'process_video',
    'add_face_mapping_image',
    'add_face_mapping_gif',
    'add_face_mapping_video',
    'clear_face_mappings_image',
    'clear_face_mappings_gif',
    'clear_face_mappings_video',
    'add_face_mapping_wrapper',
    'clear_face_mappings_wrapper',
]
