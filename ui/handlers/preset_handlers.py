"""
Preset management handlers for FaceOff UI.

This module handles:
- Loading presets and returning updates for all UI components
- Saving current settings as presets
- Deleting presets
- Preset information display
"""

import logging
import gradio as gr
from typing import Tuple, List, Any, Optional

from utils.preset_manager import PresetManager, initialize_default_presets
from utils.constants import (
    MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_TILE_SIZE,
    DEFAULT_OUTSCALE, DEFAULT_USE_FP32, DEFAULT_PRE_PAD
)

logger = logging.getLogger("FaceOff")

# Module-level preset manager instance
_preset_manager: Optional[PresetManager] = None


def get_preset_manager() -> PresetManager:
    """Get or create the preset manager singleton."""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
        initialize_default_presets(_preset_manager)
    return _preset_manager


def get_preset_choices() -> List[str]:
    """Get list of available preset names for dropdown."""
    presets = get_preset_manager().list_presets()
    return [p['name'] for p in presets]


def get_default_preset() -> Optional[str]:
    """Get the default preset name (Balanced if available, otherwise first in list)."""
    choices = get_preset_choices()
    if "Balanced" in choices:
        return "Balanced"
    return choices[0] if choices else None


def load_preset_settings(preset_name: str) -> List[Any]:
    """
    Load preset and return settings as tuple for updating UI controls.

    Returns list of 8 gr.update() objects:
    - enhance toggle
    - restore_faces toggle
    - model selector
    - tile_size slider
    - outscale slider
    - use_fp32 checkbox
    - pre_pad slider
    - restoration_weight slider
    """
    if not preset_name:
        return [gr.update() for _ in range(8)]

    try:
        settings = get_preset_manager().load_preset(preset_name)
        logger.info(f"Loaded preset: {preset_name}")

        # Get the model name from settings
        model_name = settings.get('model') or settings.get('model_name', DEFAULT_MODEL)
        logger.debug(f"Preset model_name: {model_name}")

        return [
            gr.update(value=settings.get('enhance', False)),
            gr.update(value=settings.get('restore_faces') or settings.get('restore', False)),
            gr.update(value=model_name),
            gr.update(value=settings.get('tile_size', DEFAULT_TILE_SIZE)),
            gr.update(value=settings.get('outscale', DEFAULT_OUTSCALE)),
            gr.update(value=settings.get('use_fp32', DEFAULT_USE_FP32)),
            gr.update(value=settings.get('pre_pad', DEFAULT_PRE_PAD)),
            gr.update(value=settings.get('restoration_weight', 0.5)),
        ]
    except Exception as e:
        logger.error(f"Error loading preset: {e}")
        return [gr.update() for _ in range(8)]


def load_preset_all_tabs(preset_name: str) -> Tuple[Any, ...]:
    """
    Load preset and return all updates for all tabs in a single call.

    This replaces the 6-step .then() chain with a single function.

    Returns tuple of 37 gr.update() objects:
    - 3 visibility updates for image tab enhancement rows
    - 1 visibility update for image tab restoration row
    - 8 value updates for image tab controls
    - 3 visibility updates for gif tab enhancement rows
    - 1 visibility update for gif tab restoration row
    - 8 value updates for gif tab controls
    - 3 visibility updates for video tab enhancement rows
    - 1 visibility update for video tab restoration row
    - 8 value updates for video tab controls
    - 1 status message
    """
    if not preset_name:
        # Return no-op updates (37 total)
        return tuple([gr.update() for _ in range(37)])

    try:
        settings = get_preset_manager().load_preset(preset_name)
        logger.info(f"Loading preset for all tabs: {preset_name}")

        enhance = settings.get('enhance', False)
        restore_faces = settings.get('restore_faces') or settings.get('restore', False)
        model_name = settings.get('model') or settings.get('model_name', DEFAULT_MODEL)
        tile_size = settings.get('tile_size', DEFAULT_TILE_SIZE)
        outscale = settings.get('outscale', DEFAULT_OUTSCALE)
        use_fp32 = settings.get('use_fp32', DEFAULT_USE_FP32)
        pre_pad = settings.get('pre_pad', DEFAULT_PRE_PAD)
        restoration_weight = settings.get('restoration_weight', 0.5)

        def make_value_updates():
            """Create fresh value update objects for each tab."""
            return [
                gr.update(value=enhance),
                gr.update(value=restore_faces),
                gr.update(value=model_name),
                gr.update(value=tile_size),
                gr.update(value=outscale),
                gr.update(value=use_fp32),
                gr.update(value=pre_pad),
                gr.update(value=restoration_weight),
            ]

        result = []

        # Image tab: visibility (4) + values (8) = 12
        result.extend([
            gr.update(visible=enhance),  # model_row
            gr.update(visible=enhance),  # enhancement_options_row
            gr.update(visible=enhance),  # enhancement_advanced_row
            gr.update(visible=restore_faces),  # restoration_row
        ])
        result.extend(make_value_updates())

        # GIF tab: visibility (4) + values (8) = 12
        result.extend([
            gr.update(visible=enhance),
            gr.update(visible=enhance),
            gr.update(visible=enhance),
            gr.update(visible=restore_faces),
        ])
        result.extend(make_value_updates())

        # Video tab: visibility (4) + values (8) = 12
        result.extend([
            gr.update(visible=enhance),
            gr.update(visible=enhance),
            gr.update(visible=enhance),
            gr.update(visible=restore_faces),
        ])
        result.extend(make_value_updates())

        # Status message (1)
        result.append("Preset loaded successfully across all tabs!")

        return tuple(result)

    except Exception as e:
        logger.error(f"Error loading preset for all tabs: {e}")
        return tuple([gr.update() for _ in range(36)] + [f"Error loading preset: {e}"])


def save_current_preset(
    preset_name: str,
    enhance: bool,
    restore: bool,
    model_display_name: str,
    tile_size: int,
    outscale: int,
    use_fp32: bool,
    pre_pad: int,
    restoration_weight: float
) -> Tuple[Any, Any]:
    """
    Save current settings as a new preset.

    Returns:
        Tuple of (status_update, dropdown_update)
    """
    if not preset_name or not preset_name.strip():
        return gr.update(value="Please enter a preset name"), gr.update()

    preset_name = preset_name.strip()

    try:
        # Extract the actual model name from the display name
        model_name = model_display_name
        for display_name, model_info in MODEL_OPTIONS.items():
            if display_name == model_display_name:
                model_name = model_info.get('model_name', model_display_name)
                break

        settings = {
            'enhance': enhance,
            'restore_faces': restore,
            'model_name': model_name,
            'tile_size': tile_size,
            'outscale': outscale,
            'use_fp32': use_fp32,
            'pre_pad': pre_pad,
            'restoration_weight': restoration_weight,
        }

        get_preset_manager().save_preset(preset_name, settings, description="Custom user preset")
        logger.info(f"Saved preset: {preset_name}")

        return (
            gr.update(value=f"Saved preset: {preset_name}"),
            gr.update(choices=get_preset_choices(), value=preset_name)
        )
    except Exception as e:
        logger.error(f"Error saving preset: {e}")
        return gr.update(value=f"Error: {str(e)}"), gr.update()


def delete_selected_preset(preset_name: str) -> Tuple[Any, Any]:
    """
    Delete the selected preset.

    Returns:
        Tuple of (status_update, dropdown_update)
    """
    if not preset_name:
        return gr.update(value="No preset selected"), gr.update()

    try:
        get_preset_manager().delete_preset(preset_name)
        logger.info(f"Deleted preset: {preset_name}")

        new_choices = get_preset_choices()
        return (
            gr.update(value=f"Deleted preset: {preset_name}"),
            gr.update(choices=new_choices, value=new_choices[0] if new_choices else None)
        )
    except Exception as e:
        logger.error(f"Error deleting preset: {e}")
        return gr.update(value=f"Error: {str(e)}"), gr.update()


def get_preset_info_text(preset_name: str) -> str:
    """Get preset information for display."""
    if not preset_name:
        return "No preset selected"

    try:
        info = get_preset_manager().get_preset_info(preset_name)
        description = info.get('description', 'No description')
        created = info.get('created', 'Unknown')
        settings = info.get('settings', {})

        model_name = settings.get('model') or settings.get('model_name', 'N/A')
        restore_faces = settings.get('restore_faces') or settings.get('restore', False)

        info_text = f"**{preset_name}**\n\n"
        info_text += f"{description}\n\n"
        info_text += f"_Created: {created}_\n\n"
        info_text += "**Settings:**\n"
        info_text += f"- Enhancement: {'Yes' if settings.get('enhance') else 'No'}\n"
        info_text += f"- Face Restoration: {'Yes' if restore_faces else 'No'}\n"
        info_text += f"- Model: {model_name}\n"
        info_text += f"- Tile Size: {settings.get('tile_size', 'N/A')}\n"
        info_text += f"- Upscale: {settings.get('outscale', 'N/A')}x\n"
        info_text += f"- FP32: {'Yes' if settings.get('use_fp32') else 'No'}\n"

        return info_text
    except Exception as e:
        return f"Error loading preset info: {str(e)}"
