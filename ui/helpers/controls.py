"""Shared Gradio control helpers (moved out of ui/app.py).

Toggle/refresh callbacks used by the per-tab wire_* functions. Living
here keeps ui/app.py a thin shell: it creates tabs and calls each tab's
wire function instead of holding all event wiring itself.
"""

import gradio as gr

from ui.components.gallery_tab import refresh_gallery
from utils.constants import (
    DEFAULT_HAT_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SWINIR_MODEL,
    HAT_MODEL_OPTIONS,
    MODEL_OPTIONS,
    SWINIR_MODEL_OPTIONS,
)


def toggle_enhancement_controls(enabled):
    """Toggle visibility of enhancement control rows and model selector."""
    return [
        gr.update(visible=enabled)
    ] * 4  # model_row, enhancement_options_row, enhancement_advanced_row, enhancement_model_selector


def update_model_choices(enhancement_framework: str):
    """
    Update the model selector choices based on the selected enhancement framework.

    Args:
        enhancement_framework: "RealESRGAN", "SwinIR", or "HAT"

    Returns:
        gr.update with new choices and value for the model selector
    """
    if enhancement_framework == "SwinIR":
        choices = list(SWINIR_MODEL_OPTIONS.keys())
        default = DEFAULT_SWINIR_MODEL
        info = "Select Swin2SR model for enhancement"
    elif enhancement_framework == "HAT":
        choices = list(HAT_MODEL_OPTIONS.keys())
        default = DEFAULT_HAT_MODEL
        info = "Select HAT (Hybrid Attention Transformer) model for enhancement"
    else:
        choices = list(MODEL_OPTIONS.keys())
        default = DEFAULT_MODEL
        info = "Select Real-ESRGAN model for enhancement"

    return gr.update(choices=choices, value=default, info=info)


def toggle_restoration_controls(enabled):
    """Toggle visibility of face restoration controls and model selector."""
    return [
        gr.update(visible=enabled)
    ] * 2  # restoration_row, restoration_model_selector


def toggle_denoise_slider(model_choice):
    """Show/hide denoise slider based on model selection."""
    supports_denoise = MODEL_OPTIONS.get(model_choice, {}).get(
        "supports_denoise", False
    )
    return gr.update(visible=supports_denoise)


JUMP_TO_GALLERY_JS = """() => {
  try {
    const tabs = document.querySelectorAll('button[role="tab"]');
    for (const t of tabs) {
      if ((t.textContent || "").includes("Gallery")) { t.click(); break; }
    }
  } catch (e) {}
}"""


def swap_with_button_toggle(fn):
    """Wrap a swap handler so the Swap button disables while it runs.

    Yields (result, button_update) pairs: disabled first, re-enabled on
    success or error. Errors propagate after re-enabling.
    """

    def _run(*args):
        yield gr.skip(), gr.update(interactive=False)
        try:
            result = fn(*args)
        except Exception:
            yield gr.skip(), gr.update(interactive=True)
            raise
        yield result, gr.update(interactive=True)

    return _run


def _refresh_gallery_for_type(media_type: str, limit: str = "24"):
    """
    Helper to refresh gallery for a specific media type.

    Used for auto-refresh after processing completes.
    """
    return refresh_gallery(media_type, limit)


def refresh_gallery_images(_result=None):
    """Refresh gallery showing images. Used after image processing."""
    return _refresh_gallery_for_type("Images")


def refresh_gallery_gifs(_result=None):
    """Refresh gallery showing GIFs. Used after GIF processing."""
    return _refresh_gallery_for_type("GIFs")


def refresh_gallery_videos(_result=None):
    """Refresh gallery showing videos. Used after video processing."""
    return _refresh_gallery_for_type("Videos")
