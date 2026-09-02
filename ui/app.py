"""
Gradio UI for FaceOff - AI Face Swapper application.
Refactored with modular components for clean architecture.
"""

# Suppress third-party warnings BEFORE any imports that might trigger them
import warnings

warnings.filterwarnings(
    "ignore", message=r".*`rcond` parameter will change.*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=r".*parameter 'pretrained' is deprecated.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=r".*Arguments other than a weight enum.*", category=UserWarning
)

import logging
from pathlib import Path

import gradio as gr

from processing.model_manager import get_model_status

# UI Components and Helpers
from ui.components.gallery_tab import create_gallery_tab, wire_gallery_tab
from ui.components.gif_tab import create_gif_tab, wire_gif_tab
from ui.components.image_tab import create_image_tab, wire_image_tab
from ui.components.models_tab import create_models_tab
from ui.components.terminal_tab import create_terminal_tab
from ui.components.video_tab import create_video_tab, wire_video_tab
from ui.faceoff_theme import get_theme

# Handler modules (extracted for cleaner architecture)
from ui.handlers.preset_handlers import (
    delete_selected_preset,
    get_default_preset,
    get_preset_choices,
    get_preset_info_text,
    load_preset_all_tabs,
    save_current_preset,
)
from ui.helpers.gpu_utils import get_gpu_status, refresh_gpu_info
from utils.logging_setup import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger("FaceOff")

# Create output directories on startup
output_base = Path("outputs")
output_base.mkdir(exist_ok=True)
(output_base / "image").mkdir(exist_ok=True)
(output_base / "gif").mkdir(exist_ok=True)
(output_base / "video").mkdir(exist_ok=True)
logger.info("Output directories initialized: outputs/image, outputs/gif, outputs/video")


def create_app():
    """Create and configure the Gradio application.

    Theme and CSS are passed to ``demo.launch()`` in ``main.py`` (Gradio 6+).
    """

    with gr.Blocks(title="FaceOff - Face Swapper") as demo:
        _, _, header_html = get_theme()
        gr.HTML(header_html)

        _missing = [r[0] for r in get_model_status() if "Missing" in r[2]]
        if _missing:
            gr.Markdown(
                f"⚠️ **Models missing:** {', '.join(_missing)} — open the "
                "**🧩 Models** tab below to download them, or run "
                "`python scripts/download_models.py`."
            )

        with gr.Tabs():
            # Create Image Tab
            img_components = create_image_tab()

            # Create GIF Tab
            gif_components = create_gif_tab()

            # Create Video Tab
            vid_components = create_video_tab()

            # Create Gallery Tab
            gallery_components = create_gallery_tab()

            # Model management (downloads, cache)
            create_models_tab()

            # Live application logs
            create_terminal_tab()

        # Preset Management Section (at bottom, collapsible)
        gr.Markdown("---")
        with gr.Accordion("🎨 Preset Management", open=False):
            gr.HTML(
                '<div class="face-swap-box"><strong>💾 Save & Load Settings</strong><br/>Manage your favorite face-swap configurations for quick access</div>'
            )

            with gr.Row():
                with gr.Column(scale=2):
                    preset_dropdown = gr.Dropdown(
                        choices=get_preset_choices(),
                        value=get_default_preset(),
                        label="Available Presets",
                        info="Select a preset to view or load",
                    )

                    preset_info_display = gr.Markdown(
                        value=get_preset_info_text(get_default_preset()),
                        label="Preset Information",
                    )

                with gr.Column(scale=1):
                    with gr.Row():
                        load_preset_btn = gr.Button(
                            "📥 Load Preset", variant="primary", size="sm"
                        )
                        delete_preset_btn = gr.Button(
                            "🗑️ Delete Preset", variant="secondary", size="sm"
                        )

                    gr.Markdown("**Save Current Settings**")
                    preset_name_input = gr.Textbox(
                        label="New Preset Name",
                        placeholder="My Custom Preset",
                        max_lines=1,
                    )
                    save_preset_btn = gr.Button(
                        "💾 Save as Preset", variant="primary", size="sm"
                    )

                    preset_status = gr.Textbox(
                        label="Status", value="", interactive=False, max_lines=1
                    )

        # GPU Memory Monitor (at bottom, collapsible)
        gr.Markdown("---")
        with gr.Accordion("🖥️ GPU Status Monitor", open=False):
            gr.HTML(
                '<div class="face-swap-box"><strong>⚡ System Performance</strong><br/>Monitor GPU memory usage and allocation</div>'
            )
            initial_gpu_info = get_gpu_status()
            gpu_textboxes = []

            with gr.Row():
                for idx, info in enumerate(initial_gpu_info):
                    gpu_textbox = gr.Textbox(
                        label=(
                            f"GPU {idx}" if len(initial_gpu_info) > 1 else "GPU Status"
                        ),
                        value=info,
                        lines=4,
                        interactive=False,
                    )
                    gpu_textboxes.append(gpu_textbox)

            refresh_gpu_btn = gr.Button("🔄 Refresh GPU Info", size="sm")

        # ============= EVENT HANDLERS =============

        # Media tab wiring lives with each tab component.
        wire_image_tab(img_components, gallery_components)
        wire_gif_tab(gif_components, gallery_components)
        wire_video_tab(vid_components, gallery_components)

        # Preset Management Event Handlers

        # Update preset info display when dropdown changes
        preset_dropdown.change(
            fn=get_preset_info_text,
            inputs=[preset_dropdown],
            outputs=[preset_info_display],
        )

        # Load preset button - updates all settings controls across all tabs
        # Uses load_preset_all_tabs to return all updates in one function call
        load_preset_btn.click(
            fn=load_preset_all_tabs,
            inputs=[preset_dropdown],
            outputs=[
                # Image tab visibility (4 outputs)
                img_components["model_row"],
                img_components["enhancement_options_row"],
                img_components["enhancement_advanced_row"],
                img_components["restoration_row"],
                # Image tab values (8 outputs)
                img_components["enhance_toggle"],
                img_components["restore_faces_toggle"],
                img_components["model_selector"],
                img_components["tile_size_slider"],
                img_components["outscale_slider"],
                img_components["use_fp32_checkbox"],
                img_components["pre_pad_slider"],
                img_components["restoration_weight_slider"],
                # GIF tab visibility (4 outputs)
                gif_components["model_row_gif"],
                gif_components["enhancement_options_row_gif"],
                gif_components["enhancement_advanced_row_gif"],
                gif_components["restoration_row_gif"],
                # GIF tab values (8 outputs)
                gif_components["enhance_toggle_gif"],
                gif_components["restore_faces_toggle_gif"],
                gif_components["model_selector_gif"],
                gif_components["tile_size_slider_gif"],
                gif_components["outscale_slider_gif"],
                gif_components["use_fp32_checkbox_gif"],
                gif_components["pre_pad_slider_gif"],
                gif_components["restoration_weight_slider_gif"],
                # Video tab visibility (4 outputs)
                vid_components["model_row_vid"],
                vid_components["enhancement_options_row_vid"],
                vid_components["enhancement_advanced_row_vid"],
                vid_components["restoration_row_vid"],
                # Video tab values (8 outputs)
                vid_components["enhance_toggle_vid"],
                vid_components["restore_faces_toggle_vid"],
                vid_components["model_selector_vid"],
                vid_components["tile_size_slider_vid"],
                vid_components["outscale_slider_vid"],
                vid_components["use_fp32_checkbox_vid"],
                vid_components["pre_pad_slider_vid"],
                vid_components["restoration_weight_slider_vid"],
                # Status message (1 output)
                preset_status,
            ],
        )

        # Save preset button
        save_preset_btn.click(
            fn=save_current_preset,
            inputs=[
                preset_name_input,
                img_components["enhance_toggle"],
                img_components["restore_faces_toggle"],
                img_components["model_selector"],
                img_components["tile_size_slider"],
                img_components["outscale_slider"],
                img_components["use_fp32_checkbox"],
                img_components["pre_pad_slider"],
                img_components["restoration_weight_slider"],
            ],
            outputs=[preset_status, preset_dropdown],
        ).then(
            fn=lambda name: "", inputs=[preset_name_input], outputs=[preset_name_input]
        )

        # Delete preset button
        delete_preset_btn.click(
            fn=delete_selected_preset,
            inputs=[preset_dropdown],
            outputs=[preset_status, preset_dropdown],
        )

        wire_gallery_tab(gallery_components)

        # GPU refresh handler
        refresh_gpu_btn.click(fn=refresh_gpu_info, outputs=gpu_textboxes)

    return demo
