"""
Gradio UI for FaceOff - AI Face Swapper application.
Refactored with modular components for clean architecture.
"""
import gradio as gr
import logging
from pathlib import Path

from utils.logging_setup import setup_logging
from utils.constants import MODEL_OPTIONS

# UI Components and Helpers
from ui.components.image_tab import create_image_tab
from ui.components.gif_tab import create_gif_tab
from ui.components.video_tab import create_video_tab
from ui.components.gallery_tab import create_gallery_tab, update_gallery, refresh_gallery
from ui.helpers.gallery_utils import delete_file
from ui.helpers.gpu_utils import get_gpu_status, refresh_gpu_info
from ui.helpers.face_detection import detect_faces_simple, detect_faces_for_mapping, detect_faces_with_thumbnails
from ui.helpers.preview import show_gif_preview

# Handler modules (extracted for cleaner architecture)
from ui.handlers.preset_handlers import (
    load_preset_all_tabs,
    load_preset_settings,
    save_current_preset,
    delete_selected_preset,
    get_preset_info_text,
    get_preset_choices,
    get_default_preset,
    get_preset_manager,
)
from ui.handlers.processing_handlers import (
    process_image,
    process_gif,
    process_video,
    add_face_mapping_wrapper,
    clear_face_mappings_wrapper,
)

# Custom CSS styling
CUSTOM_CSS = """
:root {
    --primary-color: #7C3AED;
    --secondary-color: #EC4899;
    --success-color: #10B981;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.header-text {
    color: #7C3AED !important;
    font-weight: 800;
    font-size: 2.5em;
    text-align: center;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    text-align: center;
    font-size: 1.1em;
    color: #6B7280;
    margin-bottom: 2rem;
}

@media (prefers-color-scheme: dark) {
    .header-text {
        color: #A78BFA !important;
    }
    .header-subtitle {
        color: #9CA3AF !important;
    }
}

.primary-btn {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: transform 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(124, 58, 237, 0.3) !important;
}

.face-swap-box {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(236, 72, 153, 0.1));
    border-left: 4px solid var(--primary-color);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
"""

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


def toggle_enhancement_controls(enabled):
    """Toggle visibility of enhancement control rows."""
    return [gr.update(visible=enabled)] * 3


def show_delete_controls(evt: gr.SelectData):
    """Show delete controls when a file is selected in the gallery."""
    logger.info(f"Gallery selection event - index: {evt.index}, value type: {type(evt.value)}")
    logger.info(f"Gallery selection value: {evt.value}")
    
    if evt.index is None:
        # No selection
        return gr.update(visible=False), gr.update(visible=False), ""
    
    # Extract filename from caption (format: "filename.ext\nYYYY-MM-DD HH:MM:SS")
    selected_filename = None
    
    if isinstance(evt.value, dict):
        # Get caption which contains the original filename
        caption = evt.value.get('caption', '')
        if caption:
            # Caption format is "filename.ext\ntimestamp"
            selected_filename = caption.split('\n')[0] if '\n' in caption else caption
            logger.info(f"Extracted filename from caption: {selected_filename}")
    
    if selected_filename:
        return (
            gr.update(value=selected_filename, visible=True),  # selected_file_display (just filename)
            gr.update(visible=True),  # delete_btn
            ""  # delete_status (clear previous messages)
        )
    else:
        logger.warning("Could not extract filename from gallery selection")
        return gr.update(visible=False), gr.update(visible=False), ""


def delete_and_refresh(file_path: str, media_type_display: str, limit: str = "24"):
    """Delete selected file and refresh the gallery."""
    logger.info(f"Delete triggered - file_path: '{file_path}', media_type: '{media_type_display}'")
    
    if not file_path:
        logger.warning("No file path provided for deletion")
        return (
            "❌ No file selected",  # delete_status
            gr.update(),  # gallery (no change)
            gr.update(),  # file_count_text (no change)
            gr.update(visible=False),  # delete_btn (hide)
            gr.update(visible=False)  # selected_file_display (hide)
        )
    
    # Map display name to internal type
    media_type_map = {
        "Images": "image",
        "GIFs": "gif",
        "Videos": "video"
    }
    media_type = media_type_map.get(media_type_display, "image")
    logger.info(f"Mapped media type: {media_type}")
    
    # Reconstruct the full path from filename
    # file_path is just the filename (e.g., "swapped_1762607114989.png")
    from pathlib import Path
    output_base = Path("outputs")
    full_file_path = str(output_base / media_type / file_path)
    logger.info(f"Reconstructed full path: {full_file_path}")
    
    # Delete the file
    _, message = delete_file(full_file_path, media_type)
    logger.info(f"Delete result: {message}")
    
    # Refresh the gallery (returns tuple of gr.update() objects)
    gallery_update, count_text_update = update_gallery(media_type_display, limit)
    
    # Return updates
    return (
        message,  # delete_status (plain string for Markdown)
        gallery_update,  # gallery
        count_text_update,  # file_count_text
        gr.update(visible=False),  # delete_btn (hide after delete)
        gr.update(visible=False)  # selected_file_display (hide after delete)
    )


def toggle_restoration_controls(enabled):
    """Toggle visibility of face restoration controls."""
    return gr.update(visible=enabled)


def toggle_denoise_slider(model_choice):
    """Show/hide denoise slider based on model selection."""
    supports_denoise = MODEL_OPTIONS.get(model_choice, {}).get("supports_denoise", False)
    return gr.update(visible=supports_denoise)


def create_app():
    """Create and configure the Gradio application."""

    with gr.Blocks(
        title="FaceOff - Face Swapper",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    ) as demo:
        gr.HTML('<h1 class="header-text">👤 FaceOff</h1>')
        gr.Markdown('<div class="header-subtitle">AI Face Swapper - Advanced face swapping with enhancement options</div>')
        
        with gr.Tabs():
            # Create Image Tab
            img_components = create_image_tab()
            
            # Create GIF Tab
            gif_components = create_gif_tab()
            
            # Create Video Tab
            vid_components = create_video_tab()
            
            # Create Gallery Tab
            gallery_components = create_gallery_tab()
        
        # Preset Management Section (at bottom, collapsible)
        gr.Markdown("---")
        with gr.Accordion("🎨 Preset Management", open=False):
            gr.HTML('<div class="face-swap-box"><strong>💾 Save & Load Settings</strong><br/>Manage your favorite face-swap configurations for quick access</div>')
            
            with gr.Row():
                with gr.Column(scale=2):
                    preset_dropdown = gr.Dropdown(
                        choices=get_preset_choices(),
                        value=get_default_preset(),
                        label="Available Presets",
                        info="Select a preset to view or load"
                    )
                    
                    preset_info_display = gr.Markdown(
                        value=get_preset_info_text(get_default_preset()),
                        label="Preset Information"
                    )
                
                with gr.Column(scale=1):
                    with gr.Row():
                        load_preset_btn = gr.Button("📥 Load Preset", variant="primary", size="sm")
                        delete_preset_btn = gr.Button("🗑️ Delete Preset", variant="secondary", size="sm")
                    
                    gr.Markdown("**Save Current Settings**")
                    preset_name_input = gr.Textbox(
                        label="New Preset Name",
                        placeholder="My Custom Preset",
                        max_lines=1
                    )
                    save_preset_btn = gr.Button("💾 Save as Preset", variant="primary", size="sm")
                    
                    preset_status = gr.Textbox(
                        label="Status",
                        value="",
                        interactive=False,
                        max_lines=1
                    )
        
        # GPU Memory Monitor (at bottom, collapsible)
        gr.Markdown("---")
        with gr.Accordion("🖥️ GPU Status Monitor", open=False):
            gr.HTML('<div class="face-swap-box"><strong>⚡ System Performance</strong><br/>Monitor GPU memory usage and allocation</div>')
            initial_gpu_info = get_gpu_status()
            gpu_textboxes = []
            
            with gr.Row():
                for idx, info in enumerate(initial_gpu_info):
                    gpu_textbox = gr.Textbox(
                        label=f"GPU {idx}" if len(initial_gpu_info) > 1 else "GPU Status",
                        value=info,
                        lines=4,
                        interactive=False
                    )
                    gpu_textboxes.append(gpu_textbox)
            
            refresh_gpu_btn = gr.Button("🔄 Refresh GPU Info", size="sm")
        
        # ============= EVENT HANDLERS =============
        
        # IMAGE TAB - Enhancement controls visibility
        img_components["enhance_toggle"].change(
            toggle_enhancement_controls,
            inputs=[img_components["enhance_toggle"]],
            outputs=[img_components["model_row"], img_components["enhancement_options_row"], img_components["enhancement_advanced_row"]]
        )
        
        img_components["restore_faces_toggle"].change(
            toggle_restoration_controls,
            inputs=[img_components["restore_faces_toggle"]],
            outputs=[img_components["restoration_row"]]
        )
        
        img_components["model_selector"].change(
            toggle_denoise_slider,
            inputs=[img_components["model_selector"]],
            outputs=[img_components["denoise_slider"]]
        )
        
        # Image face detection on upload
        img_components["source_img"].change(
            detect_faces_simple,
            inputs=[img_components["source_img"], img_components["face_confidence"]],
            outputs=[img_components["face_info_img"]]
        )
        img_components["face_confidence"].change(
            detect_faces_simple,
            inputs=[img_components["source_img"], img_components["face_confidence"]],
            outputs=[img_components["face_info_img"]]
        )
        
        # Image face mapping
        img_components["detect_faces_btn"].click(
            detect_faces_for_mapping,
            inputs=[img_components["source_img"], img_components["target_img"], img_components["face_confidence"]],
            outputs=[
                img_components["source_faces_gallery"],
                img_components["target_faces_gallery"],
                img_components["face_mapping_status"],
                img_components["mapping_source_idx"],
                img_components["mapping_target_idx"],
                img_components["source_faces_gallery"],
                img_components["target_faces_gallery"]
            ]
        )
        
        img_components["add_mapping_btn"].click(
            add_face_mapping_wrapper,
            inputs=[img_components["mapping_source_idx"], img_components["mapping_target_idx"], img_components["current_mappings"]],
            outputs=[img_components["face_mapping_status"], img_components["current_mappings"]]
        )
        
        img_components["clear_mappings_btn"].click(
            clear_face_mappings_wrapper,
            outputs=[img_components["face_mapping_status"], img_components["current_mappings"]]
        )
        
        # Image processing button
        img_components["run_image_btn"].click(
            process_image,
            inputs=[
                img_components["source_img"], img_components["target_img"], img_components["enhance_toggle"],
                img_components["face_confidence"], img_components["gpu_selection"], img_components["model_selector"],
                img_components["denoise_slider"], img_components["tile_size_slider"], img_components["outscale_slider"],
                img_components["use_fp32_checkbox"], img_components["pre_pad_slider"], img_components["restore_faces_toggle"],
                img_components["restoration_weight_slider"]
            ],
            outputs=[img_components["result_img"]],
            show_progress='full'
        )
        
        # GIF TAB - Enhancement controls
        gif_components["enhance_toggle_gif"].change(
            toggle_enhancement_controls,
            inputs=[gif_components["enhance_toggle_gif"]],
            outputs=[gif_components["model_row_gif"], gif_components["enhancement_options_row_gif"], gif_components["enhancement_advanced_row_gif"]]
        )
        
        gif_components["restore_faces_toggle_gif"].change(
            toggle_restoration_controls,
            inputs=[gif_components["restore_faces_toggle_gif"]],
            outputs=[gif_components["restoration_row_gif"]]
        )
        
        gif_components["model_selector_gif"].change(
            toggle_denoise_slider,
            inputs=[gif_components["model_selector_gif"]],
            outputs=[gif_components["denoise_slider_gif"]]
        )
        
        # GIF face detection
        gif_components["source_gif"].change(
            detect_faces_simple,
            inputs=[gif_components["source_gif"], gif_components["face_confidence_gif"]],
            outputs=[gif_components["face_info_gif"]]
        )
        gif_components["face_confidence_gif"].change(
            detect_faces_simple,
            inputs=[gif_components["source_gif"], gif_components["face_confidence_gif"]],
            outputs=[gif_components["face_info_gif"]]
        )
        
        # GIF preview
        gif_components["target_gif_file"].change(
            show_gif_preview,
            inputs=[gif_components["target_gif_file"]],
            outputs=[gif_components["target_gif_preview"]]
        )
        
        # GIF face mapping
        gif_components["detect_faces_btn_gif"].click(
            detect_faces_with_thumbnails,
            inputs=[gif_components["source_gif"], gif_components["target_gif_file"], gif_components["face_confidence_gif"]],
            outputs=[
                gif_components["source_faces_gallery_gif"],
                gif_components["target_faces_gallery_gif"],
                gif_components["face_mapping_status_gif"],
                gif_components["mapping_source_idx_gif"],
                gif_components["mapping_target_idx_gif"],
                gif_components["source_faces_gallery_gif"],
                gif_components["target_faces_gallery_gif"]
            ]
        )
        
        gif_components["add_mapping_btn_gif"].click(
            add_face_mapping_wrapper,
            inputs=[gif_components["mapping_source_idx_gif"], gif_components["mapping_target_idx_gif"], gif_components["current_mappings_gif"]],
            outputs=[gif_components["face_mapping_status_gif"], gif_components["current_mappings_gif"]]
        )
        
        gif_components["clear_mappings_btn_gif"].click(
            clear_face_mappings_wrapper,
            outputs=[gif_components["face_mapping_status_gif"], gif_components["current_mappings_gif"]]
        )
        
        # GIF processing button
        gif_components["run_gif_btn"].click(
            process_gif,
            inputs=[
                gif_components["source_gif"], gif_components["target_gif_file"], gif_components["enhance_toggle_gif"],
                gif_components["face_confidence_gif"], gif_components["gpu_selection_gif"], gif_components["model_selector_gif"],
                gif_components["denoise_slider_gif"], gif_components["tile_size_slider_gif"], gif_components["outscale_slider_gif"],
                gif_components["use_fp32_checkbox_gif"], gif_components["pre_pad_slider_gif"], gif_components["restore_faces_toggle_gif"],
                gif_components["restoration_weight_slider_gif"]
            ],
            outputs=[gif_components["result_gif"]],
            show_progress='full'
        )
        
        # VIDEO TAB - Enhancement controls
        vid_components["enhance_toggle_vid"].change(
            toggle_enhancement_controls,
            inputs=[vid_components["enhance_toggle_vid"]],
            outputs=[vid_components["model_row_vid"], vid_components["enhancement_options_row_vid"], vid_components["enhancement_advanced_row_vid"]]
        )
        
        vid_components["restore_faces_toggle_vid"].change(
            toggle_restoration_controls,
            inputs=[vid_components["restore_faces_toggle_vid"]],
            outputs=[vid_components["restoration_row_vid"]]
        )
        
        vid_components["model_selector_vid"].change(
            toggle_denoise_slider,
            inputs=[vid_components["model_selector_vid"]],
            outputs=[vid_components["denoise_slider_vid"]]
        )
        
        # Video face detection
        vid_components["source_vid"].change(
            detect_faces_simple,
            inputs=[vid_components["source_vid"], vid_components["face_confidence_video"]],
            outputs=[vid_components["face_info_vid"]]
        )
        vid_components["face_confidence_video"].change(
            detect_faces_simple,
            inputs=[vid_components["source_vid"], vid_components["face_confidence_video"]],
            outputs=[vid_components["face_info_vid"]]
        )
        
        # Video face mapping
        vid_components["detect_faces_btn_vid"].click(
            detect_faces_with_thumbnails,
            inputs=[vid_components["source_vid"], vid_components["target_vid"], vid_components["face_confidence_video"]],
            outputs=[
                vid_components["source_faces_gallery_vid"],
                vid_components["target_faces_gallery_vid"],
                vid_components["face_mapping_status_vid"],
                vid_components["mapping_source_idx_vid"],
                vid_components["mapping_target_idx_vid"],
                vid_components["source_faces_gallery_vid"],
                vid_components["target_faces_gallery_vid"]
            ]
        )
        
        vid_components["add_mapping_btn_vid"].click(
            add_face_mapping_wrapper,
            inputs=[vid_components["mapping_source_idx_vid"], vid_components["mapping_target_idx_vid"], vid_components["current_mappings_vid"]],
            outputs=[vid_components["face_mapping_status_vid"], vid_components["current_mappings_vid"]]
        )
        
        vid_components["clear_mappings_btn_vid"].click(
            clear_face_mappings_wrapper,
            outputs=[vid_components["face_mapping_status_vid"], vid_components["current_mappings_vid"]]
        )
        
        # Video processing button
        vid_components["run_video_btn"].click(
            process_video,
            inputs=[
                vid_components["source_vid"], vid_components["target_vid"], vid_components["enhance_toggle_vid"],
                vid_components["face_confidence_video"], vid_components["gpu_selection_video"], vid_components["model_selector_vid"],
                vid_components["denoise_slider_vid"], vid_components["tile_size_slider_vid"], vid_components["outscale_slider_vid"],
                vid_components["use_fp32_checkbox_vid"], vid_components["pre_pad_slider_vid"], vid_components["restore_faces_toggle_vid"],
                vid_components["restoration_weight_slider_vid"]
            ],
            outputs=[vid_components["result_vid"]],
            show_progress='full'
        )
        
        # Preset Management Event Handlers
        
        # Update preset info display when dropdown changes
        preset_dropdown.change(
            fn=get_preset_info_text,
            inputs=[preset_dropdown],
            outputs=[preset_info_display]
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
            ]
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
                img_components["restoration_weight_slider"]
            ],
            outputs=[preset_status, preset_dropdown]
        ).then(
            fn=lambda name: "",
            inputs=[preset_name_input],
            outputs=[preset_name_input]
        )
        
        # Delete preset button
        delete_preset_btn.click(
            fn=delete_selected_preset,
            inputs=[preset_dropdown],
            outputs=[preset_status, preset_dropdown]
        )
        
        # Gallery Tab Event Handlers
        
        # Update gallery when media type changes
        gallery_components["media_type_radio"].change(
            fn=update_gallery,
            inputs=[gallery_components["media_type_radio"], gallery_components["limit_selector"]],
            outputs=[gallery_components["gallery"], gallery_components["file_count_text"]]
        )
        
        # Limit selector change
        gallery_components["limit_selector"].change(
            fn=update_gallery,
            inputs=[gallery_components["media_type_radio"], gallery_components["limit_selector"]],
            outputs=[gallery_components["gallery"], gallery_components["file_count_text"]]
        )
        
        # Refresh gallery button (clears cache and reloads)
        gallery_components["refresh_btn"].click(
            fn=refresh_gallery,
            inputs=[gallery_components["media_type_radio"], gallery_components["limit_selector"]],
            outputs=[gallery_components["gallery"], gallery_components["file_count_text"]]
        )
        
        # Gallery selection handler (show delete button when file selected)
        gallery_components["gallery"].select(
            fn=show_delete_controls,
            outputs=[
                gallery_components["selected_file_display"],
                gallery_components["delete_btn"],
                gallery_components["delete_status"]
            ]
        )
        
        # Delete button handler
        gallery_components["delete_btn"].click(
            fn=delete_and_refresh,
            inputs=[
                gallery_components["selected_file_display"],
                gallery_components["media_type_radio"],
                gallery_components["limit_selector"]
            ],
            outputs=[
                gallery_components["delete_status"],
                gallery_components["gallery"],
                gallery_components["file_count_text"],
                gallery_components["delete_btn"],
                gallery_components["selected_file_display"]
            ]
        )
        
        # GPU refresh handler
        refresh_gpu_btn.click(
            fn=refresh_gpu_info,
            outputs=gpu_textboxes
        )
    
    return demo
