"""
Gradio UI for FaceOff - AI Face Swapper application.
Refactored with modular components for clean architecture.
"""
import gradio as gr
import logging
import numpy as np
from pathlib import Path
from PIL import Image

from utils.logging_setup import setup_logging

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
from core.face_processor import FaceMappingManager
from utils.validation import (
    validate_file_size, validate_image_resolution,
    validate_video_duration, validate_gif_frames, validate_media_type
)
from utils.constants import MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_TILE_SIZE, DEFAULT_OUTSCALE, DEFAULT_USE_FP32, DEFAULT_PRE_PAD
from processing.orchestrator import process_media
from utils.preset_manager import PresetManager, initialize_default_presets
from utils.error_handler import ErrorHandler, FriendlyError

# UI Components and Helpers
from ui.components.image_tab import create_image_tab
from ui.components.gif_tab import create_gif_tab
from ui.components.video_tab import create_video_tab
from ui.components.gallery_tab import create_gallery_tab, update_gallery, refresh_gallery
from ui.helpers.gallery_utils import count_media_files, delete_file
from ui.helpers.gpu_utils import get_gpu_status, refresh_gpu_info
from ui.helpers.face_detection import detect_faces_simple, detect_faces_for_mapping, detect_faces_with_thumbnails
from ui.helpers.face_mapping import add_face_mapping as helper_add_mapping, clear_face_mappings as helper_clear_mappings
from ui.helpers.preview import show_gif_preview

# Set up logging
setup_logging()
logger = logging.getLogger("FaceOff")

# Global state for face mappings
face_mapping_manager = FaceMappingManager()

# Initialize preset manager
preset_manager = PresetManager()
initialize_default_presets(preset_manager)

# Create output directories on startup
output_base = Path("outputs")
output_base.mkdir(exist_ok=True)
(output_base / "image").mkdir(exist_ok=True)
(output_base / "gif").mkdir(exist_ok=True)
(output_base / "video").mkdir(exist_ok=True)
logger.info("Output directories initialized: outputs/image, outputs/gif, outputs/video")


def get_preset_choices():
    """Get list of available preset names for dropdown."""
    presets = preset_manager.list_presets()
    return [p['name'] for p in presets]


def get_default_preset():
    """Get the default preset name (Balanced if available, otherwise first in list)."""
    choices = get_preset_choices()
    if "Balanced" in choices:
        return "Balanced"
    return choices[0] if choices else None


def load_preset_settings(preset_name):
    """Load preset and return settings as tuple for updating UI controls."""
    if not preset_name:
        return [gr.update() for _ in range(8)]  # Return no updates
    
    try:
        settings = preset_manager.load_preset(preset_name)
        logger.info(f"Loaded preset: {preset_name}")
        
        # Get the model name from settings (check both 'model' and 'model_name' for compatibility)
        model_name = settings.get('model') or settings.get('model_name', DEFAULT_MODEL)
        logger.info(f"Preset model_name from settings: {model_name}")
        
        # The image_tab.py dropdowns use simple model names (e.g., "RealESRGAN_x4plus")
        # not display names, so we use the model_name directly
        logger.info(f"Using model name for dropdown: {model_name}")
        
        # Return updates for all relevant controls in order
        # enhance, restore_faces, model, tile_size, outscale, use_fp32, pre_pad, restoration_weight
        return [
            gr.update(value=settings.get('enhance', False)),
            gr.update(value=settings.get('restore_faces') or settings.get('restore', False)),
            gr.update(value=model_name),  # Use model_name directly, not display name
            gr.update(value=settings.get('tile_size', DEFAULT_TILE_SIZE)),
            gr.update(value=settings.get('outscale', DEFAULT_OUTSCALE)),
            gr.update(value=settings.get('use_fp32', DEFAULT_USE_FP32)),
            gr.update(value=settings.get('pre_pad', DEFAULT_PRE_PAD)),
            gr.update(value=settings.get('restoration_weight', 0.5)),
        ]
    except Exception as e:
        logger.error(f"Error loading preset: {e}")
        return [gr.update() for _ in range(8)]


def save_current_preset(preset_name, enhance, restore, model_display_name, tile_size, outscale, use_fp32, pre_pad, restoration_weight):
    """Save current settings as a new preset."""
    if not preset_name or not preset_name.strip():
        return gr.update(value="❌ Please enter a preset name"), gr.update()
    
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
            'restore_faces': restore,  # Use consistent field name
            'model_name': model_name,  # Store the actual model name, not display name
            'tile_size': tile_size,
            'outscale': outscale,
            'use_fp32': use_fp32,
            'pre_pad': pre_pad,
            'restoration_weight': restoration_weight,
        }
        
        preset_manager.save_preset(preset_name, settings, description="Custom user preset")
        logger.info(f"Saved preset: {preset_name}")
        
        # Update dropdown with new preset list and status message
        return (
            gr.update(value=f"✅ Saved preset: {preset_name}"),
            gr.update(choices=get_preset_choices(), value=preset_name)
        )
    except Exception as e:
        logger.error(f"Error saving preset: {e}")
        return gr.update(value=f"❌ Error: {str(e)}"), gr.update()


def delete_selected_preset(preset_name):
    """Delete the selected preset."""
    if not preset_name:
        return gr.update(value="❌ No preset selected"), gr.update()
    
    try:
        preset_manager.delete_preset(preset_name)
        logger.info(f"Deleted preset: {preset_name}")
        
        new_choices = get_preset_choices()
        return (
            gr.update(value=f"✅ Deleted preset: {preset_name}"),
            gr.update(choices=new_choices, value=new_choices[0] if new_choices else None)
        )
    except Exception as e:
        logger.error(f"Error deleting preset: {e}")
        return gr.update(value=f"❌ Error: {str(e)}"), gr.update()


def get_preset_info_text(preset_name):
    """Get preset information for display."""
    if not preset_name:
        return "No preset selected"
    
    try:
        info = preset_manager.get_preset_info(preset_name)
        description = info.get('description', 'No description')
        created = info.get('created', 'Unknown')
        settings = info.get('settings', {})
        
        # Get model name (check both 'model' and 'model_name')
        model_name = settings.get('model') or settings.get('model_name', 'N/A')
        
        # Get restore faces setting (check both field names)
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


def process_input(source_image, target_image_path=None, target_video_path=None, 
                 enhance=False, confidence=0.5, gpu_selection=None, 
                 face_mappings=None, model_selection=None, denoise_strength=0.5, 
                 tile_size=None, outscale=None, use_fp32=None, pre_pad=None,
                 restore_faces=False, restoration_weight=0.5):
    """
    Process input and route to appropriate media processing function.
    """
    import numpy as np
    
    logger.info("="*40)
    logger.info("Starting processing...")
    
    try:
        # Validate and prepare inputs
        if source_image is None:
            raise gr.Error("Please upload a source image.")
        
        # Determine media type and target path
        if target_image_path:
            target_path = target_image_path
            if isinstance(target_path, Image.Image):
                # Use timestamp-based unique filename
                import time
                timestamp = int(time.time() * 1000)
                temp_path = Path("inputs") / f"temp_target_{timestamp}.png"
                temp_path.parent.mkdir(exist_ok=True)
                target_path.save(temp_path)
                target_path = str(temp_path)
            media_type = "image"
        elif target_video_path:
            target_path = target_video_path.name if hasattr(target_video_path, 'name') else target_video_path
            media_type = validate_media_type(target_path)
        else:
            raise gr.Error("Please upload a target file.")
        
        # Validate files
        validate_file_size(target_path)
        
        if media_type == "image":
            validate_image_resolution(target_path)
        elif media_type == "video":
            validate_video_duration(target_path)
        elif media_type == "gif":
            validate_gif_frames(target_path)
        
        # Use provided enhancement settings or defaults
        if tile_size is None:
            tile_size = DEFAULT_TILE_SIZE
        if outscale is None:
            outscale = DEFAULT_OUTSCALE
        if use_fp32 is None:
            use_fp32 = DEFAULT_USE_FP32
        if pre_pad is None:
            pre_pad = DEFAULT_PRE_PAD
        
        # Parse model selection
        if model_selection and model_selection in MODEL_OPTIONS:
            model_name = MODEL_OPTIONS[model_selection]["model_name"]
        else:
            model_name = MODEL_OPTIONS[DEFAULT_MODEL]["model_name"]
        
        # Convert source to numpy array
        source_array = np.array(source_image)
        
        # Get face mappings from manager
        mappings = face_mapping_manager.get()
        logger.info("Face mapping manager returned: %s", mappings)
        
        # Process media
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Processing %s with enhancement=%s, model=%s, denoise=%.2f, confidence=%.2f, tile=%d, outscale=%d, fp32=%s, prepad=%d, restore=%s, weight=%.2f",
                   media_type, enhance, model_name, denoise_strength, confidence, tile_size, outscale, use_fp32, pre_pad, restore_faces, restoration_weight)
        
        result_img, result_vid = process_media(
            source_image=source_array,
            dest_path=target_path,
            media_type=media_type,
            output_dir=str(output_dir),
            enhance=enhance,
            tile_size=tile_size,
            outscale=outscale,
            face_confidence=confidence,
            gpu_selection=gpu_selection,
            face_mappings=mappings,
            model_name=model_name,
            denoise_strength=denoise_strength,
            use_fp32=use_fp32,
            pre_pad=pre_pad,
            restore_faces=restore_faces,
            restoration_weight=restoration_weight
        )
        
        logger.info("Processing complete!")
        logger.info("="*40)
        
        # Return appropriate result based on media type
        if media_type == "image":
            return result_img
        else:
            return result_vid
    
    except FriendlyError as fe:
        # Display friendly error message
        logger.error("="*40)
        raise gr.Error(fe.format_message())
    except gr.Error:
        raise
    except Exception as e:
        # Convert technical errors to friendly messages
        context = {
            'media_type': media_type if 'media_type' in locals() else 'unknown',
            'enhance': enhance,
            'tile_size': tile_size if tile_size else DEFAULT_TILE_SIZE,
            'outscale': outscale if outscale else DEFAULT_OUTSCALE,
            'restore_faces': restore_faces,
            'use_fp32': use_fp32 if use_fp32 else DEFAULT_USE_FP32
        }
        friendly_error = ErrorHandler.handle_error(e, context)
        logger.error("Processing error: %s", e, exc_info=True)
        logger.error("="*40)
        raise gr.Error(friendly_error.format_message())


def add_face_mapping_wrapper(source_idx, target_idx, current_mappings_text):
    """Wrapper to add face mapping using global manager."""
    return helper_add_mapping(source_idx, target_idx, current_mappings_text, face_mapping_manager)


def clear_face_mappings_wrapper():
    """Wrapper to clear face mappings using global manager."""
    return helper_clear_mappings(face_mapping_manager)


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
        def wrapped_process_image(src, tgt, enhance, confidence, gpu, model, denoise, tile, outscale, fp32, prepad, restore, weight):
            return process_input(src, tgt, None, enhance, confidence, gpu, None, model, denoise, tile, outscale, fp32, prepad, restore, weight)
        
        img_components["run_image_btn"].click(
            wrapped_process_image,
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
        def wrapped_process_gif(src, tgt, enhance, confidence, gpu, model, denoise, tile, outscale, fp32, prepad, restore, weight):
            return process_input(src, None, tgt, enhance, confidence, gpu, None, model, denoise, tile, outscale, fp32, prepad, restore, weight)
        
        gif_components["run_gif_btn"].click(
            wrapped_process_gif,
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
        def wrapped_process_video(src, tgt, enhance, confidence, gpu, model, denoise, tile, outscale, fp32, prepad, restore, weight):
            return process_input(src, None, tgt, enhance, confidence, gpu, None, model, denoise, tile, outscale, fp32, prepad, restore, weight)
        
        vid_components["run_video_btn"].click(
            wrapped_process_video,
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
        # IMPORTANT: Must update visibility FIRST, then values
        load_preset_btn.click(
            # Step 1: Update visibility for image tab based on preset settings
            fn=lambda preset: toggle_enhancement_controls(preset_manager.load_preset(preset).get('enhance', False) if preset else False),
            inputs=[preset_dropdown],
            outputs=[img_components["model_row"], img_components["enhancement_options_row"], img_components["enhancement_advanced_row"]]
        ).then(
            fn=lambda preset: toggle_restoration_controls(preset_manager.load_preset(preset).get('restore_faces', False) if preset else False),
            inputs=[preset_dropdown],
            outputs=[img_components["restoration_row"]]
        ).then(
            # Step 2: Now update the actual values (after visibility is set)
            fn=load_preset_settings,
            inputs=[preset_dropdown],
            outputs=[
                # Image tab controls
                img_components["enhance_toggle"],
                img_components["restore_faces_toggle"],
                img_components["model_selector"],
                img_components["tile_size_slider"],
                img_components["outscale_slider"],
                img_components["use_fp32_checkbox"],
                img_components["pre_pad_slider"],
                img_components["restoration_weight_slider"]
            ]
        ).then(
            # Step 3: Update visibility for GIF tab
            fn=lambda preset: toggle_enhancement_controls(preset_manager.load_preset(preset).get('enhance', False) if preset else False),
            inputs=[preset_dropdown],
            outputs=[gif_components["model_row_gif"], gif_components["enhancement_options_row_gif"], gif_components["enhancement_advanced_row_gif"]]
        ).then(
            fn=lambda preset: toggle_restoration_controls(preset_manager.load_preset(preset).get('restore_faces', False) if preset else False),
            inputs=[preset_dropdown],
            outputs=[gif_components["restoration_row_gif"]]
        ).then(
            # Step 4: Update GIF tab values
            fn=load_preset_settings,
            inputs=[preset_dropdown],
            outputs=[
                # GIF tab controls
                gif_components["enhance_toggle_gif"],
                gif_components["restore_faces_toggle_gif"],
                gif_components["model_selector_gif"],
                gif_components["tile_size_slider_gif"],
                gif_components["outscale_slider_gif"],
                gif_components["use_fp32_checkbox_gif"],
                gif_components["pre_pad_slider_gif"],
                gif_components["restoration_weight_slider_gif"]
            ]
        ).then(
            # Step 5: Update visibility for video tab
            fn=lambda preset: toggle_enhancement_controls(preset_manager.load_preset(preset).get('enhance', False) if preset else False),
            inputs=[preset_dropdown],
            outputs=[vid_components["model_row_vid"], vid_components["enhancement_options_row_vid"], vid_components["enhancement_advanced_row_vid"]]
        ).then(
            fn=lambda preset: toggle_restoration_controls(preset_manager.load_preset(preset).get('restore_faces', False) if preset else False),
            inputs=[preset_dropdown],
            outputs=[vid_components["restoration_row_vid"]]
        ).then(
            # Step 6: Update video tab values
            fn=load_preset_settings,
            inputs=[preset_dropdown],
            outputs=[
                # Video tab controls
                vid_components["enhance_toggle_vid"],
                vid_components["restore_faces_toggle_vid"],
                vid_components["model_selector_vid"],
                vid_components["tile_size_slider_vid"],
                vid_components["outscale_slider_vid"],
                vid_components["use_fp32_checkbox_vid"],
                vid_components["pre_pad_slider_vid"],
                vid_components["restoration_weight_slider_vid"]
            ]
        ).then(
            fn=lambda: "✅ Preset loaded successfully across all tabs!",
            outputs=[preset_status]
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
