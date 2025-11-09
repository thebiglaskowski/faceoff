"""GIF tab component for face swapping."""

import gradio as gr
from ui.helpers.gpu_utils import get_gpu_options
from ui.helpers.face_detection import detect_faces_simple, detect_faces_with_thumbnails
from ui.helpers.face_mapping import add_face_mapping, clear_face_mappings
from ui.helpers.preview import show_gif_preview


# Constants for enhancement models
MODEL_OPTIONS = {
    "RealESRGAN_x4plus": {"scale": 4, "supports_denoise": False},
    "RealESRNet_x4plus": {"scale": 4, "supports_denoise": False},
    "RealESRGAN_x4plus_anime_6B": {"scale": 4, "supports_denoise": False},
    "RealESRGAN_x2plus": {"scale": 2, "supports_denoise": False},
    "realesr-general-x4v3": {"scale": 4, "supports_denoise": True},
}

DEFAULT_MODEL = "RealESRGAN_x4plus"
DEFAULT_TILE_SIZE = 256
DEFAULT_OUTSCALE = 4
DEFAULT_USE_FP32 = False
DEFAULT_PRE_PAD = 10


def create_gif_tab():
    """Create the GIF face swapping tab."""
    
    with gr.Tab("GIF"):
        with gr.Row():
            with gr.Column():
                source_gif = gr.Image(type="pil", label="Source Image")
                face_info_gif = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                source_faces_gallery_gif = gr.Gallery(label="Source Faces", columns=4, height="auto", visible=False)
            with gr.Column():
                target_gif_file = gr.File(label="Target GIF", file_types=[".gif"])
                target_gif_preview = gr.Image(label="Preview", visible=False, show_label=True)
                target_faces_gallery_gif = gr.Gallery(label="Target Faces (First Frame)", columns=4, height="auto", visible=False)
            with gr.Column():
                result_gif = gr.Image(label="Swapped Result", show_label=True)
        
        # Face mapping controls for GIF
        with gr.Accordion("🎭 Face Mapping (Multi-Face Swap)", open=False):
            gr.Markdown("""
            **Instructions:**
            1. Upload source image and target GIF
            2. Click "Detect Faces" to preview faces (GIF: first frame analyzed)
            3. Add mappings for which source face goes to which target face
            4. Mappings apply to all frames in the GIF
            """)
            
            detect_faces_btn_gif = gr.Button("🔍 Detect Faces", variant="primary", size="sm")
            
            face_mapping_status_gif = gr.Textbox(
                label="Mapping Status",
                value="Upload images to detect faces",
                interactive=False
            )
            
            with gr.Row():
                mapping_source_idx_gif = gr.Dropdown(
                    label="Source Face Index",
                    choices=[],
                    interactive=True
                )
                mapping_arrow_gif = gr.Markdown("→")
                mapping_target_idx_gif = gr.Dropdown(
                    label="Target Face Index",
                    choices=[],
                    interactive=True
                )
            
            current_mappings_gif = gr.Textbox(
                label="Current Mappings",
                value="No mappings",
                interactive=False,
                lines=3
            )
            
            with gr.Row():
                add_mapping_btn_gif = gr.Button("➕ Add Mapping", size="sm", variant="secondary")
                clear_mappings_btn_gif = gr.Button("🗑️ Clear All Mappings", size="sm", variant="secondary")
        
        with gr.Row():
            enhance_toggle_gif = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
            restore_faces_toggle_gif = gr.Checkbox(
                label="Restore Faces (GFPGAN)", 
                value=False,
                info="Enhance face quality after swapping"
            )
        
        with gr.Row(visible=False) as restoration_row_gif:
            restoration_weight_slider_gif = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
                info="0=original face, 1=fully restored"
            )
        
        with gr.Row(visible=False) as model_row_gif:
            model_selector_gif = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Enhancement Model"
            )
            denoise_slider_gif = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                label="Denoise Strength", visible=False
            )
        
        with gr.Row(visible=False) as enhancement_options_row_gif:
            tile_size_slider_gif = gr.Slider(
                minimum=128, maximum=512, value=DEFAULT_TILE_SIZE, step=64,
                label="Tile Size", info="Lower = less VRAM usage"
            )
            outscale_slider_gif = gr.Slider(
                minimum=2, maximum=4, value=DEFAULT_OUTSCALE, step=1,
                label="Upscale Factor", info="2x or 4x upscaling"
            )
        
        with gr.Row(visible=False) as enhancement_advanced_row_gif:
            use_fp32_checkbox_gif = gr.Checkbox(
                label="High Precision (FP32)", value=DEFAULT_USE_FP32,
                info="Uses more VRAM but slightly better quality"
            )
            pre_pad_slider_gif = gr.Slider(
                minimum=0, maximum=20, value=DEFAULT_PRE_PAD, step=5,
                label="Edge Padding", info="Reduces edge artifacts"
            )
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                face_confidence_gif = gr.Slider(
                    minimum=0.3,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Face Detection Confidence Threshold",
                    info="Higher = stricter face detection (0.5 recommended)"
                )
                gpu_selection_gif = gr.Dropdown(
                    choices=get_gpu_options(),
                    value=get_gpu_options()[0] if get_gpu_options() else None,
                    label="GPU Selection",
                    info="Choose which GPU(s) to use for processing"
                )
        
        run_gif_btn = gr.Button("Run GIF Swap", variant="primary")
    
    # Return all components needed for event binding
    return {
        "source_gif": source_gif,
        "target_gif_file": target_gif_file,
        "target_gif_preview": target_gif_preview,
        "result_gif": result_gif,
        "face_info_gif": face_info_gif,
        "source_faces_gallery_gif": source_faces_gallery_gif,
        "target_faces_gallery_gif": target_faces_gallery_gif,
        "detect_faces_btn_gif": detect_faces_btn_gif,
        "face_mapping_status_gif": face_mapping_status_gif,
        "mapping_source_idx_gif": mapping_source_idx_gif,
        "mapping_target_idx_gif": mapping_target_idx_gif,
        "current_mappings_gif": current_mappings_gif,
        "add_mapping_btn_gif": add_mapping_btn_gif,
        "clear_mappings_btn_gif": clear_mappings_btn_gif,
        "enhance_toggle_gif": enhance_toggle_gif,
        "restore_faces_toggle_gif": restore_faces_toggle_gif,
        "restoration_row_gif": restoration_row_gif,
        "restoration_weight_slider_gif": restoration_weight_slider_gif,
        "model_row_gif": model_row_gif,
        "model_selector_gif": model_selector_gif,
        "denoise_slider_gif": denoise_slider_gif,
        "enhancement_options_row_gif": enhancement_options_row_gif,
        "tile_size_slider_gif": tile_size_slider_gif,
        "outscale_slider_gif": outscale_slider_gif,
        "enhancement_advanced_row_gif": enhancement_advanced_row_gif,
        "use_fp32_checkbox_gif": use_fp32_checkbox_gif,
        "pre_pad_slider_gif": pre_pad_slider_gif,
        "face_confidence_gif": face_confidence_gif,
        "gpu_selection_gif": gpu_selection_gif,
        "run_gif_btn": run_gif_btn,
    }
