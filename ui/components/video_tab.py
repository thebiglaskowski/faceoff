"""Video tab component for face swapping."""

import gradio as gr
from ui.helpers.gpu_utils import get_gpu_options
from ui.helpers.face_detection import detect_faces_simple, detect_faces_with_thumbnails
from ui.helpers.face_mapping import add_face_mapping, clear_face_mappings


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


def create_video_tab():
    """Create the video face swapping tab."""
    
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                source_vid = gr.Image(type="pil", label="Source Image")
                face_info_vid = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                source_faces_gallery_vid = gr.Gallery(label="Source Faces", columns=4, height="auto", visible=False)
            with gr.Column():
                target_vid = gr.Video(label="Target Video", sources=None, autoplay=True)
                target_faces_gallery_vid = gr.Gallery(label="Target Faces (First Frame)", columns=4, height="auto", visible=False)
            with gr.Column():
                result_vid = gr.Video(label="Swapped Result", autoplay=True)
        
        # Face mapping controls for Video
        with gr.Accordion("🎭 Face Mapping (Multi-Face Swap)", open=False):
            gr.Markdown("""
            **Instructions:**
            1. Upload source image and target video
            2. Click "Detect Faces" to preview faces (Video: first frame analyzed)
            3. Add mappings for which source face goes to which target face
            4. Mappings apply to all frames in the video
            """)
            
            detect_faces_btn_vid = gr.Button("🔍 Detect Faces", variant="primary", size="sm")
            
            face_mapping_status_vid = gr.Textbox(
                label="Mapping Status",
                value="Upload images to detect faces",
                interactive=False
            )
            
            with gr.Row():
                mapping_source_idx_vid = gr.Dropdown(
                    label="Source Face Index",
                    choices=[],
                    interactive=True
                )
                mapping_arrow_vid = gr.Markdown("→")
                mapping_target_idx_vid = gr.Dropdown(
                    label="Target Face Index",
                    choices=[],
                    interactive=True
                )
            
            current_mappings_vid = gr.Textbox(
                label="Current Mappings",
                value="No mappings",
                interactive=False,
                lines=3
            )
            
            with gr.Row():
                add_mapping_btn_vid = gr.Button("➕ Add Mapping", size="sm", variant="secondary")
                clear_mappings_btn_vid = gr.Button("🗑️ Clear All Mappings", size="sm", variant="secondary")
        
        with gr.Row():
            enhance_toggle_vid = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
            restore_faces_toggle_vid = gr.Checkbox(
                label="Restore Faces (GFPGAN)", 
                value=False,
                info="Enhance face quality after swapping"
            )
        
        with gr.Row(visible=False) as restoration_row_vid:
            restoration_weight_slider_vid = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
                info="0=original face, 1=fully restored"
            )
        
        with gr.Row(visible=False) as model_row_vid:
            model_selector_vid = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Enhancement Model"
            )
            denoise_slider_vid = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                label="Denoise Strength", visible=False
            )
        
        with gr.Row(visible=False) as enhancement_options_row_vid:
            tile_size_slider_vid = gr.Slider(
                minimum=128, maximum=512, value=DEFAULT_TILE_SIZE, step=64,
                label="Tile Size", info="Lower = less VRAM usage"
            )
            outscale_slider_vid = gr.Slider(
                minimum=2, maximum=4, value=DEFAULT_OUTSCALE, step=1,
                label="Upscale Factor", info="2x or 4x upscaling"
            )
        
        with gr.Row(visible=False) as enhancement_advanced_row_vid:
            use_fp32_checkbox_vid = gr.Checkbox(
                label="High Precision (FP32)", value=DEFAULT_USE_FP32,
                info="Uses more VRAM but slightly better quality"
            )
            pre_pad_slider_vid = gr.Slider(
                minimum=0, maximum=20, value=DEFAULT_PRE_PAD, step=5,
                label="Edge Padding", info="Reduces edge artifacts"
            )
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                face_confidence_video = gr.Slider(
                    minimum=0.3,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Face Detection Confidence Threshold",
                    info="Higher = stricter face detection (0.5 recommended)"
                )
                gpu_selection_video = gr.Dropdown(
                    choices=get_gpu_options(),
                    value=get_gpu_options()[0] if get_gpu_options() else None,
                    label="GPU Selection",
                    info="Choose which GPU(s) to use for processing"
                )
        
        run_video_btn = gr.Button("Run Video Swap", variant="primary")
    
    # Return all components needed for event binding
    return {
        "source_vid": source_vid,
        "target_vid": target_vid,
        "result_vid": result_vid,
        "face_info_vid": face_info_vid,
        "source_faces_gallery_vid": source_faces_gallery_vid,
        "target_faces_gallery_vid": target_faces_gallery_vid,
        "detect_faces_btn_vid": detect_faces_btn_vid,
        "face_mapping_status_vid": face_mapping_status_vid,
        "mapping_source_idx_vid": mapping_source_idx_vid,
        "mapping_target_idx_vid": mapping_target_idx_vid,
        "current_mappings_vid": current_mappings_vid,
        "add_mapping_btn_vid": add_mapping_btn_vid,
        "clear_mappings_btn_vid": clear_mappings_btn_vid,
        "enhance_toggle_vid": enhance_toggle_vid,
        "restore_faces_toggle_vid": restore_faces_toggle_vid,
        "restoration_row_vid": restoration_row_vid,
        "restoration_weight_slider_vid": restoration_weight_slider_vid,
        "model_row_vid": model_row_vid,
        "model_selector_vid": model_selector_vid,
        "denoise_slider_vid": denoise_slider_vid,
        "enhancement_options_row_vid": enhancement_options_row_vid,
        "tile_size_slider_vid": tile_size_slider_vid,
        "outscale_slider_vid": outscale_slider_vid,
        "enhancement_advanced_row_vid": enhancement_advanced_row_vid,
        "use_fp32_checkbox_vid": use_fp32_checkbox_vid,
        "pre_pad_slider_vid": pre_pad_slider_vid,
        "face_confidence_video": face_confidence_video,
        "gpu_selection_video": gpu_selection_video,
        "run_video_btn": run_video_btn,
    }
