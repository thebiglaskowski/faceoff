"""Image tab component for face swapping."""

import gradio as gr
from ui.helpers.gpu_utils import get_gpu_options
from ui.helpers.face_detection import detect_faces_simple, detect_faces_for_mapping
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


def create_image_tab():
    """Create the image face swapping tab."""
    
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                source_img = gr.Image(type="pil", label="Source Image")
                face_info_img = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                source_faces_gallery = gr.Gallery(label="Source Faces", columns=4, height="auto", visible=False)
            with gr.Column():
                target_img = gr.Image(type="pil", label="Target Image")
                target_faces_gallery = gr.Gallery(label="Target Faces", columns=4, height="auto", visible=False)
            with gr.Column():
                result_img = gr.Image(label="Swapped Result")
        
        # Face mapping controls
        with gr.Accordion("🎭 Face Mapping (Multi-Face Swap)", open=False):
            gr.Markdown("""
            **Instructions:**
            1. Upload both source and target images
            2. Click "Detect Faces" to preview detected faces
            3. Add mappings: Select which source face goes to which target face
            4. Run swap with your custom mappings
            
            *Leave empty to use default behavior (first source face → all target faces)*
            """)
            
            detect_faces_btn = gr.Button("🔍 Detect Faces", variant="primary", size="sm")
            
            face_mapping_status = gr.Textbox(
                label="Mapping Status",
                value="Upload images to detect faces",
                interactive=False
            )
            
            with gr.Row():
                mapping_source_idx = gr.Dropdown(
                    label="Source Face Index",
                    choices=[],
                    interactive=True
                )
                mapping_arrow = gr.Markdown("→")
                mapping_target_idx = gr.Dropdown(
                    label="Target Face Index",
                    choices=[],
                    interactive=True
                )
            
            current_mappings = gr.Textbox(
                label="Current Mappings",
                value="No mappings",
                interactive=False,
                lines=3
            )
            
            with gr.Row():
                add_mapping_btn = gr.Button("➕ Add Mapping", size="sm", variant="secondary")
                clear_mappings_btn = gr.Button("🗑️ Clear All Mappings", size="sm", variant="secondary")
        
        with gr.Row():
            enhance_toggle = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
            restore_faces_toggle = gr.Checkbox(
                label="Restore Faces (GFPGAN)", 
                value=False,
                info="Enhance face quality after swapping"
            )
        
        with gr.Row(visible=False) as restoration_row:
            restoration_weight_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
                info="0=original face, 1=fully restored"
            )
        
        with gr.Row(visible=False) as model_row:
            model_selector = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Enhancement Model",
                info="Select Real-ESRGAN model for enhancement"
            )
            denoise_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Denoise Strength",
                info="Only works with 'realesr-general-x4v3' model",
                visible=False
            )
        
        with gr.Row(visible=False) as enhancement_options_row:
            tile_size_slider = gr.Slider(
                minimum=128,
                maximum=512,
                value=DEFAULT_TILE_SIZE,
                step=64,
                label="Tile Size",
                info="Lower = less VRAM usage, slower processing"
            )
            outscale_slider = gr.Slider(
                minimum=2,
                maximum=4,
                value=DEFAULT_OUTSCALE,
                step=1,
                label="Upscale Factor",
                info="2x or 4x upscaling"
            )
        
        with gr.Row(visible=False) as enhancement_advanced_row:
            use_fp32_checkbox = gr.Checkbox(
                label="High Precision (FP32)",
                value=DEFAULT_USE_FP32,
                info="Uses more VRAM but slightly better quality"
            )
            pre_pad_slider = gr.Slider(
                minimum=0,
                maximum=20,
                value=DEFAULT_PRE_PAD,
                step=5,
                label="Edge Padding",
                info="Reduces edge artifacts (0-20)"
            )
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                face_confidence = gr.Slider(
                    minimum=0.3,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Face Detection Confidence Threshold",
                    info="Higher = stricter face detection (0.5 recommended)"
                )
                gpu_selection = gr.Dropdown(
                    choices=get_gpu_options(),
                    value=get_gpu_options()[0] if get_gpu_options() else None,
                    label="GPU Selection",
                    info="Choose which GPU(s) to use for processing"
                )
        
        run_image_btn = gr.Button("Run Image Swap", variant="primary")
    
    # Return all components needed for event binding
    return {
        "source_img": source_img,
        "target_img": target_img,
        "result_img": result_img,
        "face_info_img": face_info_img,
        "source_faces_gallery": source_faces_gallery,
        "target_faces_gallery": target_faces_gallery,
        "detect_faces_btn": detect_faces_btn,
        "face_mapping_status": face_mapping_status,
        "mapping_source_idx": mapping_source_idx,
        "mapping_target_idx": mapping_target_idx,
        "current_mappings": current_mappings,
        "add_mapping_btn": add_mapping_btn,
        "clear_mappings_btn": clear_mappings_btn,
        "enhance_toggle": enhance_toggle,
        "restore_faces_toggle": restore_faces_toggle,
        "restoration_row": restoration_row,
        "restoration_weight_slider": restoration_weight_slider,
        "model_row": model_row,
        "model_selector": model_selector,
        "denoise_slider": denoise_slider,
        "enhancement_options_row": enhancement_options_row,
        "tile_size_slider": tile_size_slider,
        "outscale_slider": outscale_slider,
        "enhancement_advanced_row": enhancement_advanced_row,
        "use_fp32_checkbox": use_fp32_checkbox,
        "pre_pad_slider": pre_pad_slider,
        "face_confidence": face_confidence,
        "gpu_selection": gpu_selection,
        "run_image_btn": run_image_btn,
    }
