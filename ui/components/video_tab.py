"""Video tab component for face swapping."""

import gradio as gr

from ui.handlers.processing_handlers import (
    add_face_mapping_video,
    clear_face_mappings_video,
    process_video,
)
from ui.helpers.controls import (
    refresh_gallery_videos,
    toggle_denoise_slider,
    toggle_enhancement_controls,
    toggle_restoration_controls,
    update_model_choices,
)
from ui.helpers.face_detection import detect_faces_simple, detect_faces_with_thumbnails
from ui.helpers.face_mapping import add_face_mapping, clear_face_mappings
from ui.helpers.gpu_utils import get_gpu_options
from ui.helpers.preview import validate_target_video_upload
from utils.constants import (
    DEFAULT_HAT_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OUTSCALE,
    DEFAULT_PRE_PAD,
    DEFAULT_SWINIR_MODEL,
    DEFAULT_TILE_SIZE,
    DEFAULT_USE_FP32,
    HAT_MODEL_OPTIONS,
    MODEL_OPTIONS,
    SWINIR_MODEL_OPTIONS,
)


def create_video_tab():
    """Create the video face swapping tab."""

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                source_vid = gr.Image(type="pil", label="Source Image")
                face_info_vid = gr.Textbox(
                    label="Face Detection Info", lines=4, interactive=False
                )
                source_faces_gallery_vid = gr.Gallery(
                    label="Source Faces", columns=4, height="auto", visible=False
                )
            with gr.Column():
                # File upload avoids Gradio's in-browser video decoder (HEVC / odd
                # profiles trigger "Video not playable"). First frame shown below.
                target_vid = gr.File(
                    label="Target Video",
                    file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"],
                )
                target_vid_preview = gr.Image(
                    label="First Frame Preview",
                    visible=False,
                    show_label=True,
                )
                target_faces_gallery_vid = gr.Gallery(
                    label="Target Faces (First Frame)",
                    columns=4,
                    height="auto",
                    visible=False,
                )
            with gr.Column():
                result_vid = gr.Video(
                    label="Swapped Result",
                    format="mp4",
                    autoplay=True,
                )

        # Face mapping controls for Video
        with gr.Accordion("🎭 Face Mapping (Multi-Face Swap)", open=False):
            gr.Markdown("""
            **Instructions:**
            1. Upload source image and target video
            2. Click "Detect Faces" to preview faces (Video: first frame analyzed)
            3. Add mappings for which source face goes to which target face
            4. Mappings apply to all frames in the video
            """)

            detect_faces_btn_vid = gr.Button(
                "🔍 Detect Faces", variant="primary", size="sm"
            )

            face_mapping_status_vid = gr.Textbox(
                label="Mapping Status",
                value="Upload images to detect faces",
                interactive=False,
            )

            with gr.Row():
                mapping_source_idx_vid = gr.Dropdown(
                    label="Source Face Index", choices=[], interactive=True
                )
                mapping_arrow_vid = gr.Markdown("→")
                mapping_target_idx_vid = gr.Dropdown(
                    label="Target Face Index", choices=[], interactive=True
                )

            current_mappings_vid = gr.Textbox(
                label="Current Mappings",
                value="No mappings",
                interactive=False,
                lines=3,
            )

            with gr.Row():
                add_mapping_btn_vid = gr.Button(
                    "➕ Add Mapping", size="sm", variant="secondary"
                )
                clear_mappings_btn_vid = gr.Button(
                    "🗑️ Clear All Mappings", size="sm", variant="secondary"
                )

        with gr.Row():
            enhance_toggle_vid = gr.Checkbox(label="Enable Enhancement", value=False)
            restore_faces_toggle_vid = gr.Checkbox(
                label="Restore Faces",
                value=False,
                info="Enhance face quality after swapping",
            )

        with gr.Row():
            enhancement_model_selector_vid = gr.Dropdown(
                choices=["RealESRGAN", "SwinIR", "HAT"],
                value="RealESRGAN",
                label="Enhancement Framework",
                info="RealESRGAN (fast) or SwinIR/HAT (transformer-based SR)",
                visible=False,
            )
            restoration_model_selector_vid = gr.Dropdown(
                choices=["GFPGAN", "CodeFormer"],
                value="GFPGAN",
                label="Face Restoration Model",
                info="GFPGAN (fast) or CodeFormer (better fidelity control)",
                visible=False,
            )

        with gr.Row(visible=False) as restoration_row_vid:
            restoration_weight_slider_vid = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
                info="0=original face, 1=fully restored",
            )

        with gr.Row(visible=False) as model_row_vid:
            model_selector_vid = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Enhancement Model",
            )
            denoise_slider_vid = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Denoise Strength",
                visible=False,
            )

        with gr.Row(visible=False) as enhancement_options_row_vid:
            tile_size_slider_vid = gr.Slider(
                minimum=128,
                maximum=512,
                value=DEFAULT_TILE_SIZE,
                step=64,
                label="Tile Size",
                info="Lower = less VRAM usage",
            )
            outscale_slider_vid = gr.Slider(
                minimum=2,
                maximum=4,
                value=DEFAULT_OUTSCALE,
                step=1,
                label="Upscale Factor",
                info="2x or 4x upscaling",
            )

        with gr.Row(visible=False) as enhancement_advanced_row_vid:
            use_fp32_checkbox_vid = gr.Checkbox(
                label="High Precision (FP32)",
                value=DEFAULT_USE_FP32,
                info="Uses more VRAM but slightly better quality",
            )
            pre_pad_slider_vid = gr.Slider(
                minimum=0,
                maximum=20,
                value=DEFAULT_PRE_PAD,
                step=5,
                label="Edge Padding",
                info="Reduces edge artifacts",
            )

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                face_confidence_video = gr.Slider(
                    minimum=0.3,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Face Detection Confidence Threshold",
                    info="Higher = stricter face detection (0.5 recommended)",
                )
                gpu_selection_video = gr.Dropdown(
                    choices=get_gpu_options(),
                    value=get_gpu_options()[0] if get_gpu_options() else None,
                    label="GPU Selection",
                    info="Choose which GPU(s) to use for processing",
                )
            with gr.Row():
                tensorrt_fp16_checkbox_vid = gr.Checkbox(
                    label="TensorRT FP16 Mode",
                    value=True,
                    info="~30% faster inference, minimal quality impact (requires TensorRT)",
                )

        run_video_btn = gr.Button("Run Video Swap", variant="primary")

    # Return all components needed for event binding
    return {
        "source_vid": source_vid,
        "target_vid": target_vid,
        "target_vid_preview": target_vid_preview,
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
        "enhancement_model_selector_vid": enhancement_model_selector_vid,
        "restoration_model_selector_vid": restoration_model_selector_vid,
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
        "tensorrt_fp16_checkbox_vid": tensorrt_fp16_checkbox_vid,
        "run_video_btn": run_video_btn,
    }


def wire_video_tab(vid_components, gallery_components):
    """Bind all Video tab events (moved out of ui/app.py)."""
    vid_components["enhance_toggle_vid"].change(
        toggle_enhancement_controls,
        inputs=[vid_components["enhance_toggle_vid"]],
        outputs=[
            vid_components["model_row_vid"],
            vid_components["enhancement_options_row_vid"],
            vid_components["enhancement_advanced_row_vid"],
            vid_components["enhancement_model_selector_vid"],
        ],
    )

    # Update model selector choices when enhancement framework changes
    vid_components["enhancement_model_selector_vid"].change(
        update_model_choices,
        inputs=[vid_components["enhancement_model_selector_vid"]],
        outputs=[vid_components["model_selector_vid"]],
    )

    vid_components["restore_faces_toggle_vid"].change(
        toggle_restoration_controls,
        inputs=[vid_components["restore_faces_toggle_vid"]],
        outputs=[
            vid_components["restoration_row_vid"],
            vid_components["restoration_model_selector_vid"],
        ],
    )

    vid_components["model_selector_vid"].change(
        toggle_denoise_slider,
        inputs=[vid_components["model_selector_vid"]],
        outputs=[vid_components["denoise_slider_vid"]],
    )

    # Video face detection
    vid_components["source_vid"].change(
        detect_faces_simple,
        inputs=[
            vid_components["source_vid"],
            vid_components["face_confidence_video"],
        ],
        outputs=[vid_components["face_info_vid"]],
    )
    vid_components["face_confidence_video"].change(
        detect_faces_simple,
        inputs=[
            vid_components["source_vid"],
            vid_components["face_confidence_video"],
        ],
        outputs=[vid_components["face_info_vid"]],
    )

    vid_components["target_vid"].change(
        validate_target_video_upload,
        inputs=[vid_components["target_vid"]],
        outputs=[
            vid_components["face_mapping_status_vid"],
            vid_components["current_mappings_vid"],
            vid_components["target_vid_preview"],
        ],
    )

    # Video face mapping
    vid_components["detect_faces_btn_vid"].click(
        detect_faces_with_thumbnails,
        inputs=[
            vid_components["source_vid"],
            vid_components["target_vid"],
            vid_components["face_confidence_video"],
        ],
        outputs=[
            vid_components["source_faces_gallery_vid"],
            vid_components["target_faces_gallery_vid"],
            vid_components["face_mapping_status_vid"],
            vid_components["mapping_source_idx_vid"],
            vid_components["mapping_target_idx_vid"],
            vid_components["source_faces_gallery_vid"],
            vid_components["target_faces_gallery_vid"],
        ],
    )

    vid_components["add_mapping_btn_vid"].click(
        add_face_mapping_video,
        inputs=[
            vid_components["mapping_source_idx_vid"],
            vid_components["mapping_target_idx_vid"],
            vid_components["current_mappings_vid"],
        ],
        outputs=[
            vid_components["face_mapping_status_vid"],
            vid_components["current_mappings_vid"],
        ],
    )

    vid_components["clear_mappings_btn_vid"].click(
        clear_face_mappings_video,
        outputs=[
            vid_components["face_mapping_status_vid"],
            vid_components["current_mappings_vid"],
        ],
    )

    # A new source invalidates index-based mappings too.
    vid_components["source_vid"].change(
        clear_face_mappings_video,
        outputs=[
            vid_components["face_mapping_status_vid"],
            vid_components["current_mappings_vid"],
        ],
    )

    # Video processing button with auto-refresh of gallery
    vid_components["run_video_btn"].click(
        process_video,
        inputs=[
            vid_components["source_vid"],
            vid_components["target_vid"],
            vid_components["enhance_toggle_vid"],
            vid_components["face_confidence_video"],
            vid_components["gpu_selection_video"],
            vid_components["model_selector_vid"],
            vid_components["denoise_slider_vid"],
            vid_components["tile_size_slider_vid"],
            vid_components["outscale_slider_vid"],
            vid_components["use_fp32_checkbox_vid"],
            vid_components["pre_pad_slider_vid"],
            vid_components["restore_faces_toggle_vid"],
            vid_components["restoration_weight_slider_vid"],
            vid_components["tensorrt_fp16_checkbox_vid"],
            vid_components["enhancement_model_selector_vid"],
            vid_components["restoration_model_selector_vid"],
        ],
        outputs=[vid_components["result_vid"]],
        show_progress="full",
    ).then(
        # Auto-refresh gallery after processing completes
        refresh_gallery_videos,
        inputs=[vid_components["result_vid"]],
        outputs=[
            gallery_components["gallery"],
            gallery_components["file_count_text"],
        ],
    )
