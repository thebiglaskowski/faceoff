"""GIF tab component for face swapping."""

import gradio as gr

from processing.job_control import request_job_cancel
from ui.handlers.processing_handlers import (
    add_face_mapping_gif,
    clear_face_mappings_gif,
    process_gif,
)
from ui.helpers.controls import (
    JUMP_TO_GALLERY_JS,
    fill_compare_view,
    refresh_gallery_gifs,
    swap_with_button_toggle,
    toggle_compare_view,
    toggle_denoise_slider,
    toggle_enhancement_controls,
    toggle_restoration_controls,
    update_model_choices,
)
from ui.helpers.face_detection import detect_faces_simple, detect_faces_with_thumbnails
from ui.helpers.face_mapping import (
    add_face_mapping,
    auto_map_single_pair,
    clear_face_mappings,
    pick_target_and_map,
    select_mapping_index,
)
from ui.helpers.gpu_utils import get_gpu_options
from ui.helpers.preview import show_gif_preview
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


def create_gif_tab():
    """Create the GIF face swapping tab."""

    with gr.Tab("GIF"):
        with gr.Row():
            with gr.Column():
                source_gif = gr.Image(type="pil", label="Source Image")
                face_info_gif = gr.Textbox(
                    label="Face Detection Info", lines=4, interactive=False
                )
                source_faces_gallery_gif = gr.Gallery(
                    label="Source Faces", columns=4, height="auto", visible=False
                )
            with gr.Column():
                target_gif_file = gr.File(label="Target GIF", file_types=[".gif"])
                target_gif_preview = gr.Image(
                    label="Preview", visible=False, show_label=True
                )
                target_faces_gallery_gif = gr.Gallery(
                    label="Target Faces (First Frame)",
                    columns=4,
                    height="auto",
                    visible=False,
                )
            with gr.Column():
                result_gif = gr.Image(label="Swapped Result", show_label=True)
                compare_toggle_gif = gr.Checkbox(
                    label="Compare with original",
                    value=False,
                    info="Show the original target next to the result",
                )
                compare_gif = gr.Image(
                    label="Original Target (compare)",
                    visible=False,
                    show_label=True,
                )

        # Face mapping controls for GIF
        with gr.Accordion("🎭 Face Mapping (Multi-Face Swap)", open=False):
            gr.Markdown("""
            **Instructions:**
            1. Upload source image and target GIF
            2. Click "Detect Faces" to preview faces (GIF: first frame analyzed)
            3. Add mappings for which source face goes to which target face
            4. Mappings apply to all frames in the GIF
            """)

            detect_faces_btn_gif = gr.Button(
                "🔍 Detect Faces", variant="primary", size="sm"
            )

            face_mapping_status_gif = gr.Textbox(
                label="Mapping Status",
                value="Upload images to detect faces",
                interactive=False,
            )

            with gr.Row():
                mapping_source_idx_gif = gr.Dropdown(
                    label="Source Face Index", choices=[], interactive=True
                )
                mapping_arrow_gif = gr.Markdown("→")
                mapping_target_idx_gif = gr.Dropdown(
                    label="Target Face Index", choices=[], interactive=True
                )

            current_mappings_gif = gr.Textbox(
                label="Current Mappings",
                value="No mappings",
                interactive=False,
                lines=3,
            )

            with gr.Row():
                add_mapping_btn_gif = gr.Button(
                    "➕ Add Mapping", size="sm", variant="secondary"
                )
                clear_mappings_btn_gif = gr.Button(
                    "🗑️ Clear All Mappings", size="sm", variant="secondary"
                )

        with gr.Row():
            enhance_toggle_gif = gr.Checkbox(label="Enable Enhancement", value=False)
            restore_faces_toggle_gif = gr.Checkbox(
                label="Restore Faces",
                value=False,
                info="Enhance face quality after swapping",
            )

        with gr.Row():
            enhancement_model_selector_gif = gr.Dropdown(
                choices=["RealESRGAN", "SwinIR", "HAT"],
                value="RealESRGAN",
                label="Enhancement Framework",
                info="RealESRGAN (fast) or SwinIR/HAT (transformer-based SR)",
                visible=False,
            )
            restoration_model_selector_gif = gr.Dropdown(
                choices=["GFPGAN", "CodeFormer"],
                value="GFPGAN",
                label="Face Restoration Model",
                info="GFPGAN (fast) or CodeFormer (better fidelity control)",
                visible=False,
            )

        with gr.Row(visible=False) as restoration_row_gif:
            restoration_weight_slider_gif = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
                info="0=original face, 1=fully restored",
            )

        with gr.Row(visible=False) as model_row_gif:
            model_selector_gif = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Enhancement Model",
            )
            denoise_slider_gif = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Denoise Strength",
                visible=False,
            )

        with gr.Row(visible=False) as enhancement_options_row_gif:
            tile_size_slider_gif = gr.Slider(
                minimum=128,
                maximum=512,
                value=DEFAULT_TILE_SIZE,
                step=64,
                label="Tile Size",
                info="Lower = less VRAM usage",
            )
            outscale_slider_gif = gr.Slider(
                minimum=2,
                maximum=4,
                value=DEFAULT_OUTSCALE,
                step=1,
                label="Upscale Factor",
                info="2x or 4x upscaling",
            )

        with gr.Row(visible=False) as enhancement_advanced_row_gif:
            use_fp32_checkbox_gif = gr.Checkbox(
                label="High Precision (FP32)",
                value=DEFAULT_USE_FP32,
                info="Uses more VRAM but slightly better quality",
            )
            pre_pad_slider_gif = gr.Slider(
                minimum=0,
                maximum=20,
                value=DEFAULT_PRE_PAD,
                step=5,
                label="Edge Padding",
                info="Reduces edge artifacts",
            )

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                face_confidence_gif = gr.Slider(
                    minimum=0.3,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Face Detection Confidence Threshold",
                    info="Higher = stricter face detection (0.5 recommended)",
                )
                gpu_selection_gif = gr.Dropdown(
                    choices=get_gpu_options(),
                    value=get_gpu_options()[0] if get_gpu_options() else None,
                    label="GPU Selection",
                    info="Choose which GPU(s) to use for processing",
                )
            with gr.Row():
                tensorrt_fp16_checkbox_gif = gr.Checkbox(
                    label="TensorRT FP16 Mode",
                    value=True,
                    info="~30% faster inference, minimal quality impact (requires TensorRT)",
                )

        with gr.Row():
            run_gif_btn = gr.Button("🔁 Swap GIF", variant="primary")
            cancel_gif_btn = gr.Button("⛔ Cancel", variant="stop", size="sm")

        swap_error = gr.Markdown(visible=False)

    # Return all components needed for event binding
    return {
        "source_gif": source_gif,
        "target_gif_file": target_gif_file,
        "target_gif_preview": target_gif_preview,
        "result_gif": result_gif,
        "compare_toggle_gif": compare_toggle_gif,
        "compare_gif": compare_gif,
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
        "enhancement_model_selector_gif": enhancement_model_selector_gif,
        "restoration_model_selector_gif": restoration_model_selector_gif,
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
        "tensorrt_fp16_checkbox_gif": tensorrt_fp16_checkbox_gif,
        "run_gif_btn": run_gif_btn,
        "cancel_gif_btn": cancel_gif_btn,
        "swap_error": swap_error,
    }


def wire_gif_tab(gif_components, gallery_components):
    """Bind all GIF tab events (moved out of ui/app.py)."""
    gif_components["enhance_toggle_gif"].change(
        toggle_enhancement_controls,
        inputs=[gif_components["enhance_toggle_gif"]],
        outputs=[
            gif_components["model_row_gif"],
            gif_components["enhancement_options_row_gif"],
            gif_components["enhancement_advanced_row_gif"],
            gif_components["enhancement_model_selector_gif"],
        ],
    )

    # Update model selector choices when enhancement framework changes
    gif_components["enhancement_model_selector_gif"].change(
        update_model_choices,
        inputs=[gif_components["enhancement_model_selector_gif"]],
        outputs=[gif_components["model_selector_gif"]],
    )

    gif_components["restore_faces_toggle_gif"].change(
        toggle_restoration_controls,
        inputs=[gif_components["restore_faces_toggle_gif"]],
        outputs=[
            gif_components["restoration_row_gif"],
            gif_components["restoration_model_selector_gif"],
        ],
    )

    gif_components["model_selector_gif"].change(
        toggle_denoise_slider,
        inputs=[gif_components["model_selector_gif"]],
        outputs=[gif_components["denoise_slider_gif"]],
    )

    # Compare-with-original toggle
    gif_components["compare_toggle_gif"].change(
        toggle_compare_view,
        inputs=[gif_components["compare_toggle_gif"]],
        outputs=[gif_components["compare_gif"]],
    )

    # GIF face detection
    gif_components["source_gif"].change(
        detect_faces_simple,
        inputs=[
            gif_components["source_gif"],
            gif_components["face_confidence_gif"],
        ],
        outputs=[gif_components["face_info_gif"]],
    )
    gif_components["face_confidence_gif"].change(
        detect_faces_simple,
        inputs=[
            gif_components["source_gif"],
            gif_components["face_confidence_gif"],
        ],
        outputs=[gif_components["face_info_gif"]],
    )

    # GIF preview (clear stale mappings when target GIF changes)
    gif_components["target_gif_file"].change(
        show_gif_preview,
        inputs=[gif_components["target_gif_file"]],
        outputs=[gif_components["target_gif_preview"]],
    ).then(
        clear_face_mappings_gif,
        outputs=[
            gif_components["face_mapping_status_gif"],
            gif_components["current_mappings_gif"],
        ],
    )

    # GIF face mapping
    gif_components["detect_faces_btn_gif"].click(
        detect_faces_with_thumbnails,
        inputs=[
            gif_components["source_gif"],
            gif_components["target_gif_file"],
            gif_components["face_confidence_gif"],
        ],
        outputs=[
            gif_components["source_faces_gallery_gif"],
            gif_components["target_faces_gallery_gif"],
            gif_components["face_mapping_status_gif"],
            gif_components["mapping_source_idx_gif"],
            gif_components["mapping_target_idx_gif"],
            gif_components["source_faces_gallery_gif"],
            gif_components["target_faces_gallery_gif"],
        ],
    ).then(
        # One face each side: mapping is unambiguous, create it directly.
        lambda s, t, c: auto_map_single_pair(s, t, c, add_face_mapping_gif),
        inputs=[
            gif_components["source_faces_gallery_gif"],
            gif_components["target_faces_gallery_gif"],
            gif_components["current_mappings_gif"],
        ],
        outputs=[
            gif_components["face_mapping_status_gif"],
            gif_components["current_mappings_gif"],
        ],
    )

    gif_components["add_mapping_btn_gif"].click(
        add_face_mapping_gif,
        inputs=[
            gif_components["mapping_source_idx_gif"],
            gif_components["mapping_target_idx_gif"],
            gif_components["current_mappings_gif"],
        ],
        outputs=[
            gif_components["face_mapping_status_gif"],
            gif_components["current_mappings_gif"],
        ],
    )

    # Click a thumbnail to choose it; clicking a target maps immediately.
    gif_components["source_faces_gallery_gif"].select(
        lambda evt: select_mapping_index(evt, "Source Face"),
        outputs=[gif_components["mapping_source_idx_gif"]],
    )
    gif_components["target_faces_gallery_gif"].select(
        lambda evt, src, cur: pick_target_and_map(
            evt, src, cur, add_face_mapping_gif, "Target Face"
        ),
        inputs=[
            gif_components["mapping_source_idx_gif"],
            gif_components["current_mappings_gif"],
        ],
        outputs=[
            gif_components["mapping_target_idx_gif"],
            gif_components["face_mapping_status_gif"],
            gif_components["current_mappings_gif"],
        ],
    )

    gif_components["clear_mappings_btn_gif"].click(
        clear_face_mappings_gif,
        outputs=[
            gif_components["face_mapping_status_gif"],
            gif_components["current_mappings_gif"],
        ],
    )

    # A new source invalidates index-based mappings too.
    gif_components["source_gif"].change(
        clear_face_mappings_gif,
        outputs=[
            gif_components["face_mapping_status_gif"],
            gif_components["current_mappings_gif"],
        ],
    )

    # GIF processing button with auto-refresh of gallery
    gif_components["run_gif_btn"].click(
        swap_with_button_toggle(process_gif),
        inputs=[
            gif_components["source_gif"],
            gif_components["target_gif_file"],
            gif_components["enhance_toggle_gif"],
            gif_components["face_confidence_gif"],
            gif_components["gpu_selection_gif"],
            gif_components["model_selector_gif"],
            gif_components["denoise_slider_gif"],
            gif_components["tile_size_slider_gif"],
            gif_components["outscale_slider_gif"],
            gif_components["use_fp32_checkbox_gif"],
            gif_components["pre_pad_slider_gif"],
            gif_components["restore_faces_toggle_gif"],
            gif_components["restoration_weight_slider_gif"],
            gif_components["tensorrt_fp16_checkbox_gif"],
            gif_components["enhancement_model_selector_gif"],
            gif_components["restoration_model_selector_gif"],
        ],
        outputs=[
            gif_components["result_gif"],
            gif_components["run_gif_btn"],
            gif_components["swap_error"],
        ],
        show_progress="full",
    ).then(
        # Mirror the target for the compare-with-original view
        fill_compare_view,
        inputs=[gif_components["target_gif_file"]],
        outputs=[gif_components["compare_gif"]],
    ).then(
        # Auto-refresh gallery and jump to it after processing completes
        refresh_gallery_gifs,
        inputs=[gif_components["result_gif"]],
        outputs=[
            gallery_components["gallery"],
            gallery_components["file_count_text"],
        ],
        js=JUMP_TO_GALLERY_JS,
    )
    gif_components["cancel_gif_btn"].click(
        request_job_cancel,
        outputs=[],
    )
