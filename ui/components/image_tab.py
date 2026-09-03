"""Image tab component for face swapping."""

import gradio as gr

from processing.job_control import request_job_cancel
from ui.handlers.processing_handlers import (
    add_face_mapping_image,
    clear_face_mappings_image,
    process_image,
)
from ui.helpers.controls import (
    JUMP_TO_GALLERY_JS,
    fill_compare_view,
    refresh_gallery_images,
    swap_with_button_toggle,
    toggle_compare_view,
    toggle_denoise_slider,
    toggle_enhancement_controls,
    toggle_restoration_controls,
    update_model_choices,
)
from ui.helpers.face_detection import detect_faces_for_mapping, detect_faces_simple
from ui.helpers.face_mapping import (
    add_face_mapping,
    auto_map_single_pair,
    clear_face_mappings,
    pick_target_and_map,
    select_mapping_index,
)
from ui.helpers.gpu_utils import get_gpu_options
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


def create_image_tab():
    """Create the image face swapping tab."""

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                source_img = gr.Image(type="pil", label="Source Image")
                face_info_img = gr.Textbox(
                    label="Face Detection Info", lines=4, interactive=False
                )
                source_faces_gallery = gr.Gallery(
                    label="Source Faces", columns=4, height="auto", visible=False
                )
            with gr.Column():
                target_img = gr.Image(type="pil", label="Target Image")
                target_faces_gallery = gr.Gallery(
                    label="Target Faces", columns=4, height="auto", visible=False
                )
            with gr.Column():
                result_img = gr.Image(label="Swapped Result")
                compare_toggle = gr.Checkbox(
                    label="Compare with original",
                    value=False,
                    info="Show the original target next to the result",
                )
                compare_img = gr.Image(
                    label="Original Target (compare)",
                    visible=False,
                )

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

            detect_faces_btn = gr.Button(
                "🔍 Detect Faces", variant="primary", size="sm"
            )

            face_mapping_status = gr.Textbox(
                label="Mapping Status",
                value="Upload images to detect faces",
                interactive=False,
            )

            with gr.Row():
                mapping_source_idx = gr.Dropdown(
                    label="Source Face Index", choices=[], interactive=True
                )
                mapping_arrow = gr.Markdown("→")
                mapping_target_idx = gr.Dropdown(
                    label="Target Face Index", choices=[], interactive=True
                )

            current_mappings = gr.Textbox(
                label="Current Mappings",
                value="No mappings",
                interactive=False,
                lines=3,
            )

            with gr.Row():
                add_mapping_btn = gr.Button(
                    "➕ Add Mapping", size="sm", variant="secondary"
                )
                clear_mappings_btn = gr.Button(
                    "🗑️ Clear All Mappings", size="sm", variant="secondary"
                )

        with gr.Row():
            enhance_toggle = gr.Checkbox(label="Enable Enhancement", value=False)
            restore_faces_toggle = gr.Checkbox(
                label="Restore Faces",
                value=False,
                info="Enhance face quality after swapping",
            )

        with gr.Row():
            enhancement_model_selector = gr.Dropdown(
                choices=["RealESRGAN", "SwinIR", "HAT"],
                value="RealESRGAN",
                label="Enhancement Framework",
                info="RealESRGAN (fast) or SwinIR/HAT (transformer-based SR)",
                visible=False,
            )
            restoration_model_selector = gr.Dropdown(
                choices=["GFPGAN", "CodeFormer"],
                value="GFPGAN",
                label="Face Restoration Model",
                info="GFPGAN (fast) or CodeFormer (better fidelity control)",
                visible=False,
            )

        with gr.Row(visible=False) as restoration_row:
            restoration_weight_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Restoration Strength",
                info="0=original face, 1=fully restored",
            )

        with gr.Row(visible=False) as model_row:
            model_selector = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Enhancement Model",
                info="Select Real-ESRGAN model for enhancement",
            )
            denoise_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Denoise Strength",
                info="Only works with 'realesr-general-x4v3' model",
                visible=False,
            )

        with gr.Row(visible=False) as enhancement_options_row:
            tile_size_slider = gr.Slider(
                minimum=128,
                maximum=512,
                value=DEFAULT_TILE_SIZE,
                step=64,
                label="Tile Size",
                info="Lower = less VRAM usage, slower processing",
            )
            outscale_slider = gr.Slider(
                minimum=2,
                maximum=4,
                value=DEFAULT_OUTSCALE,
                step=1,
                label="Upscale Factor",
                info="2x or 4x upscaling",
            )

        with gr.Row(visible=False) as enhancement_advanced_row:
            use_fp32_checkbox = gr.Checkbox(
                label="High Precision (FP32)",
                value=DEFAULT_USE_FP32,
                info="Uses more VRAM but slightly better quality",
            )
            pre_pad_slider = gr.Slider(
                minimum=0,
                maximum=20,
                value=DEFAULT_PRE_PAD,
                step=5,
                label="Edge Padding",
                info="Reduces edge artifacts (0-20)",
            )

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                face_confidence = gr.Slider(
                    minimum=0.3,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Face Detection Confidence Threshold",
                    info="Higher = stricter face detection (0.5 recommended)",
                )
                gpu_selection = gr.Dropdown(
                    choices=get_gpu_options(),
                    value=get_gpu_options()[0] if get_gpu_options() else None,
                    label="GPU Selection",
                    info="Choose which GPU(s) to use for processing",
                )
            with gr.Row():
                tensorrt_fp16_checkbox = gr.Checkbox(
                    label="TensorRT FP16 Mode",
                    value=True,
                    info="~30% faster inference, minimal quality impact (requires TensorRT)",
                )

        with gr.Row():
            run_image_btn = gr.Button("🔁 Swap Image", variant="primary")
            cancel_image_btn = gr.Button("⛔ Cancel", variant="stop", size="sm")

        swap_error = gr.Markdown(visible=False)

    # Return all components needed for event binding
    return {
        "source_img": source_img,
        "target_img": target_img,
        "result_img": result_img,
        "compare_toggle": compare_toggle,
        "compare_img": compare_img,
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
        "enhancement_model_selector": enhancement_model_selector,
        "restoration_model_selector": restoration_model_selector,
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
        "tensorrt_fp16_checkbox": tensorrt_fp16_checkbox,
        "run_image_btn": run_image_btn,
        "cancel_image_btn": cancel_image_btn,
        "swap_error": swap_error,
    }


def wire_image_tab(img_components, gallery_components):
    """Bind all Image tab events (moved out of ui/app.py)."""
    # Enhancement controls visibility
    img_components["enhance_toggle"].change(
        toggle_enhancement_controls,
        inputs=[img_components["enhance_toggle"]],
        outputs=[
            img_components["model_row"],
            img_components["enhancement_options_row"],
            img_components["enhancement_advanced_row"],
            img_components["enhancement_model_selector"],
        ],
    )

    # Update model selector choices when enhancement framework changes
    img_components["enhancement_model_selector"].change(
        update_model_choices,
        inputs=[img_components["enhancement_model_selector"]],
        outputs=[img_components["model_selector"]],
    )

    img_components["restore_faces_toggle"].change(
        toggle_restoration_controls,
        inputs=[img_components["restore_faces_toggle"]],
        outputs=[
            img_components["restoration_row"],
            img_components["restoration_model_selector"],
        ],
    )

    img_components["model_selector"].change(
        toggle_denoise_slider,
        inputs=[img_components["model_selector"]],
        outputs=[img_components["denoise_slider"]],
    )

    # Compare-with-original toggle
    img_components["compare_toggle"].change(
        toggle_compare_view,
        inputs=[img_components["compare_toggle"]],
        outputs=[img_components["compare_img"]],
    )

    # Image face detection on upload
    img_components["source_img"].change(
        detect_faces_simple,
        inputs=[img_components["source_img"], img_components["face_confidence"]],
        outputs=[img_components["face_info_img"]],
    )
    img_components["face_confidence"].change(
        detect_faces_simple,
        inputs=[img_components["source_img"], img_components["face_confidence"]],
        outputs=[img_components["face_info_img"]],
    )

    # Image face mapping
    img_components["detect_faces_btn"].click(
        detect_faces_for_mapping,
        inputs=[
            img_components["source_img"],
            img_components["target_img"],
            img_components["face_confidence"],
        ],
        outputs=[
            img_components["source_faces_gallery"],
            img_components["target_faces_gallery"],
            img_components["face_mapping_status"],
            img_components["mapping_source_idx"],
            img_components["mapping_target_idx"],
            img_components["source_faces_gallery"],
            img_components["target_faces_gallery"],
        ],
    ).then(
        # One face each side: mapping is unambiguous, create it directly.
        lambda s, t, c: auto_map_single_pair(s, t, c, add_face_mapping_image),
        inputs=[
            img_components["source_faces_gallery"],
            img_components["target_faces_gallery"],
            img_components["current_mappings"],
        ],
        outputs=[
            img_components["face_mapping_status"],
            img_components["current_mappings"],
        ],
    )

    # Click a thumbnail to choose it; clicking a target maps immediately.
    img_components["source_faces_gallery"].select(
        lambda evt: select_mapping_index(evt, "Source Face"),
        outputs=[img_components["mapping_source_idx"]],
    )
    img_components["target_faces_gallery"].select(
        lambda evt, src, cur: pick_target_and_map(
            evt, src, cur, add_face_mapping_image, "Target Face"
        ),
        inputs=[
            img_components["mapping_source_idx"],
            img_components["current_mappings"],
        ],
        outputs=[
            img_components["mapping_target_idx"],
            img_components["face_mapping_status"],
            img_components["current_mappings"],
        ],
    )

    img_components["target_img"].change(
        clear_face_mappings_image,
        outputs=[
            img_components["face_mapping_status"],
            img_components["current_mappings"],
        ],
    )

    # A new source invalidates index-based mappings too.
    img_components["source_img"].change(
        clear_face_mappings_image,
        outputs=[
            img_components["face_mapping_status"],
            img_components["current_mappings"],
        ],
    )

    img_components["add_mapping_btn"].click(
        add_face_mapping_image,
        inputs=[
            img_components["mapping_source_idx"],
            img_components["mapping_target_idx"],
            img_components["current_mappings"],
        ],
        outputs=[
            img_components["face_mapping_status"],
            img_components["current_mappings"],
        ],
    )

    img_components["clear_mappings_btn"].click(
        clear_face_mappings_image,
        outputs=[
            img_components["face_mapping_status"],
            img_components["current_mappings"],
        ],
    )

    # Image processing button with auto-refresh of gallery
    img_components["run_image_btn"].click(
        swap_with_button_toggle(process_image),
        inputs=[
            img_components["source_img"],
            img_components["target_img"],
            img_components["enhance_toggle"],
            img_components["face_confidence"],
            img_components["gpu_selection"],
            img_components["model_selector"],
            img_components["denoise_slider"],
            img_components["tile_size_slider"],
            img_components["outscale_slider"],
            img_components["use_fp32_checkbox"],
            img_components["pre_pad_slider"],
            img_components["restore_faces_toggle"],
            img_components["restoration_weight_slider"],
            img_components["tensorrt_fp16_checkbox"],
            img_components["enhancement_model_selector"],
            img_components["restoration_model_selector"],
        ],
        outputs=[
            img_components["result_img"],
            img_components["run_image_btn"],
            img_components["swap_error"],
        ],
        show_progress="full",
    ).then(
        # Mirror the target for the compare-with-original view
        fill_compare_view,
        inputs=[img_components["target_img"]],
        outputs=[img_components["compare_img"]],
    ).then(
        # Auto-refresh gallery and jump to it after processing completes
        refresh_gallery_images,
        inputs=[img_components["result_img"]],
        outputs=[
            gallery_components["gallery"],
            gallery_components["file_count_text"],
        ],
        js=JUMP_TO_GALLERY_JS,
    )
    img_components["cancel_image_btn"].click(
        request_job_cancel,
        outputs=[],
    )
