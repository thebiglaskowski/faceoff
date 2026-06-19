"""
Tab factory for creating consistent media processing tabs.

This module provides factory functions for creating the common
UI components used across image, GIF, and video tabs.
"""

import gradio as gr
from typing import Dict, Any, Optional

from utils.constants import MODEL_OPTIONS, DEFAULT_MODEL
from ui.helpers.gpu_utils import get_gpu_choices


def create_source_section(tab_suffix: str = "") -> Dict[str, Any]:
    """
    Create the source image section common to all tabs.

    Args:
        tab_suffix: Suffix for component keys (e.g., "_gif", "_vid")

    Returns:
        Dictionary of Gradio components
    """
    with gr.Row():
        source_img = gr.Image(
            label="Source Face",
            type="pil",
            height=300,
            elem_id=f"source_img{tab_suffix}"
        )
        face_info = gr.Textbox(
            label="Detected Faces",
            interactive=False,
            lines=3,
            elem_id=f"face_info{tab_suffix}"
        )

    return {
        f"source{tab_suffix}": source_img,
        f"face_info{tab_suffix}": face_info,
    }


def create_enhancement_controls(tab_suffix: str = "") -> Dict[str, Any]:
    """
    Create enhancement control components.

    Args:
        tab_suffix: Suffix for component keys

    Returns:
        Dictionary of Gradio components
    """
    components = {}

    # Enhancement toggle
    with gr.Row():
        enhance_toggle = gr.Checkbox(
            label="Enable Enhancement",
            value=False,
            elem_id=f"enhance_toggle{tab_suffix}"
        )
        restore_faces_toggle = gr.Checkbox(
            label="Enable Face Restoration",
            value=False,
            elem_id=f"restore_faces_toggle{tab_suffix}"
        )

    components[f"enhance_toggle{tab_suffix}"] = enhance_toggle
    components[f"restore_faces_toggle{tab_suffix}"] = restore_faces_toggle

    # Model selection row (hidden by default)
    with gr.Row(visible=False) as model_row:
        model_selector = gr.Dropdown(
            choices=list(MODEL_OPTIONS.keys()),
            value=DEFAULT_MODEL,
            label="Enhancement Model",
            info="Select upscaling model",
            allow_custom_value=True,
        )
        denoise_slider = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.1,
            label="Denoise Strength",
            visible=False
        )

    components[f"model_row{tab_suffix}"] = model_row
    components[f"model_selector{tab_suffix}"] = model_selector
    components[f"denoise_slider{tab_suffix}"] = denoise_slider

    # Enhancement options row
    with gr.Row(visible=False) as enhancement_options_row:
        tile_size_slider = gr.Slider(
            minimum=64,
            maximum=512,
            value=256,
            step=64,
            label="Tile Size",
            info="Lower = less VRAM"
        )
        outscale_slider = gr.Slider(
            minimum=2,
            maximum=4,
            value=4,
            step=2,
            label="Output Scale",
            info="2x or 4x"
        )

    components[f"enhancement_options_row{tab_suffix}"] = enhancement_options_row
    components[f"tile_size_slider{tab_suffix}"] = tile_size_slider
    components[f"outscale_slider{tab_suffix}"] = outscale_slider

    # Advanced enhancement row
    with gr.Row(visible=False) as enhancement_advanced_row:
        use_fp32_checkbox = gr.Checkbox(
            label="Use FP32",
            value=False,
            info="More VRAM, slightly better quality"
        )
        pre_pad_slider = gr.Slider(
            minimum=0,
            maximum=20,
            value=0,
            step=1,
            label="Pre-Padding",
            info="Reduce edge artifacts"
        )

    components[f"enhancement_advanced_row{tab_suffix}"] = enhancement_advanced_row
    components[f"use_fp32_checkbox{tab_suffix}"] = use_fp32_checkbox
    components[f"pre_pad_slider{tab_suffix}"] = pre_pad_slider

    # Restoration row
    with gr.Row(visible=False) as restoration_row:
        restoration_weight_slider = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.1,
            label="Restoration Weight",
            info="0 = original, 1 = fully restored"
        )

    components[f"restoration_row{tab_suffix}"] = restoration_row
    components[f"restoration_weight_slider{tab_suffix}"] = restoration_weight_slider

    return components


def create_detection_controls(tab_suffix: str = "") -> Dict[str, Any]:
    """
    Create face detection control components.

    Args:
        tab_suffix: Suffix for component keys

    Returns:
        Dictionary of Gradio components
    """
    components = {}

    with gr.Row():
        face_confidence = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.5,
            step=0.1,
            label="Face Confidence",
            info="Higher = more strict detection"
        )
        gpu_selection = gr.Dropdown(
            choices=get_gpu_choices(),
            value=get_gpu_choices()[0] if get_gpu_choices() else None,
            label="GPU Selection"
        )

    components[f"face_confidence{tab_suffix}"] = face_confidence
    components[f"gpu_selection{tab_suffix}"] = gpu_selection

    return components


def create_face_mapping_section(tab_suffix: str = "") -> Dict[str, Any]:
    """
    Create face mapping UI components.

    Args:
        tab_suffix: Suffix for component keys

    Returns:
        Dictionary of Gradio components
    """
    components = {}

    with gr.Accordion("Face Mapping (Advanced)", open=False):
        gr.Markdown("Map specific source faces to target faces for multi-face swapping.")

        with gr.Row():
            detect_faces_btn = gr.Button(
                "Detect Faces",
                variant="secondary",
                scale=1
            )

        with gr.Row():
            source_faces_gallery = gr.Gallery(
                label="Source Faces",
                columns=4,
                height=150,
                allow_preview=False
            )
            target_faces_gallery = gr.Gallery(
                label="Target Faces",
                columns=4,
                height=150,
                allow_preview=False
            )

        with gr.Row():
            mapping_source_idx = gr.Number(
                label="Source Face Index",
                value=0,
                minimum=0,
                precision=0,
                scale=1
            )
            mapping_target_idx = gr.Number(
                label="Target Face Index",
                value=0,
                minimum=0,
                precision=0,
                scale=1
            )
            add_mapping_btn = gr.Button(
                "Add Mapping",
                variant="secondary",
                scale=1
            )
            clear_mappings_btn = gr.Button(
                "Clear All",
                variant="secondary",
                scale=1
            )

        with gr.Row():
            current_mappings = gr.Textbox(
                label="Current Mappings",
                value="No mappings",
                interactive=False,
                lines=3
            )
            face_mapping_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )

    components[f"detect_faces_btn{tab_suffix}"] = detect_faces_btn
    components[f"source_faces_gallery{tab_suffix}"] = source_faces_gallery
    components[f"target_faces_gallery{tab_suffix}"] = target_faces_gallery
    components[f"mapping_source_idx{tab_suffix}"] = mapping_source_idx
    components[f"mapping_target_idx{tab_suffix}"] = mapping_target_idx
    components[f"add_mapping_btn{tab_suffix}"] = add_mapping_btn
    components[f"clear_mappings_btn{tab_suffix}"] = clear_mappings_btn
    components[f"current_mappings{tab_suffix}"] = current_mappings
    components[f"face_mapping_status{tab_suffix}"] = face_mapping_status

    return components


def create_process_button(
    label: str = "Process",
    tab_suffix: str = ""
) -> gr.Button:
    """
    Create the main process button.

    Args:
        label: Button label
        tab_suffix: Suffix for component key

    Returns:
        Gradio Button component
    """
    return gr.Button(
        label,
        variant="primary",
        elem_classes=["primary-btn"],
        elem_id=f"process_btn{tab_suffix}"
    )
