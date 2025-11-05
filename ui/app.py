"""
Gradio UI for FaceOff - AI Face Swapper application.
"""
import gradio as gr
import logging
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

from logging_utils import setup_logging
from core.gpu_manager import GPUManager
from core.face_processor import FaceProcessor, FaceMappingManager
from utils.validation import (
    validate_file_size, validate_image_resolution,
    validate_video_duration, validate_gif_frames, validate_media_type
)
from utils.constants import QUALITY_PRESETS, MODEL_OPTIONS, DEFAULT_MODEL
from processing.orchestrator import process_media

# Set up logging
setup_logging()
logger = logging.getLogger("FaceOff")

# Global state for face mappings
face_mapping_manager = FaceMappingManager()


def get_gpu_options():
    """Get GPU dropdown options."""
    return GPUManager.get_available_gpus()


def get_gpu_status():
    """Get GPU memory status for all GPUs."""
    return GPUManager.get_memory_info()


def refresh_gpu_info():
    """Refresh GPU information display."""
    gpu_info_list = get_gpu_status()
    # Ensure we have at least one entry
    if not gpu_info_list:
        gpu_info_list = ["No GPU info available"]
    return gpu_info_list


def detect_faces_from_upload(img, confidence=0.5):
    """Detect faces when image is uploaded."""
    if img is None:
        return ""
    try:
        # Save temporarily
        temp_path = Path("inputs") / "temp_detect.png"
        temp_path.parent.mkdir(exist_ok=True)
        img.save(temp_path)
        
        # Use FaceProcessor for detection
        processor = FaceProcessor(device_id=0, confidence=confidence)
        info = processor.detect_faces_info(str(temp_path))
        
        temp_path.unlink()
        return info
    except Exception as e:
        logger.error("Face detection error: %s", e)
        return f"❌ Error: {str(e)}"


def detect_and_extract_faces_ui(image_path: str, confidence: float):
    """Detect and extract face thumbnails for UI display."""
    try:
        processor = FaceProcessor(device_id=0, confidence=confidence)
        return processor.detect_and_extract_faces(image_path)
    except Exception as e:
        logger.error("Face extraction error: %s", e)
        return [], f"❌ Error: {str(e)}"


def detect_faces_for_mapping(source_img, target_img, face_confidence):
    """Detect faces in both images for mapping UI."""
    if source_img is None or target_img is None:
        return (
            [], [],
            "Upload both images first",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as src_tmp:
        source_img.save(src_tmp.name)
        src_faces, src_info = detect_and_extract_faces_ui(src_tmp.name, face_confidence)
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tgt_tmp:
        target_img.save(tgt_tmp.name)
        tgt_faces, tgt_info = detect_and_extract_faces_ui(tgt_tmp.name, face_confidence)
    
    if not src_faces or not tgt_faces:
        status = f"⚠️ Detection failed - Source: {src_info}, Target: {tgt_info}"
        return (
            [], [],
            status,
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    src_choices = [f"Source Face {i}" for i in range(len(src_faces))]
    tgt_choices = [f"Target Face {i}" for i in range(len(tgt_faces))]
    status = f"✅ Detected {len(src_faces)} source face(s) and {len(tgt_faces)} target face(s)"
    
    return (
        src_faces,
        tgt_faces,
        status,
        gr.update(choices=src_choices, value=src_choices[0] if src_choices else None),
        gr.update(choices=tgt_choices, value=tgt_choices[0] if tgt_choices else None),
        gr.update(visible=True),
        gr.update(visible=True)
    )


def detect_faces_gif_video(source_img, target_file, face_confidence):
    """Detect faces in source image and first frame of GIF/Video."""
    if source_img is None or target_file is None:
        return (
            [], [],
            "Upload both source and target file first",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    from moviepy.editor import VideoFileClip
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as src_tmp:
        source_img.save(src_tmp.name)
        src_faces, src_info = detect_and_extract_faces_ui(src_tmp.name, face_confidence)
    
    target_path = target_file.name if hasattr(target_file, 'name') else target_file
    
    try:
        if target_path.lower().endswith('.gif'):
            gif = Image.open(target_path)
            first_frame = gif.convert('RGB')
        else:
            clip = VideoFileClip(target_path)
            first_frame_array = clip.get_frame(0)
            first_frame = Image.fromarray(first_frame_array)
            clip.close()
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as frame_tmp:
            first_frame.save(frame_tmp.name)
            tgt_faces, tgt_info = detect_and_extract_faces_ui(frame_tmp.name, face_confidence)
    
    except Exception as e:
        logger.error("Failed to extract first frame: %s", e)
        return (
            [], [],
            f"⚠️ Failed to extract first frame: {str(e)}",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    if not src_faces or not tgt_faces:
        status = f"⚠️ Detection failed - Source: {src_info}, Target: {tgt_info}"
        return (
            [], [],
            status,
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    src_choices = [f"Source Face {i}" for i in range(len(src_faces))]
    tgt_choices = [f"Target Face {i}" for i in range(len(tgt_faces))]
    status = f"✅ Detected {len(src_faces)} source face(s) and {len(tgt_faces)} target face(s)"
    
    return (
        src_faces,
        tgt_faces,
        status,
        gr.update(choices=src_choices, value=src_choices[0] if src_choices else None),
        gr.update(choices=tgt_choices, value=tgt_choices[0] if tgt_choices else None),
        gr.update(visible=True),
        gr.update(visible=True)
    )


def detect_faces_batch_source(source_img, face_confidence):
    """Detect faces in batch source image."""
    if source_img is None:
        return (
            [],
            "Upload source image first",
            gr.update(choices=[]),
            gr.update(visible=False)
        )
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as src_tmp:
        source_img.save(src_tmp.name)
        src_faces, src_info = detect_and_extract_faces_ui(src_tmp.name, face_confidence)
    
    if not src_faces:
        return (
            [],
            f"⚠️ No faces detected: {src_info}",
            gr.update(choices=[]),
            gr.update(visible=False)
        )
    
    src_choices = [f"Source Face {i}" for i in range(len(src_faces))]
    status = f"✅ Detected {len(src_faces)} source face(s)"
    
    return (
        src_faces,
        status,
        gr.update(choices=src_choices, value=src_choices[0] if src_choices else None),
        gr.update(visible=True)
    )


def add_face_mapping(source_idx, target_idx, current_mappings_text):
    """Add a face mapping."""
    if source_idx is None or target_idx is None:
        return "⚠️ Select both source and target faces", current_mappings_text
    
    src_idx = int(source_idx.split()[-1])
    tgt_idx = int(target_idx.split()[-1])
    
    face_mapping_manager.add(src_idx, tgt_idx)
    
    status = f"✅ Added mapping: Source {src_idx} → Target {tgt_idx}"
    mappings_text = face_mapping_manager.get_display_text()
    
    return status, mappings_text


def clear_face_mappings():
    """Clear all face mappings."""
    face_mapping_manager.clear()
    return "Mappings cleared", "No mappings"


def show_gif_preview(file):
    """Show GIF preview when uploaded."""
    if file is None:
        return gr.update(visible=False, value=None)
    file_path = file.name if hasattr(file, 'name') else file
    return gr.update(visible=True, value=file_path)


def parse_quality_preset(quality_str):
    """Parse quality preset string to tile_size and outscale."""
    preset = QUALITY_PRESETS.get(quality_str, QUALITY_PRESETS["Balanced (4x, Tile 256)"])
    return preset["tile_size"], preset["outscale"]


def process_input(source_image, target_image_path, target_video_path, enhance, quality, confidence, gpu_selection=None, face_mappings=None, model_selection=None, denoise_strength=0.5):
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
        
        # Parse quality settings
        tile_size, outscale = parse_quality_preset(quality)
        
        # Parse model selection
        if model_selection and model_selection in MODEL_OPTIONS:
            model_name = MODEL_OPTIONS[model_selection]["model_name"]
        else:
            model_name = MODEL_OPTIONS[DEFAULT_MODEL]["model_name"]
        
        # Convert source to numpy array
        source_array = np.array(source_image)
        
        # Get face mappings from manager
        mappings = face_mapping_manager.get()
        
        # Process media
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Processing %s with enhancement=%s, quality=%s, model=%s, denoise=%.2f, confidence=%.2f",
                   media_type, enhance, quality, model_name, denoise_strength, confidence)
        
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
            denoise_strength=denoise_strength
        )
        
        logger.info("Processing complete!")
        logger.info("="*40)
        
        # Return appropriate result based on media type
        if media_type == "image":
            return result_img
        else:
            return result_vid
    
    except gr.Error:
        raise
    except Exception as e:
        logger.error("Processing error: %s", e, exc_info=True)
        logger.error("="*40)
        raise gr.Error(f"Processing error: {e}")


def process_batch(source, targets, enhance, quality, confidence, compare, gpu):
    """
    Process multiple files in batch mode.
    """
    import shutil
    import zipfile
    from datetime import datetime
    
    if source is None or not targets:
        raise gr.Error("Please upload source image and target files.")
    
    try:
        output_dir = Path("outputs") / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        status_lines = [f"Processing {len(targets)} files..."]
        
        # Get face mappings
        mappings = face_mapping_manager.get()
        
        for idx, target_file in enumerate(targets, 1):
            try:
                target_path = target_file.name if hasattr(target_file, 'name') else target_file
                media_type = validate_media_type(target_path)
                
                status_lines.append(f"\n[{idx}/{len(targets)}] Processing {Path(target_path).name}...")
                
                tile_size, outscale = parse_quality_preset(quality)
                source_array = np.array(source)
                
                result_img, result_vid = process_media(
                    source_image=source_array,
                    dest_path=target_path,
                    media_type=media_type,
                    output_dir=str(output_dir),
                    enhance=enhance,
                    tile_size=tile_size,
                    outscale=outscale,
                    face_confidence=confidence,
                    gpu_selection=gpu,
                    face_mappings=mappings
                )
                
                result_path = result_img if result_img else result_vid
                results.append(result_path)
                if result_path:
                    status_lines.append(f"✅ Complete: {Path(result_path).name}")
                else:
                    status_lines.append("⚠️ No output generated")
                
            except Exception as e:
                status_lines.append(f"❌ Failed: {e}")
                logger.error("Batch item failed: %s", e)
        
        # Create ZIP file
        zip_path = output_dir / "results.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for result in results:
                if result and Path(result).exists():
                    zipf.write(result, Path(result).name)
        
        status_lines.append(f"\n✅ Batch complete! {len(results)} files processed.")
        
        return str(zip_path), "\n".join(status_lines)
    
    except Exception as e:
        logger.error("Batch processing failed: %s", e)
        raise gr.Error(f"Batch processing failed: {e}")


def create_app():
    """Create and configure the Gradio application."""
    
    with gr.Blocks(title="FaceOff - Face Swapper") as demo:
        gr.Markdown("## FaceOff - AI Face Swapper")
        gr.Markdown("Swap faces from a source image to destination images, GIFs, or videos with optional AI enhancement.")
        
        with gr.Tabs():
            # Image Tab
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
                    2. Preview detected faces below
                    3. Add mappings: Select which source face goes to which target face
                    4. Run swap with your custom mappings
                    
                    *Leave empty to use default behavior (first source face → all target faces)*
                    """)
                    
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
                        add_mapping_btn = gr.Button("Add Mapping", size="sm")
                    
                    current_mappings = gr.Textbox(
                        label="Current Mappings",
                        value="No mappings",
                        interactive=False,
                        lines=3
                    )
                    
                    with gr.Row():
                        clear_mappings_btn = gr.Button("Clear All Mappings", size="sm")
                        detect_faces_btn = gr.Button("Detect Faces", variant="secondary", size="sm")
                
                with gr.Row():
                    enhance_toggle = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                    quality_preset = gr.Dropdown(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Balanced (4x, Tile 256)",
                        label="Enhancement Quality",
                        visible=False
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
            
            # GIF Tab
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
                        add_mapping_btn_gif = gr.Button("Add Mapping", size="sm")
                    
                    current_mappings_gif = gr.Textbox(
                        label="Current Mappings",
                        value="No mappings",
                        interactive=False,
                        lines=3
                    )
                    
                    with gr.Row():
                        clear_mappings_btn_gif = gr.Button("Clear All Mappings", size="sm")
                        detect_faces_btn_gif = gr.Button("Detect Faces", variant="secondary", size="sm")
                
                with gr.Row():
                    enhance_toggle_gif = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                    quality_preset_gif = gr.Dropdown(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Balanced (4x, Tile 256)",
                        label="Enhancement Quality",
                        visible=False
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
            
            # Video Tab
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
                        add_mapping_btn_vid = gr.Button("Add Mapping", size="sm")
                    
                    current_mappings_vid = gr.Textbox(
                        label="Current Mappings",
                        value="No mappings",
                        interactive=False,
                        lines=3
                    )
                    
                    with gr.Row():
                        clear_mappings_btn_vid = gr.Button("Clear All Mappings", size="sm")
                        detect_faces_btn_vid = gr.Button("Detect Faces", variant="secondary", size="sm")
                
                with gr.Row():
                    enhance_toggle_vid = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                    quality_preset_vid = gr.Dropdown(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Balanced (4x, Tile 256)",
                        label="Enhancement Quality",
                        visible=False
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
            
            # Batch Processing Tab
            with gr.Tab("Batch Processing"):
                gr.Markdown("### Process Multiple Files at Once")
                gr.Markdown("Upload a single source face image and multiple target files for batch face swapping.")
                
                with gr.Row():
                    with gr.Column():
                        batch_source = gr.Image(type="pil", label="Source Face (Single)")
                        batch_face_info = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                        source_faces_gallery_batch = gr.Gallery(label="Source Faces", columns=4, height="auto", visible=False)
                    with gr.Column():
                        batch_targets = gr.Files(label="Target Files (Multiple Images/GIFs/Videos)", file_types=["image", "video"])
                        gr.Markdown("*Note: Face mapping will apply to all batch files*")
                    with gr.Column():
                        batch_results = gr.File(label="Download Results (ZIP)")
                
                # Face mapping controls for Batch
                with gr.Accordion("🎭 Face Mapping (Multi-Face Swap)", open=False):
                    gr.Markdown("""
                    **Instructions:**
                    1. Upload source image (face mapping setup)
                    2. Upload target files for batch processing
                    3. Click "Detect Faces" to preview source faces
                    4. Add mappings - will apply to ALL batch files
                    5. Target faces detected individually per file
                    """)
                    
                    face_mapping_status_batch = gr.Textbox(
                        label="Mapping Status",
                        value="Upload source image to detect faces",
                        interactive=False
                    )
                    
                    with gr.Row():
                        mapping_source_idx_batch = gr.Dropdown(
                            label="Source Face Index",
                            choices=[],
                            interactive=True
                        )
                        mapping_arrow_batch = gr.Markdown("→")
                        mapping_target_idx_batch = gr.Dropdown(
                            label="Target Face Index",
                            choices=[],
                            interactive=True
                        )
                        add_mapping_btn_batch = gr.Button("Add Mapping", size="sm")
                    
                    current_mappings_batch = gr.Textbox(
                        label="Current Mappings",
                        value="No mappings",
                        interactive=False,
                        lines=3
                    )
                    
                    with gr.Row():
                        clear_mappings_btn_batch = gr.Button("Clear All Mappings", size="sm")
                        detect_faces_btn_batch = gr.Button("Detect Source Faces", variant="secondary", size="sm")
                
                with gr.Row():
                    batch_enhance = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                    batch_quality = gr.Dropdown(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Balanced (4x, Tile 256)",
                        label="Enhancement Quality",
                        visible=False
                    )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        batch_confidence = gr.Slider(
                            minimum=0.3,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Face Detection Confidence Threshold",
                            info="Higher = stricter face detection (0.5 recommended)"
                        )
                        batch_gpu_selection = gr.Dropdown(
                            choices=get_gpu_options(),
                            value=get_gpu_options()[0] if get_gpu_options() else None,
                            label="GPU Selection",
                            info="Choose which GPU(s) to use for processing"
                        )
                    with gr.Row():
                        batch_compare = gr.Checkbox(
                            label="Generate Comparison Images",
                            value=False,
                            info="Create side-by-side comparisons for each result"
                        )
                
                batch_status = gr.Textbox(label="Batch Processing Status", lines=3, interactive=False)
                run_batch_btn = gr.Button("🚀 Run Batch Processing", variant="primary", size="lg")
        
        # GPU Memory Monitor
        gr.Markdown("---")
        gr.Markdown("### 🖥️ GPU Status Monitor")
        
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
        
        # Event handlers - Enhancement controls visibility
        def toggle_enhancement_controls(enabled):
            return {
                quality_preset: gr.update(visible=enabled),
                model_row: gr.update(visible=enabled)
            }
        
        enhance_toggle.change(
            toggle_enhancement_controls,
            inputs=[enhance_toggle],
            outputs=[quality_preset, model_row]
        )
        
        # Model selection handler - show denoise slider for general-x4v3 model
        def toggle_denoise_slider(model_choice):
            supports_denoise = MODEL_OPTIONS.get(model_choice, {}).get("supports_denoise", False)
            return gr.update(visible=supports_denoise)
        
        model_selector.change(
            toggle_denoise_slider,
            inputs=[model_selector],
            outputs=[denoise_slider]
        )
        
        # GIF tab enhancement controls
        def toggle_gif_enhancement_controls(enabled):
            return {
                quality_preset_gif: gr.update(visible=enabled),
                model_row_gif: gr.update(visible=enabled)
            }
        
        enhance_toggle_gif.change(
            toggle_gif_enhancement_controls,
            inputs=[enhance_toggle_gif],
            outputs=[quality_preset_gif, model_row_gif]
        )
        
        model_selector_gif.change(
            toggle_denoise_slider,
            inputs=[model_selector_gif],
            outputs=[denoise_slider_gif]
        )
        
        # Video tab enhancement controls
        def toggle_vid_enhancement_controls(enabled):
            return {
                quality_preset_vid: gr.update(visible=enabled),
                model_row_vid: gr.update(visible=enabled)
            }
        
        enhance_toggle_vid.change(
            toggle_vid_enhancement_controls,
            inputs=[enhance_toggle_vid],
            outputs=[quality_preset_vid, model_row_vid]
        )
        
        model_selector_vid.change(
            toggle_denoise_slider,
            inputs=[model_selector_vid],
            outputs=[denoise_slider_vid]
        )
        batch_enhance.change(
            lambda x: gr.update(visible=x),
            inputs=[batch_enhance],
            outputs=[batch_quality]
        )
        
        # Face detection on upload
        source_img.change(
            detect_faces_from_upload,
            inputs=[source_img, face_confidence],
            outputs=[face_info_img]
        )
        face_confidence.change(
            detect_faces_from_upload,
            inputs=[source_img, face_confidence],
            outputs=[face_info_img]
        )
        
        source_gif.change(
            detect_faces_from_upload,
            inputs=[source_gif, face_confidence_gif],
            outputs=[face_info_gif]
        )
        face_confidence_gif.change(
            detect_faces_from_upload,
            inputs=[source_gif, face_confidence_gif],
            outputs=[face_info_gif]
        )
        
        source_vid.change(
            detect_faces_from_upload,
            inputs=[source_vid, face_confidence_video],
            outputs=[face_info_vid]
        )
        face_confidence_video.change(
            detect_faces_from_upload,
            inputs=[source_vid, face_confidence_video],
            outputs=[face_info_vid]
        )
        
        batch_source.change(
            detect_faces_from_upload,
            inputs=[batch_source, batch_confidence],
            outputs=[batch_face_info]
        )
        batch_confidence.change(
            detect_faces_from_upload,
            inputs=[batch_source, batch_confidence],
            outputs=[batch_face_info]
        )
        
        # Wrappers for processing
        def wrapped_process_image(src, tgt, enhance, quality, confidence, gpu, model, denoise):
            return process_input(src, tgt, None, enhance, quality, confidence, gpu, None, model, denoise)
        
        def wrapped_process_gif(src, tgt, enhance, quality, confidence, gpu, model, denoise):
            return process_input(src, None, tgt, enhance, quality, confidence, gpu, None, model, denoise)
        
        def wrapped_process_video(src, tgt, enhance, quality, confidence, gpu, model, denoise):
            return process_input(src, None, tgt, enhance, quality, confidence, gpu, None, model, denoise)
        
        # Face mapping event handlers
        detect_faces_btn.click(
            detect_faces_for_mapping,
            inputs=[source_img, target_img, face_confidence],
            outputs=[
                source_faces_gallery,
                target_faces_gallery,
                face_mapping_status,
                mapping_source_idx,
                mapping_target_idx,
                source_faces_gallery,
                target_faces_gallery
            ]
        )
        
        add_mapping_btn.click(
            add_face_mapping,
            inputs=[mapping_source_idx, mapping_target_idx, current_mappings],
            outputs=[face_mapping_status, current_mappings]
        )
        
        clear_mappings_btn.click(
            clear_face_mappings,
            outputs=[face_mapping_status, current_mappings]
        )
        
        # GIF preview and face mapping
        target_gif_file.change(
            show_gif_preview,
            inputs=[target_gif_file],
            outputs=[target_gif_preview]
        )
        
        detect_faces_btn_gif.click(
            detect_faces_gif_video,
            inputs=[source_gif, target_gif_file, face_confidence_gif],
            outputs=[
                source_faces_gallery_gif,
                target_faces_gallery_gif,
                face_mapping_status_gif,
                mapping_source_idx_gif,
                mapping_target_idx_gif,
                source_faces_gallery_gif,
                target_faces_gallery_gif
            ]
        )
        
        add_mapping_btn_gif.click(
            add_face_mapping,
            inputs=[mapping_source_idx_gif, mapping_target_idx_gif, current_mappings_gif],
            outputs=[face_mapping_status_gif, current_mappings_gif]
        )
        
        clear_mappings_btn_gif.click(
            clear_face_mappings,
            outputs=[face_mapping_status_gif, current_mappings_gif]
        )
        
        # Video face mapping
        detect_faces_btn_vid.click(
            detect_faces_gif_video,
            inputs=[source_vid, target_vid, face_confidence_video],
            outputs=[
                source_faces_gallery_vid,
                target_faces_gallery_vid,
                face_mapping_status_vid,
                mapping_source_idx_vid,
                mapping_target_idx_vid,
                source_faces_gallery_vid,
                target_faces_gallery_vid
            ]
        )
        
        add_mapping_btn_vid.click(
            add_face_mapping,
            inputs=[mapping_source_idx_vid, mapping_target_idx_vid, current_mappings_vid],
            outputs=[face_mapping_status_vid, current_mappings_vid]
        )
        
        clear_mappings_btn_vid.click(
            clear_face_mappings,
            outputs=[face_mapping_status_vid, current_mappings_vid]
        )
        
        # Batch face mapping
        detect_faces_btn_batch.click(
            detect_faces_batch_source,
            inputs=[batch_source, batch_confidence],
            outputs=[
                source_faces_gallery_batch,
                face_mapping_status_batch,
                mapping_source_idx_batch,
                source_faces_gallery_batch
            ]
        )
        
        batch_source.change(
            lambda: gr.update(choices=[f"Target Face {i}" for i in range(10)], value="Target Face 0"),
            outputs=[mapping_target_idx_batch]
        )
        
        add_mapping_btn_batch.click(
            add_face_mapping,
            inputs=[mapping_source_idx_batch, mapping_target_idx_batch, current_mappings_batch],
            outputs=[face_mapping_status_batch, current_mappings_batch]
        )
        
        clear_mappings_btn_batch.click(
            clear_face_mappings,
            outputs=[face_mapping_status_batch, current_mappings_batch]
        )
        
        # Processing button handlers
        run_image_btn.click(
            wrapped_process_image,
            inputs=[source_img, target_img, enhance_toggle, quality_preset, face_confidence, gpu_selection, model_selector, denoise_slider],
            outputs=[result_img],
            show_progress='full'
        )
        
        run_gif_btn.click(
            wrapped_process_gif,
            inputs=[source_gif, target_gif_file, enhance_toggle_gif, quality_preset_gif, face_confidence_gif, gpu_selection_gif, model_selector_gif, denoise_slider_gif],
            outputs=[result_gif],
            show_progress='full'
        )
        
        run_video_btn.click(
            wrapped_process_video,
            inputs=[source_vid, target_vid, enhance_toggle_vid, quality_preset_vid, face_confidence_video, gpu_selection_video, model_selector_vid, denoise_slider_vid],
            outputs=[result_vid],
            show_progress='full'
        )
        
        run_batch_btn.click(
            process_batch,
            inputs=[batch_source, batch_targets, batch_enhance, batch_quality, batch_confidence, batch_compare, batch_gpu_selection],
            outputs=[batch_results, batch_status],
            show_progress='full'
        )
        
        # GPU refresh handler
        refresh_gpu_btn.click(
            fn=refresh_gpu_info,
            outputs=gpu_textboxes
        )
    
    return demo
