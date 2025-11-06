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
from utils.constants import MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_TILE_SIZE, DEFAULT_OUTSCALE, DEFAULT_USE_FP32, DEFAULT_PRE_PAD
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
    """Detect faces in both images for mapping UI using exact same method as processing."""
    if source_img is None or target_img is None:
        return (
            [], [],
            "Upload both images first",
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    from core.face_processor import FaceProcessor, sort_faces_by_position, filter_faces_by_confidence
    import cv2
    from PIL import ImageDraw, ImageFont
    
    processor = FaceProcessor(device_id=0, confidence=face_confidence)
    
    # Detect source faces using CV2 (same as processing)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as src_tmp:
        if source_img.mode != 'RGB':
            source_img = source_img.convert('RGB')
        source_img.save(src_tmp.name)
        src_cv2 = cv2.imread(src_tmp.name)
        src_cv2_rgb = cv2.cvtColor(src_cv2, cv2.COLOR_BGR2RGB)
        
        src_faces_raw = processor._processor.get_faces(src_cv2_rgb)
        src_faces_raw = filter_faces_by_confidence(src_faces_raw, face_confidence)
        src_faces_raw = sort_faces_by_position(src_faces_raw)
        
        # Extract thumbnails with index numbers
        src_faces = []
        for idx, face in enumerate(src_faces_raw):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            padding = 20
            h, w = src_cv2_rgb.shape[:2]
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
            face_crop = src_cv2_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_crop)
            
            # Add index number overlay
            draw = ImageDraw.Draw(face_pil)
            font_size = max(24, int(face_pil.width * 0.15))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            text = str(idx)
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                text_width, text_height = draw.textsize(text, font=font)
            
            padding_text = 8
            x, y = padding_text, padding_text
            bg_rect = [x - 4, y - 4, x + text_width + 4, y + text_height + 4]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            src_faces.append(face_pil)
    
    # Detect target faces using CV2 (same as processing)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tgt_tmp:
        if target_img.mode != 'RGB':
            target_img = target_img.convert('RGB')
        target_img.save(tgt_tmp.name)
        tgt_cv2 = cv2.imread(tgt_tmp.name)
        tgt_cv2_rgb = cv2.cvtColor(tgt_cv2, cv2.COLOR_BGR2RGB)
        
        tgt_faces_raw = processor._processor.get_faces(tgt_cv2_rgb)
        tgt_faces_raw = filter_faces_by_confidence(tgt_faces_raw, face_confidence)
        tgt_faces_raw = sort_faces_by_position(tgt_faces_raw)
        
        # Extract thumbnails with index numbers
        tgt_faces = []
        for idx, face in enumerate(tgt_faces_raw):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            padding = 20
            h, w = tgt_cv2_rgb.shape[:2]
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
            face_crop = tgt_cv2_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_crop)
            
            # Add index number overlay
            draw = ImageDraw.Draw(face_pil)
            font_size = max(24, int(face_pil.width * 0.15))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            text = str(idx)
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                text_width, text_height = draw.textsize(text, font=font)
            
            padding_text = 8
            x, y = padding_text, padding_text
            bg_rect = [x - 4, y - 4, x + text_width + 4, y + text_height + 4]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            tgt_faces.append(face_pil)
    
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
    """Detect faces in source image and sampled frames from GIF/Video to find all unique faces."""
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
    from core.face_processor import FaceProcessor, sort_faces_by_position, filter_faces_by_confidence, calculate_iou
    import cv2
    import numpy as np
    
    # Detect source faces using CV2 (same as processing)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as src_tmp:
        if source_img.mode != 'RGB':
            source_img = source_img.convert('RGB')
        source_img.save(src_tmp.name)
        src_cv2 = cv2.imread(src_tmp.name)
        src_cv2_rgb = cv2.cvtColor(src_cv2, cv2.COLOR_BGR2RGB)
        
        processor = FaceProcessor(device_id=0, confidence=face_confidence)
        src_faces_raw = processor._processor.get_faces(src_cv2_rgb)
        src_faces_raw = filter_faces_by_confidence(src_faces_raw, face_confidence)
        src_faces_raw = sort_faces_by_position(src_faces_raw)
        
        # Extract thumbnails from sorted faces
        src_faces = []
        for idx, face in enumerate(src_faces_raw):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            padding = 20
            h, w = src_cv2_rgb.shape[:2]
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
            face_crop = src_cv2_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_crop)
            
            # Add index number overlay (same as detect_and_extract_faces)
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(face_pil)
            font_size = max(24, int(face_pil.width * 0.15))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            text = str(idx)
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                text_width, text_height = draw.textsize(text, font=font)
            
            padding_text = 8
            x, y = padding_text, padding_text
            bg_rect = [x - 4, y - 4, x + text_width + 4, y + text_height + 4]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            src_faces.append(face_pil)
        
        src_info = f"✅ {len(src_faces)} face(s)"
    
    target_path = target_file.name if hasattr(target_file, 'name') else target_file
    
    try:
        # Get first frame - this will be the reference frame (same as processing)
        # ONLY show faces from frame 0 to match processing exactly
        first_frame_rgb = None
        
        if target_path.lower().endswith('.gif'):
            gif = Image.open(target_path)
            first_frame = gif.convert('RGB')
            first_frame_rgb = np.array(first_frame)
            gif.close()
        else:
            clip = VideoFileClip(target_path)
            first_frame_rgb = clip.get_frame(0)
            clip.close()
        
        # Detect faces in reference frame (frame 0) - this is what processing uses
        reference_faces = processor._processor.get_faces(first_frame_rgb)
        reference_faces = filter_faces_by_confidence(reference_faces, face_confidence)
        reference_faces = sort_faces_by_position(reference_faces)
        
        final_faces = reference_faces
        
        logger.info("Face detection: %d faces in frame 0 (reference frame only)",
                   len(reference_faces))
        
        # Now we need to get a good crop for each face from frame 0
        tgt_faces = []
        for idx, face in enumerate(final_faces):
            # Extract thumbnail from frame 0
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            padding = 20
            h, w = first_frame_rgb.shape[:2]
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
            face_crop = first_frame_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_crop)
            
            # Add index number overlay
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(face_pil)
            font_size = max(24, int(face_pil.width * 0.15))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            text = str(idx)
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                text_width, text_height = draw.textsize(text, font=font)
            
            padding_text = 8
            x, y = padding_text, padding_text
            bg_rect = [x - 4, y - 4, x + text_width + 4, y + text_height + 4]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            tgt_faces.append(face_pil)
        
        tgt_info = f"✅ {len(reference_faces)} face(s) in frame 0 (reference frame)"
    
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


def process_input(source_image, target_image_path, target_video_path, enhance, confidence, gpu_selection=None, face_mappings=None, model_selection=None, denoise_strength=0.5, tile_size=None, outscale=None, use_fp32=False, pre_pad=0):
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
        
        logger.info("Processing %s with enhancement=%s, model=%s, denoise=%.2f, confidence=%.2f, tile=%d, outscale=%d, fp32=%s, prepad=%d",
                   media_type, enhance, model_name, denoise_strength, confidence, tile_size, outscale, use_fp32, pre_pad)
        
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
            pre_pad=pre_pad
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


def process_batch(source, targets, enhance, confidence, compare, gpu, model_selection, denoise_strength, tile_size=None, outscale=None, use_fp32=False, pre_pad=0):
    """
    Process multiple files in batch mode.
    """
    import shutil
    import zipfile
    from datetime import datetime
    
    if source is None or not targets:
        raise gr.Error("Please upload source image and target files.")
    
    # Use provided enhancement settings or defaults
    if tile_size is None:
        tile_size = DEFAULT_TILE_SIZE
    if outscale is None:
        outscale = DEFAULT_OUTSCALE
    if use_fp32 is None:
        use_fp32 = DEFAULT_USE_FP32
    if pre_pad is None:
        pre_pad = DEFAULT_PRE_PAD
    
    try:
        output_dir = Path("outputs") / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        status_lines = [f"Processing {len(targets)} files..."]
        
        # Get face mappings
        mappings = face_mapping_manager.get()
        
        # Use default enhancement settings
        tile_size = DEFAULT_TILE_SIZE
        outscale = DEFAULT_OUTSCALE
        
        # Parse model selection
        if model_selection and model_selection in MODEL_OPTIONS:
            model_name = MODEL_OPTIONS[model_selection]["model_name"]
        else:
            model_name = MODEL_OPTIONS[DEFAULT_MODEL]["model_name"]
        
        for idx, target_file in enumerate(targets, 1):
            try:
                target_path = target_file.name if hasattr(target_file, 'name') else target_file
                media_type = validate_media_type(target_path)
                
                status_lines.append(f"\n[{idx}/{len(targets)}] Processing {Path(target_path).name}...")
                
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
                    face_mappings=mappings,
                    model_name=model_name,
                    denoise_strength=denoise_strength,
                    use_fp32=use_fp32,
                    pre_pad=pre_pad
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
                    
                    detect_faces_btn_batch = gr.Button("🔍 Detect Source Faces", variant="primary", size="sm")
                    
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
                    
                    current_mappings_batch = gr.Textbox(
                        label="Current Mappings",
                        value="No mappings",
                        interactive=False,
                        lines=3
                    )
                    
                    with gr.Row():
                        add_mapping_btn_batch = gr.Button("➕ Add Mapping", size="sm", variant="secondary")
                        clear_mappings_btn_batch = gr.Button("🗑️ Clear All Mappings", size="sm", variant="secondary")
                
                with gr.Row():
                    batch_enhance = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                
                with gr.Row(visible=False) as model_row_batch:
                    model_selector_batch = gr.Dropdown(
                        choices=list(MODEL_OPTIONS.keys()),
                        value=DEFAULT_MODEL,
                        label="Enhancement Model"
                    )
                    denoise_slider_batch = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                        label="Denoise Strength", visible=False
                    )
                
                with gr.Row(visible=False) as enhancement_options_row_batch:
                    tile_size_slider_batch = gr.Slider(
                        minimum=128, maximum=512, value=DEFAULT_TILE_SIZE, step=64,
                        label="Tile Size", info="Lower = less VRAM usage"
                    )
                    outscale_slider_batch = gr.Slider(
                        minimum=2, maximum=4, value=DEFAULT_OUTSCALE, step=1,
                        label="Upscale Factor", info="2x or 4x upscaling"
                    )
                
                with gr.Row(visible=False) as enhancement_advanced_row_batch:
                    use_fp32_checkbox_batch = gr.Checkbox(
                        label="High Precision (FP32)", value=DEFAULT_USE_FP32,
                        info="Uses more VRAM but slightly better quality"
                    )
                    pre_pad_slider_batch = gr.Slider(
                        minimum=0, maximum=20, value=DEFAULT_PRE_PAD, step=5,
                        label="Edge Padding", info="Reduces edge artifacts"
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
            return [gr.update(visible=enabled)] * 3  # model_row, enhancement_options_row, enhancement_advanced_row
        
        enhance_toggle.change(
            toggle_enhancement_controls,
            inputs=[enhance_toggle],
            outputs=[model_row, enhancement_options_row, enhancement_advanced_row]
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
            return [gr.update(visible=enabled)] * 3
        
        enhance_toggle_gif.change(
            toggle_gif_enhancement_controls,
            inputs=[enhance_toggle_gif],
            outputs=[model_row_gif, enhancement_options_row_gif, enhancement_advanced_row_gif]
        )
        
        model_selector_gif.change(
            toggle_denoise_slider,
            inputs=[model_selector_gif],
            outputs=[denoise_slider_gif]
        )
        
        # Video tab enhancement controls
        def toggle_vid_enhancement_controls(enabled):
            return [gr.update(visible=enabled)] * 3
        
        enhance_toggle_vid.change(
            toggle_vid_enhancement_controls,
            inputs=[enhance_toggle_vid],
            outputs=[model_row_vid, enhancement_options_row_vid, enhancement_advanced_row_vid]
        )
        
        model_selector_vid.change(
            toggle_denoise_slider,
            inputs=[model_selector_vid],
            outputs=[denoise_slider_vid]
        )
        
        # Batch tab enhancement controls
        def toggle_batch_enhancement_controls(enabled):
            return [gr.update(visible=enabled)] * 3
        
        batch_enhance.change(
            toggle_batch_enhancement_controls,
            inputs=[batch_enhance],
            outputs=[model_row_batch, enhancement_options_row_batch, enhancement_advanced_row_batch]
        )
        
        model_selector_batch.change(
            toggle_denoise_slider,
            inputs=[model_selector_batch],
            outputs=[denoise_slider_batch]
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
        def wrapped_process_image(src, tgt, enhance, confidence, gpu, model, denoise, tile, outscale, fp32, prepad):
            return process_input(src, tgt, None, enhance, confidence, gpu, None, model, denoise, tile, outscale, fp32, prepad)
        
        def wrapped_process_gif(src, tgt, enhance, confidence, gpu, model, denoise, tile, outscale, fp32, prepad):
            return process_input(src, None, tgt, enhance, confidence, gpu, None, model, denoise, tile, outscale, fp32, prepad)
        
        def wrapped_process_video(src, tgt, enhance, confidence, gpu, model, denoise, tile, outscale, fp32, prepad):
            return process_input(src, None, tgt, enhance, confidence, gpu, None, model, denoise, tile, outscale, fp32, prepad)
        
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
            inputs=[source_img, target_img, enhance_toggle, face_confidence, gpu_selection, model_selector, denoise_slider, tile_size_slider, outscale_slider, use_fp32_checkbox, pre_pad_slider],
            outputs=[result_img],
            show_progress='full'
        )
        
        run_gif_btn.click(
            wrapped_process_gif,
            inputs=[source_gif, target_gif_file, enhance_toggle_gif, face_confidence_gif, gpu_selection_gif, model_selector_gif, denoise_slider_gif, tile_size_slider_gif, outscale_slider_gif, use_fp32_checkbox_gif, pre_pad_slider_gif],
            outputs=[result_gif],
            show_progress='full'
        )
        
        run_video_btn.click(
            wrapped_process_video,
            inputs=[source_vid, target_vid, enhance_toggle_vid, face_confidence_video, gpu_selection_video, model_selector_vid, denoise_slider_vid, tile_size_slider_vid, outscale_slider_vid, use_fp32_checkbox_vid, pre_pad_slider_vid],
            outputs=[result_vid],
            show_progress='full'
        )
        
        run_batch_btn.click(
            process_batch,
            inputs=[batch_source, batch_targets, batch_enhance, batch_confidence, batch_compare, batch_gpu_selection, model_selector_batch, denoise_slider_batch, tile_size_slider_batch, outscale_slider_batch, use_fp32_checkbox_batch, pre_pad_slider_batch],
            outputs=[batch_results, batch_status],
            show_progress='full'
        )
        
        # GPU refresh handler
        refresh_gpu_btn.click(
            fn=refresh_gpu_info,
            outputs=gpu_textboxes
        )
    
    return demo
