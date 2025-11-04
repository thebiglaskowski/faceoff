import gradio as gr
import shutil
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np
import magic
from media_processing import process_media
from logging_utils import setup_logging
import os
import torch
from moviepy.editor import VideoFileClip
import subprocess

# Set up logging
setup_logging()
logger = logging.getLogger("FaceOff")

# Configuration limits
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
MAX_VIDEO_DURATION_SEC = 300  # Maximum video duration (5 minutes)
MAX_IMAGE_PIXELS = 4096 * 4096  # Maximum image resolution (4K)
MAX_GIF_FRAMES = 500  # Maximum GIF frames

def get_gpu_memory_info() -> str:
    """
    Get GPU memory usage information for all available GPUs.
    Returns formatted string with VRAM usage.
    """
    if not torch.cuda.is_available():
        return "🖥️ GPU: Not available (using CPU)"
    
    try:
        gpu_count = torch.cuda.device_count()
        info_lines = [f"🎮 Found {gpu_count} GPU(s):\n"]
        
        # Try to get info for all GPUs using nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    idx, name, total_mb, used_mb, free_mb = line.split(',')
                    idx = idx.strip()
                    name = name.strip()
                    total_gb = float(total_mb) / 1024
                    used_gb = float(used_mb) / 1024
                    free_gb = float(free_mb) / 1024
                    usage_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
                    
                    # Get PyTorch allocation info
                    allocated_gb = torch.cuda.memory_allocated(int(idx)) / (1024**3)
                    
                    info_lines.append(
                        f"GPU {idx}: {name}\n"
                        f"  📊 VRAM: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_pct:.1f}%)\n"
                        f"  💚 Free: {free_gb:.1f}GB | PyTorch: {allocated_gb:.2f}GB\n"
                    )
                return "".join(info_lines)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        # Fallback to PyTorch info only
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
            info_lines.append(
                f"GPU {i}: {gpu_name}\n"
                f"  🔧 PyTorch: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved\n"
            )
        
        return "".join(info_lines)
        
    except Exception as e:
        logger.error("Failed to get GPU info: %s", e)
        return "⚠️ GPU info unavailable"

def get_available_gpus() -> list:
    """Get list of available GPU options for dropdown."""
    if not torch.cuda.is_available():
        return ["CPU Only"]
    
    gpu_count = torch.cuda.device_count()
    options = []
    
    # Add multi-GPU option if more than one GPU
    if gpu_count > 1:
        options.append(f"All GPUs ({gpu_count} GPUs)")
    
    # Add individual GPU options
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        options.append(f"GPU {i}: {gpu_name}")
    
    return options if options else ["CPU Only"]

def validate_file_size(file_path: str, max_size_mb: int = MAX_FILE_SIZE_MB) -> None:
    """Validate file size is within limits."""
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise gr.Error(f"File too large: {file_size_mb:.1f}MB. Maximum allowed: {max_size_mb}MB")
    logger.info("File size: %.1f MB", file_size_mb)

def validate_image_resolution(image_path: str) -> None:
    """Validate image resolution is within limits."""
    img = Image.open(image_path)
    pixels = img.width * img.height
    if pixels > MAX_IMAGE_PIXELS:
        raise gr.Error(f"Image resolution too high: {img.width}x{img.height}. Maximum: 4096x4096 pixels")
    logger.info("Image resolution: %dx%d", img.width, img.height)

def validate_video_duration(video_path: str) -> None:
    """Validate video duration is within limits."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        
        if duration > MAX_VIDEO_DURATION_SEC:
            minutes = duration / 60
            max_minutes = MAX_VIDEO_DURATION_SEC / 60
            raise gr.Error(f"Video too long: {minutes:.1f} minutes. Maximum: {max_minutes:.0f} minutes")
        
        logger.info("Video duration: %.1f seconds", duration)
    except Exception as e:
        logger.error("Failed to validate video: %s", e)
        raise gr.Error("Invalid video file or unable to read video metadata")

def validate_gif_frames(gif_path: str) -> None:
    """Validate GIF frame count is within limits."""
    try:
        gif = Image.open(gif_path)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                gif.seek(frame_count)
        except EOFError:
            pass
        
        if frame_count > MAX_GIF_FRAMES:
            raise gr.Error(f"GIF has too many frames: {frame_count}. Maximum: {MAX_GIF_FRAMES} frames")
        
        logger.info("GIF frame count: %d", frame_count)
    except gr.Error:
        raise
    except Exception as e:
        logger.error("Failed to validate GIF: %s", e)
        raise gr.Error("Invalid GIF file or unable to read GIF metadata")

def detect_faces_info(image_path: str, confidence_threshold: float = 0.5) -> str:
    """
    Detect faces in image and return information string.
    Returns formatted string with face count and confidence scores.
    Filters faces based on confidence threshold.
    """
    try:
        from media_utils import MediaProcessor
        
        # Initialize processor
        processor = MediaProcessor()
        
        # Load and detect faces
        img = processor.read_image(image_path)
        all_faces = processor.get_faces(img)
        
        # Filter faces by confidence threshold
        faces = [f for f in all_faces if (f.det_score if hasattr(f, 'det_score') else 1.0) >= confidence_threshold]
        
        if not faces:
            filtered_count = len(all_faces) - len(faces)
            if filtered_count > 0:
                return f"⚠️ No faces meet confidence threshold {confidence_threshold:.0%} (filtered out {filtered_count} low-confidence detection(s))"
            return "⚠️ No faces detected in source image"
        
        # Format face information
        info_lines = [f"✅ Detected {len(faces)} face(s) in source image (threshold: {confidence_threshold:.0%}):"]
        for i, face in enumerate(faces, 1):
            confidence = face.det_score if hasattr(face, 'det_score') else 1.0
            age = face.age if hasattr(face, 'age') else "Unknown"
            gender = "Male" if hasattr(face, 'gender') and face.gender == 1 else "Female" if hasattr(face, 'gender') else "Unknown"
            info_lines.append(f"  Face {i}: Confidence {confidence:.1%}, Age ~{age}, Gender: {gender}")
        
        return "\n".join(info_lines)
        
    except Exception as e:
        logger.error("Face detection failed: %s", e)
        return f"❌ Face detection failed: {str(e)}"

def validate_media(file_path: str) -> str:
    mime_type = magic.from_file(file_path, mime=True)
    if mime_type == "image/gif":
        return "gif"
    elif mime_type.startswith("image"):
        return "image"
    elif mime_type.startswith("video"):
        return "video"
    raise gr.Error("Unsupported media type: only images, GIFs, and videos are allowed.")

def process_batch(source_img, target_files, enhance, quality_preset, face_confidence, create_comparisons, gpu_selection):
    """
    Process multiple target files with the same source face.
    Returns a ZIP file containing all results.
    """
    import zipfile
    import tempfile
    
    if source_img is None:
        return None, "❌ Please upload a source face image"
    
    if not target_files or len(target_files) == 0:
        return None, "❌ Please upload at least one target file"
    
    try:
        # Create temporary directory for batch results
        batch_dir = Path("outputs") / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = len(target_files)
        results = []
        status_messages = []
        
        for idx, target_file in enumerate(target_files):
            file_name = "unknown"
            try:
                file_name = Path(target_file.name).name
                status_messages.append(f"⏳ Processing {file_name}...")
                
                # Determine if it's image or video
                target_path = Path(target_file.name)
                mime_type = magic.from_file(str(target_path), mime=True)
                
                if mime_type.startswith("image"):
                    target_img = Image.open(target_path)
                    result = process_input(source_img, target_img, None, enhance, quality_preset, 
                                         face_confidence, create_comparisons, gpu_selection)
                elif mime_type.startswith("video") or mime_type == "image/gif":
                    result = process_input(source_img, None, target_file, enhance, quality_preset, 
                                         face_confidence, create_comparisons, gpu_selection)
                else:
                    status_messages.append(f"⚠️ Skipped {file_name}: Unsupported format")
                    continue
                
                # Copy result to batch directory
                if result:
                    result_path = Path(result)
                    dest_path = batch_dir / f"{idx:03d}_{result_path.name}"
                    shutil.copy(result_path, dest_path)
                    results.append(dest_path)
                    status_messages.append(f"✅ Completed {file_name}")
                
            except Exception as e:
                logger.error("Batch processing error for %s: %s", file_name, e)
                status_messages.append(f"❌ Failed {file_name}: {str(e)}")
        
        # Create ZIP file
        zip_path = batch_dir.parent / f"{batch_dir.name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for result_file in results:
                zipf.write(result_file, result_file.name)
        
        final_status = f"✅ Batch Complete!\n" + \
                      f"Processed: {len(results)}/{total_files} files\n" + \
                      "\n".join(status_messages[-10:])  # Show last 10 messages
        
        return str(zip_path), final_status
        
    except Exception as e:
        logger.error("Batch processing failed: %s", e)
        return None, f"❌ Batch processing failed: {str(e)}"

# Ensure GPU prioritization
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Optimize process_input for better handling of media
def process_input(src_img, dest_img, dest_vid, enhance, quality_preset="Balanced (4x, Tile 256)", 
                 face_confidence=0.5, compare_view=False, gpu_selection=None):
    start_time = time.time()
    
    inputs_dir = Path("inputs")
    outputs_dir = Path("outputs")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse quality preset
    tile_size = 256
    outscale = 4
    if "Fast" in quality_preset:
        tile_size = 512
        outscale = 2
    elif "Quality" in quality_preset:
        tile_size = 128
        outscale = 4
    # Balanced is default (256, 4)
    
    logger.info("=== Starting Face Swap Process ===")
    logger.info("Enhancement: %s, Quality: %s", "Enabled" if enhance else "Disabled", quality_preset if enhance else "N/A")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    source_path = inputs_dir / f"source_{timestamp}.png"
    src_img.save(source_path)
    
    # Validate source image
    validate_image_resolution(str(source_path))

    media_path = None
    if dest_img is not None:
        media_path = inputs_dir / f"target_image_{timestamp}.png"
        dest_img.save(media_path)
        # Validate destination image
        validate_image_resolution(str(media_path))
    elif dest_vid is not None:
        media_path = inputs_dir / f"target_video_{timestamp}{Path(dest_vid.name).suffix}"
        shutil.copy(dest_vid.name, media_path)
        # Validate destination media file
        validate_file_size(str(media_path))
    else:
        raise gr.Error("Please upload a destination image or video.")

    mime_type = magic.from_file(str(media_path), mime=True)
    logger.info("Media type detected: %s", mime_type)

    media_type = validate_media(str(media_path))
    
    # Additional validation based on media type
    if media_type == "video":
        validate_video_duration(str(media_path))
    elif media_type == "gif":
        validate_gif_frames(str(media_path))

    # Fix: Ensure proper numpy array conversion with correct data type and contiguity
    source_arr = np.array(src_img.convert("RGB"), dtype=np.uint8)
    source_arr = np.ascontiguousarray(source_arr)

    # Validate the array
    if source_arr.size == 0 or source_arr.ndim != 3:
        raise gr.Error("Invalid source image format")

    logger.info("Source image details before processing:")
    logger.info("Shape: %s, Dtype: %s, Contiguous: %s", source_arr.shape, source_arr.dtype, source_arr.flags['C_CONTIGUOUS'])

    try:
        output_path_img, output_path_vid = process_media(
            source_image=source_arr,
            dest_path=str(media_path),
            media_type=media_type,
            output_dir=str(outputs_dir),
            enhance=enhance,
            tile_size=tile_size,
            outscale=outscale,
            face_confidence=face_confidence,
            gpu_selection=gpu_selection
        )

        output_file = output_path_img or output_path_vid
        if not output_file or not Path(output_file).resolve().exists():
            logger.error("Output media does not exist: %s", output_file)
            raise gr.Error("Processing error: output file not found")

        logger.info("Output path returned: %s", output_file)

        # Cleanup temporary input files
        try:
            if source_path.exists():
                source_path.unlink()
                logger.info("Deleted temporary source file: %s", source_path)
            if media_path and Path(media_path).exists():
                Path(media_path).unlink()
                logger.info("Deleted temporary media file: %s", media_path)
        except OSError as e:  # More specific exception
            logger.warning("Failed to delete temporary input files: %s", e)
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info("=== Processing Complete ===")
        logger.info("Total processing time: %.2f seconds (%.2f minutes)", total_time, total_time / 60)
        logger.info("Media type: %s, Enhanced: %s", media_type, enhance)
        if enhance:
            logger.info("Quality preset: %s (Tile: %d, Outscale: %dx)", quality_preset, tile_size, outscale)
        logger.info("Output file size: %.2f MB", Path(output_file).stat().st_size / (1024 * 1024))
        logger.info("=" * 40)

        return str(output_file)

    except Exception as e:
        total_time = time.time() - start_time
        logger.error("=== Processing Failed ===")
        logger.error("Error after %.2f seconds: %s", total_time, e)
        logger.error("=" * 40)
        raise gr.Error("Processing error: %s" % e)


with gr.Blocks(title="FaceOff - Face Swapper") as demo:
    gr.Markdown("## FaceOff - AI Face Swapper")
    gr.Markdown("Swap faces from a source image to destination images, GIFs, or videos with optional AI enhancement.")
    
    # GPU Memory Monitor
    gpu_info = gr.Textbox(label="GPU Status", value=get_gpu_memory_info(), lines=4, interactive=False)
    refresh_gpu_btn = gr.Button("🔄 Refresh GPU Info", size="sm")
    
    refresh_gpu_btn.click(
        fn=get_gpu_memory_info,
        outputs=[gpu_info]
    )

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    source_img = gr.Image(type="pil", label="Source Image")
                    face_info_img = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                with gr.Column():
                    target_img = gr.Image(type="pil", label="Target Image")
                with gr.Column():
                    result_img = gr.Image(label="Swapped Result")

            with gr.Row():
                enhance_toggle = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                quality_preset = gr.Dropdown(
                    choices=["Fast (2x, Tile 512)", "Balanced (4x, Tile 256)", "Quality (4x, Tile 128)"],
                    value="Balanced (4x, Tile 256)",
                    label="Enhancement Quality",
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
                        choices=get_available_gpus(),
                        value=get_available_gpus()[0],
                        label="GPU Selection",
                        info="Choose which GPU(s) to use for processing"
                    )
            
            run_image_btn = gr.Button("Run Image Swap", variant="primary")

        with gr.Tab("GIF"):
            with gr.Row():
                with gr.Column():
                    source_gif = gr.Image(type="pil", label="Source Image")
                    face_info_gif = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                with gr.Column():
                    target_gif_file = gr.File(label="Target GIF", file_types=[".gif"])
                with gr.Column():
                    result_gif = gr.Image(label="Swapped Result")

            with gr.Row():
                enhance_toggle_gif = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                quality_preset_gif = gr.Dropdown(
                    choices=["Fast (2x, Tile 512)", "Balanced (4x, Tile 256)", "Quality (4x, Tile 128)"],
                    value="Balanced (4x, Tile 256)",
                    label="Enhancement Quality",
                    visible=False
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
                        choices=get_available_gpus(),
                        value=get_available_gpus()[0],
                        label="GPU Selection",
                        info="Choose which GPU(s) to use for processing"
                    )
            
            run_gif_btn = gr.Button("Run GIF Swap", variant="primary")

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    source_vid = gr.Image(type="pil", label="Source Image")
                    face_info_vid = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                with gr.Column():
                    target_vid = gr.File(label="Target Video", file_types=[".mp4", ".webp"])
                with gr.Column():
                    result_vid = gr.Video(label="Swapped Result")

            with gr.Row():
                enhance_toggle_vid = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                quality_preset_vid = gr.Dropdown(
                    choices=["Fast (2x, Tile 512)", "Balanced (4x, Tile 256)", "Quality (4x, Tile 128)"],
                    value="Balanced (4x, Tile 256)",
                    label="Enhancement Quality",
                    visible=False
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
                        choices=get_available_gpus(),
                        value=get_available_gpus()[0],
                        label="GPU Selection",
                        info="Choose which GPU(s) to use for processing"
                    )
            
            run_video_btn = gr.Button("Run Video Swap", variant="primary")
        
        with gr.Tab("Batch Processing"):
            gr.Markdown("### Process Multiple Files at Once")
            gr.Markdown("Upload a single source face image and multiple target files for batch face swapping.")
            
            with gr.Row():
                with gr.Column():
                    batch_source = gr.Image(type="pil", label="Source Face (Single)")
                    batch_face_info = gr.Textbox(label="Face Detection Info", lines=4, interactive=False)
                with gr.Column():
                    batch_targets = gr.Files(label="Target Files (Multiple Images/GIFs/Videos)", file_types=["image", "video"])
                with gr.Column():
                    batch_results = gr.File(label="Download Results (ZIP)")
            
            with gr.Row():
                batch_enhance = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
                batch_quality = gr.Dropdown(
                    choices=["Fast (2x, Tile 512)", "Balanced (4x, Tile 256)", "Quality (4x, Tile 128)"],
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
                        choices=get_available_gpus(),
                        value=get_available_gpus()[0],
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
    
    # Show/hide quality dropdown based on enhancement toggle
    enhance_toggle.change(
        lambda x: gr.update(visible=x),
        inputs=[enhance_toggle],
        outputs=[quality_preset]
    )
    enhance_toggle_gif.change(
        lambda x: gr.update(visible=x),
        inputs=[enhance_toggle_gif],
        outputs=[quality_preset_gif]
    )
    enhance_toggle_vid.change(
        lambda x: gr.update(visible=x),
        inputs=[enhance_toggle_vid],
        outputs=[quality_preset_vid]
    )
    batch_enhance.change(
        lambda x: gr.update(visible=x),
        inputs=[batch_enhance],
        outputs=[batch_quality]
    )
    
    # Face detection on image upload
    def detect_faces_from_upload(img, confidence=0.5):
        if img is None:
            return ""
        try:
            # Save temporarily to detect faces
            temp_path = Path("inputs") / "temp_detect.png"
            temp_path.parent.mkdir(exist_ok=True)
            img.save(temp_path)
            info = detect_faces_info(str(temp_path), confidence)
            temp_path.unlink()
            return info
        except Exception as e:
            logger.error("Face detection error: %s", e)
            return f"❌ Error: {str(e)}"
    
    # Image tab - face detection updates with confidence threshold
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
    
    # GIF tab - face detection updates with confidence threshold
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
    
    # Video tab - face detection updates with confidence threshold
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

    def wrapped_process_image(src, tgt, enhance, quality, confidence, gpu):
        return process_input(src, tgt, None, enhance, quality, confidence, gpu_selection=gpu)

    def wrapped_process_gif(src, tgt, enhance, quality, confidence, gpu):
        return process_input(src, None, tgt, enhance, quality, confidence, gpu_selection=gpu)

    def wrapped_process_video(src, tgt, enhance, quality, confidence, gpu):
        return process_input(src, None, tgt, enhance, quality, confidence, gpu_selection=gpu)
    
    # Batch source face detection
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

    run_image_btn.click(
        wrapped_process_image,
        inputs=[source_img, target_img, enhance_toggle, quality_preset, face_confidence, gpu_selection],
        outputs=[result_img],
        show_progress='full'
    )

    run_gif_btn.click(
        wrapped_process_gif,
        inputs=[source_gif, target_gif_file, enhance_toggle_gif, quality_preset_gif, face_confidence_gif, gpu_selection_gif],
        outputs=[result_gif],
        show_progress='full'
    )

    run_video_btn.click(
        wrapped_process_video,
        inputs=[source_vid, target_vid, enhance_toggle_vid, quality_preset_vid, face_confidence_video, gpu_selection_video],
        outputs=[result_vid],
        show_progress='full'
    )
    
    run_batch_btn.click(
        process_batch,
        inputs=[batch_source, batch_targets, batch_enhance, batch_quality, batch_confidence, batch_compare, batch_gpu_selection],
        outputs=[batch_results, batch_status],
        show_progress='full'
    )

demo.launch(share=True)
