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

# Set up logging
setup_logging()
logger = logging.getLogger("FaceOff")

def validate_media(file_path: str) -> str:
    mime_type = magic.from_file(file_path, mime=True)
    if mime_type == "image/gif":
        return "gif"
    elif mime_type.startswith("image"):
        return "image"
    elif mime_type.startswith("video"):
        return "video"
    raise gr.Error("Unsupported media type: only images, GIFs, and videos are allowed.")

# Ensure GPU prioritization
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Optimize process_input for better handling of media
def process_input(src_img, dest_img, dest_vid, enhance):
    inputs_dir = Path("inputs")
    outputs_dir = Path("outputs")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    source_path = inputs_dir / f"source_{timestamp}.png"
    src_img.save(source_path)

    media_path = None
    if dest_img is not None:
        media_path = inputs_dir / f"target_image_{timestamp}.png"
        dest_img.save(media_path)
    elif dest_vid is not None:
        media_path = inputs_dir / f"target_video_{timestamp}{Path(dest_vid.name).suffix}"
        shutil.copy(dest_vid.name, media_path)
    else:
        raise gr.Error("Please upload a destination image or video.")

    mime_type = magic.from_file(str(media_path), mime=True)
    logger.info("Media type detected: %s", mime_type)

    media_type = validate_media(str(media_path))

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
            enhance=enhance  # Pass the enhance flag to the processing function
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

        return str(output_file)  # Return the file path directly

    except Exception as e:
        logger.error("Processing failed: %s", e)
        raise gr.Error("Processing error: %s" % e)


with gr.Blocks(title="FaceOff - Face Swapper") as demo:
    gr.Markdown("## FaceOff - AI Face Swapper")

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                source_img = gr.Image(type="pil", label="Source Image")
                target_img = gr.Image(type="pil", label="Target Image")
                result_img = gr.Image(label="Swapped Result")

            enhance_toggle = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
            run_image_btn = gr.Button("Run Image Swap")

        with gr.Tab("GIF"):
            with gr.Row():
                source_gif = gr.Image(type="pil", label="Source Image")
                target_gif_file = gr.File(label="Target GIF", file_types=[".gif"])
                result_gif = gr.Image(label="Swapped Result")

            enhance_toggle_gif = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
            run_gif_btn = gr.Button("Run GIF Swap")

        with gr.Tab("Video"):
            with gr.Row():
                source_vid = gr.Image(type="pil", label="Source Image")
                target_vid = gr.File(label="Target Video", file_types=[".mp4", ".webp"])
                result_vid = gr.Video(label="Swapped Result")

            enhance_toggle_vid = gr.Checkbox(label="Enable Enhancement (Real-ESRGAN)", value=False)
            run_video_btn = gr.Button("Run Video Swap")

    def wrapped_process_image(src, tgt, enhance):
        return process_input(src, tgt, None, enhance)

    def wrapped_process_gif(src, tgt, enhance):
        return process_input(src, None, tgt, enhance)

    def wrapped_process_video(src, tgt, enhance):
        return process_input(src, None, tgt, enhance)

    run_image_btn.click(
        wrapped_process_image,
        inputs=[source_img, target_img, enhance_toggle],
        outputs=[result_img],
        show_progress='full'
    )

    run_gif_btn.click(
        wrapped_process_gif,
        inputs=[source_gif, target_gif_file, enhance_toggle_gif],
        outputs=[result_gif],
        show_progress='full'
    )

    run_video_btn.click(
        wrapped_process_video,
        inputs=[source_vid, target_vid, enhance_toggle_vid],
        outputs=[result_vid],
        show_progress='full'
    )

demo.launch(share=True)
