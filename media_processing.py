import logging
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from media_utils import MediaProcessor
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor
import sys
from realesrgan import RealESRGANer
import json
import subprocess

logger = logging.getLogger("FaceOff")

def _process_image(processor, source_image, dest_path, output_dir, enhance):
    dest_image = Image.open(dest_path).convert("RGB")
    logger.info(f"inswapper-shape: {processor.swapper.input_shape}")

    # Validate source_image and dest_image
    source_array = np.array(source_image)
    dest_array = np.array(dest_image)
    logger.info(f"Source image shape: {source_array.shape}, dtype: {source_array.dtype}")
    logger.info(f"Destination image shape: {dest_array.shape}, dtype: {dest_array.dtype}")

    src_faces = processor.get_faces(source_array)
    dst_faces = processor.get_faces(dest_array)

    # Log detected faces
    logger.info(f"Source faces detected: {len(src_faces)}")
    logger.info(f"Destination faces detected: {len(dst_faces)}")

    if not src_faces:
        raise ValueError("No faces detected in the source image.")
    if not dst_faces:
        raise ValueError("No faces detected in the destination image.")

    swapped = np.array(dest_image.copy(), dtype=np.uint8)  # Convert to numpy array
    swapped = np.ascontiguousarray(swapped)  # Ensure contiguity

    for face in dst_faces:
        try:
            swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
        except Exception as e:
            logger.error(f"Error during face swapping: {e}")
            raise

    # Apply Real-ESRGAN enhancement if enabled
    if enhance:
        logger.info("Enhancement enabled. Applying Real-ESRGAN.")
        try:
            # Use the cloned Real-ESRGAN repository to enhance the image
            input_path = output_dir / f"swapped_{Path(dest_path).stem}.png"
            enhanced_path = output_dir / f"enhanced_{Path(dest_path).stem}.png"

            # Construct the command to call inference_realesrgan.py
            command = [
                'python',
                'G:/My Drive/scripts/faceoff/external/Real-ESRGAN/inference_realesrgan.py',
                '-n', 'RealESRGAN_x4plus',
                '-i', str(input_path),
                '-o', str(output_dir),
                '--outscale', '4',
                '--face_enhance'
            ]

            try:
                # Run the command and wait for it to complete
                subprocess.run(command, check=True)
                logger.info(f"Enhanced image saved to {enhanced_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to enhance image using Real-ESRGAN: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN model: {e}")
            raise

    # Ensure the array is properly converted to a PIL Image for saving
    output_path = output_dir / f"swapped_{Path(dest_path).stem}.png"
    Image.fromarray(np.uint8(swapped)).save(output_path)  # Convert back to PIL Image for saving
    return str(output_path), None

def _process_animated(processor, source_image, dest_path, output_dir, enhance):
    clip = VideoFileClip(dest_path)
    fps = clip.fps

    # Extract audio from the original video
    audio = clip.audio
    if audio:
        logger.info("Audio track successfully extracted from the original video.")
    else:
        logger.warning("No audio track found in the original video.")

    frames = list(clip.iter_frames())
    logger.info(f"inswapper-shape: {processor.swapper.input_shape}, total frames: {len(frames)}")
    src_faces = processor.get_faces(source_image)

    def process_frame(frame):
        dst_faces = processor.get_faces(frame)
        swapped = frame.copy()
        for face in dst_faces:
            swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
        return swapped

    # Limit parallelism and use batch processing
    max_workers = 4  # Limit the number of threads
    batch_size = 50  # Process frames in batches
    result = []

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                result.extend(executor.map(process_frame, batch))
    except Exception as e:
        logger.error("Parallel processing failed, falling back to sequential processing: %s", e)
        result = [process_frame(frame) for frame in frames]  # Fallback to sequential

    # Apply Real-ESRGAN enhancement to each frame if enabled
    if enhance:
        logger.info("Enhancement enabled. Applying Real-ESRGAN to frames.")
        result = [processor.enhance_image(frame) for frame in result]

    # Create the processed video
    output_path = output_dir / "swapped_{}.mp4".format(Path(dest_path).stem)
    clip = ImageSequenceClip([np.array(f) for f in result], fps=fps)

    # Add the original audio to the processed video
    if audio:
        clip = clip.set_audio(audio)
        logger.info("Audio track successfully added to the processed video.")
    else:
        logger.warning("Processed video will not contain audio.")

    clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    return None, str(output_path)

def _process_gif(processor, source_image, dest_path, output_dir, enhance):
    # Extract frames and durations using MediaProcessor's process_gif method
    frame_paths, frame_durations = processor.process_gif(dest_path, output_dir)
    logger.info("Extracted %d frames from GIF.", len(frame_paths))

    # Ensure frame_durations is a list of integers, handling various cases
    def extract_duration(duration):
        try:
            if isinstance(duration, (list, np.ndarray)):
                return int(duration[0]) if len(duration) > 0 else 0
            return int(duration)
        except (TypeError, ValueError):
            logger.warning("Invalid duration value: %s. Defaulting to 100ms.", duration)
            return 100  # Default to 100ms if invalid

    frame_durations = [extract_duration(duration) for duration in frame_durations]

    # Perform face swapping on each frame
    src_faces = processor.get_faces(np.array(source_image))
    result_frames = []
    for frame_path in frame_paths:
        frame = processor.read_image(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing
        dst_faces = processor.get_faces(frame)
        swapped = frame.copy()
        for face in dst_faces:
            swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
        result_frames.append(swapped)

    # Apply Real-ESRGAN enhancement to each frame if enabled
    if enhance:
        logger.info("Enhancement enabled. Applying Real-ESRGAN to frames.")
        result_frames = [processor.enhance_image(frame) for frame in result_frames]

    # Reassemble frames into a GIF with original durations
    output_path = Path(output_dir) / f"swapped_{Path(dest_path).stem}.gif"
    result_images = [Image.fromarray(f) for f in result_frames]  # Ensure frames are in RGB
    result_images[0].save(output_path, save_all=True, append_images=result_images[1:], loop=0, duration=frame_durations)
    logger.info("GIF saved at: %s", output_path)

    # Cleanup temporary frame files
    for frame_path in frame_paths:
        try:
            Path(frame_path).unlink()
            logger.info("Deleted temporary frame: %s", frame_path)
        except Exception as e:
            logger.warning("Failed to delete temporary frame %s: %s", frame_path, e)

    return None, str(output_path)  # Return only the GIF path

def process_media(source_image, dest_path, media_type, output_dir, enhance=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info("Initializing MediaProcessor with CUDA support")
    processor = MediaProcessor()

    if media_type == "image":
        try:
            return _process_image(processor, source_image, dest_path, output_dir, enhance)
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    elif media_type == "video":
        try:
            return _process_animated(processor, source_image, dest_path, output_dir, enhance)
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
    elif media_type == "gif":
        try:
            return _process_gif(processor, source_image, dest_path, output_dir, enhance)
        except Exception as e:
            logger.error(f"GIF processing failed: {e}")
            raise
    else:
        raise ValueError(f"Unsupported media type: {media_type}")
