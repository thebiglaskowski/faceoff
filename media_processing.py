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
import subprocess

logger = logging.getLogger("FaceOff")

def _apply_realesrgan_enhancement(input_path, output_dir, media_type="image", frame_pattern=None, 
                                   fps=None, audio=None, frame_durations=None, tile_size=256, outscale=4):
    """
    Apply Real-ESRGAN enhancement to images, videos, or GIF frames.
    
    Args:
        input_path: Path to single image OR directory containing frames
        output_dir: Output directory for enhanced results
        media_type: "image", "video", or "gif"
        frame_pattern: Pattern for frame filenames (e.g., "frame_{:06d}.png" for video)
        fps: Frames per second (for video reconstruction)
        audio: Audio track to add back (for video)
        frame_durations: List of durations for each frame (for GIF)
        tile_size: Tile size for processing (256 for 8GB GPU, 512 for faster/more VRAM)
        outscale: Upscaling factor (2 or 4)
    
    Returns:
        Path to enhanced output file (for image/video/gif) or None if enhancement failed
    """
    logger.info("Enhancement enabled. Applying Real-ESRGAN to %s.", media_type)
    
    # Free up GPU memory before running Real-ESRGAN
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Cleared GPU cache before enhancement")
    
    try:
        # Construct the command to call inference_realesrgan.py
        # Using --tile to reduce memory usage and avoid CUDA OOM errors
        command = [
            'python',
            'G:/My Drive/scripts/faceoff/external/Real-ESRGAN/inference_realesrgan.py',
            '-n', 'RealESRGAN_x4plus',
            '-i', str(input_path),
            '-o', str(output_dir / f"temp_{media_type}_enhanced" if media_type != "image" else output_dir),
            '--outscale', str(outscale),
            '--tile', str(tile_size),
            '--face_enhance'
        ]

        try:
            # Run the command and wait for it to complete
            subprocess.run(command, check=True)
            
            # Handle different media types
            if media_type == "image":
                # Real-ESRGAN saves with _out suffix by default
                enhanced_path = output_dir / f"{Path(input_path).stem}_out.png"
                
                if enhanced_path.exists():
                    # Replace the original with the enhanced version
                    shutil.move(str(enhanced_path), str(input_path))
                    logger.info("Enhanced image saved to %s", input_path)
                    return input_path
                else:
                    logger.warning("Enhanced image not found at %s, using original", enhanced_path)
                    return None
                    
            elif media_type == "video":
                # Load enhanced frames and reconstruct video
                enhanced_output_dir = output_dir / "temp_video_enhanced"
                temp_frames_dir = Path(input_path)  # input_path is the frames directory
                
                if not enhanced_output_dir.exists():
                    logger.error("Enhanced frames directory not found: %s", enhanced_output_dir)
                    return None
                
                # Count frames in original directory
                frame_files = sorted(temp_frames_dir.glob("frame_*.png"))
                expected_frames = len(frame_files)
                
                # Load enhanced frames
                enhanced_frames = []
                for i in range(expected_frames):
                    enhanced_frame_path = enhanced_output_dir / f"frame_{i:06d}_out.png"
                    if enhanced_frame_path.exists():
                        enhanced_frames.append(np.array(Image.open(enhanced_frame_path)))
                    else:
                        logger.warning("Enhanced frame %s not found at %s", i, enhanced_frame_path)
                
                # Handle frame count mismatch (Real-ESRGAN sometimes produces +/- 1-2 frames)
                actual_frames = len(enhanced_frames)
                frame_diff = abs(actual_frames - expected_frames)
                
                if enhanced_frames and frame_diff <= 2:
                    if frame_diff > 0:
                        logger.warning("Frame count mismatch. Expected %s, got %s. Adjusting...", expected_frames, actual_frames)
                        # If we have too many frames, trim them
                        if actual_frames > expected_frames:
                            enhanced_frames = enhanced_frames[:expected_frames]
                        # If we have too few frames, duplicate the last frame
                        elif actual_frames < expected_frames:
                            while len(enhanced_frames) < expected_frames:
                                enhanced_frames.append(enhanced_frames[-1])
                    
                    # Reconstruct video with audio
                    enhanced_clip = ImageSequenceClip(enhanced_frames, fps=fps)
                    if audio:
                        enhanced_clip = enhanced_clip.set_audio(audio)
                    
                    logger.info("Enhanced video with %s frames", len(enhanced_frames))
                    return enhanced_frames, enhanced_clip
                else:
                    logger.warning("Frame count mismatch too large. Expected %s, got %s", expected_frames, actual_frames)
                    return None
                    
            elif media_type == "gif":
                # Load enhanced frames and reconstruct GIF
                enhanced_output_dir = output_dir / "temp_gif_enhanced"
                temp_frames_dir = Path(input_path)  # input_path is the frames directory
                
                if not enhanced_output_dir.exists():
                    logger.error("Enhanced frames directory not found: %s", enhanced_output_dir)
                    return None
                
                # Count frames in original directory
                frame_files = sorted(temp_frames_dir.glob("frame_*.png"))
                expected_frames = len(frame_files)
                
                # Load enhanced frames
                enhanced_frames = []
                for i in range(expected_frames):
                    enhanced_frame_path = enhanced_output_dir / f"frame_{i:04d}_out.png"
                    if enhanced_frame_path.exists():
                        enhanced_img = Image.open(enhanced_frame_path)
                        logger.info("Loaded enhanced frame %s: size=%s", i, enhanced_img.size)
                        enhanced_frames.append(enhanced_img)
                    else:
                        logger.warning("Enhanced frame %s not found at %s", i, enhanced_frame_path)
                        return None
                
                if len(enhanced_frames) == expected_frames:
                    logger.info("Enhanced GIF with %s frames", len(enhanced_frames))
                    return enhanced_frames
                else:
                    logger.warning("Frame count mismatch. Expected %s, got %s", expected_frames, len(enhanced_frames))
                    return None
                    
        except subprocess.CalledProcessError as e:
            logger.error("Failed to enhance %s using Real-ESRGAN: %s", media_type, e)
            return None
            
    except Exception as e:
        logger.error("Enhancement initialization failed: %s", e)
        return None

def _process_image(processor, source_image, dest_path, output_dir, enhance, tile_size=256, outscale=4):
    dest_image = Image.open(dest_path).convert("RGB")
    logger.info("inswapper-shape: %s", processor.swapper.input_shape)

    # Validate source_image and dest_image
    source_array = np.array(source_image)
    dest_array = np.array(dest_image)
    logger.info("Source image shape: %s, dtype: %s", source_array.shape, source_array.dtype)
    logger.info("Destination image shape: %s, dtype: %s", dest_array.shape, dest_array.dtype)

    src_faces = processor.get_faces(source_array)
    dst_faces = processor.get_faces(dest_array)

    # Log detected faces
    logger.info("Source faces detected: %s", len(src_faces))
    logger.info("Destination faces detected: %s", len(dst_faces))

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
            logger.error("Error during face swapping: %s", e)
            raise

    # Ensure the array is properly converted to a PIL Image for saving
    output_path = output_dir / f"swapped_{Path(dest_path).stem}.png"
    Image.fromarray(np.uint8(swapped)).save(output_path)
    
    # Apply Real-ESRGAN enhancement if enabled
    if enhance:
        _apply_realesrgan_enhancement(output_path, output_dir, media_type="image", tile_size=tile_size, outscale=outscale)
    
    return str(output_path), None

def _process_animated(processor, source_image, dest_path, output_dir, enhance, tile_size=256, outscale=4):
    clip = VideoFileClip(dest_path)
    fps = clip.fps

    # Extract audio from the original video
    audio = clip.audio
    if audio:
        logger.info("Audio track successfully extracted from the original video.")
    else:
        logger.warning("No audio track found in the original video.")

    frames = list(clip.iter_frames())
    logger.info("inswapper-shape: %s, total frames: %s", processor.swapper.input_shape, len(frames))
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

    # Create the processed video first
    output_path = output_dir / "swapped_{}.mp4".format(Path(dest_path).stem)
    clip = ImageSequenceClip([np.array(f) for f in result], fps=fps)

    # Create the processed video first
    output_path = output_dir / "swapped_{}.mp4".format(Path(dest_path).stem)
    clip = ImageSequenceClip([np.array(f) for f in result], fps=fps)

    # Add the original audio to the processed video
    if audio:
        clip = clip.set_audio(audio)
        logger.info("Audio track successfully added to the processed video.")
    else:
        logger.warning("Processed video will not contain audio.")

    clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    
    # Apply Real-ESRGAN enhancement if enabled
    if enhance:
        temp_frames_dir = output_dir / "temp_video_frames"
        enhanced_output_dir = output_dir / "temp_video_enhanced"
        
        try:
            # Create temporary directories for frames
            temp_frames_dir.mkdir(exist_ok=True)
            
            enhanced_output_dir = output_dir / "temp_video_enhanced"
            
            # Remove enhanced directory if it exists from previous run
            if enhanced_output_dir.exists():
                shutil.rmtree(enhanced_output_dir)
            
            # Extract frames from the video
            video_clip = VideoFileClip(str(output_path))
            
            logger.info("Extracting frames from video for enhancement")
            
            for i, frame in enumerate(video_clip.iter_frames()):
                frame_path = temp_frames_dir / f"frame_{i:06d}.png"
                Image.fromarray(frame).save(frame_path)
            
            video_clip.close()
            
            # Enhance frames and get result
            result = _apply_realesrgan_enhancement(
                temp_frames_dir, 
                output_dir, 
                media_type="video",
                fps=fps,
                audio=audio,
                tile_size=tile_size,
                outscale=outscale
            )
            
            if result:
                enhanced_frames, enhanced_clip = result
                # Overwrite the original video with enhanced version
                enhanced_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
                logger.info("Enhanced video saved to %s with %s frames", output_path, len(enhanced_frames))
            else:
                logger.warning("Enhancement failed, keeping original video")
                
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            logger.warning("Keeping original video")
        finally:
            # Clean up temporary frames
            try:
                if temp_frames_dir.exists():
                    shutil.rmtree(temp_frames_dir)
                    logger.info("Cleaned up temporary frame files")
            except Exception as e:
                logger.warning("Failed to clean up temporary frames: %s", e)
            
            try:
                if enhanced_output_dir.exists():
                    shutil.rmtree(enhanced_output_dir)
                    logger.info("Cleaned up enhanced frame files")
            except Exception as e:
                logger.warning("Failed to clean up enhanced frames: %s", e)
    
    return None, str(output_path)

def _process_gif(processor, source_image, dest_path, output_dir, enhance, tile_size=256, outscale=4):
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

    # Reassemble frames into a GIF first
    output_path = Path(output_dir) / f"swapped_{Path(dest_path).stem}.gif"
    result_images = [Image.fromarray(f) for f in result_frames]  # Ensure frames are in RGB
    result_images[0].save(output_path, save_all=True, append_images=result_images[1:], loop=0, duration=frame_durations)
    logger.info("GIF saved at: %s", output_path)

    # Apply Real-ESRGAN enhancement if enabled
    if enhance:
        temp_frames_dir = output_dir / "temp_gif_frames"
        enhanced_output_dir = output_dir / "temp_gif_enhanced"
        
        try:
            # Create a temporary directory for enhanced frames
            temp_frames_dir.mkdir(exist_ok=True)
            
            # Remove enhanced directory if it exists from previous run
            if enhanced_output_dir.exists():
                shutil.rmtree(enhanced_output_dir)
            
            # Save each frame as a separate image
            for i, frame in enumerate(result_frames):
                frame_path = temp_frames_dir / f"frame_{i:04d}.png"
                Image.fromarray(frame).save(frame_path)
            
            # Enhance frames and get result
            enhanced_frames = _apply_realesrgan_enhancement(
                temp_frames_dir,
                output_dir,
                media_type="gif",
                frame_durations=frame_durations,
                tile_size=tile_size,
                outscale=outscale
            )
            
            # Create enhanced GIF if enhancement succeeded
            if enhanced_frames and len(enhanced_frames) == len(result_frames):
                enhanced_frames[0].save(
                    output_path, 
                    save_all=True, 
                    append_images=enhanced_frames[1:], 
                    loop=0, 
                    duration=frame_durations
                )
                logger.info("Enhanced GIF saved to %s with %s frames", output_path, len(enhanced_frames))
            else:
                logger.warning("Enhancement failed, keeping original GIF")
                
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            logger.warning("Keeping original GIF")
        finally:
            # Clean up temporary frames
            try:
                if temp_frames_dir.exists():
                    shutil.rmtree(temp_frames_dir)
                    logger.info("Cleaned up temporary frame files")
            except Exception as e:
                logger.warning("Failed to clean up temporary frames: %s", e)
            
            try:
                if enhanced_output_dir.exists():
                    shutil.rmtree(enhanced_output_dir)
                    logger.info("Cleaned up enhanced frame files")
            except Exception as e:
                logger.warning("Failed to clean up enhanced frames: %s", e)

    # Cleanup temporary frame files
    for frame_path in frame_paths:
        try:
            Path(frame_path).unlink()
            logger.info("Deleted temporary frame: %s", frame_path)
        except Exception as e:
            logger.warning("Failed to delete temporary frame %s: %s", frame_path, e)

    return None, str(output_path)  # Return only the GIF path

def process_media(source_image, dest_path, media_type, output_dir, enhance=False, tile_size=256, outscale=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info("Initializing MediaProcessor with CUDA support")
    processor = MediaProcessor()

    if media_type == "image":
        try:
            return _process_image(processor, source_image, dest_path, output_dir, enhance, tile_size, outscale)
        except Exception as e:
            logger.error("Image processing failed: %s", e)
            raise
    elif media_type == "video":
        try:
            return _process_animated(processor, source_image, dest_path, output_dir, enhance, tile_size, outscale)
        except Exception as e:
            logger.error("Video processing failed: %s", e)
            raise
    elif media_type == "gif":
        try:
            return _process_gif(processor, source_image, dest_path, output_dir, enhance, tile_size, outscale)
        except Exception as e:
            logger.error("GIF processing failed: %s", e)
            raise
    else:
        raise ValueError(f"Unsupported media type: {media_type}")
