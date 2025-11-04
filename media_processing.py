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

    # Ensure the array is properly converted to a PIL Image for saving
    output_path = output_dir / f"swapped_{Path(dest_path).stem}.png"
    Image.fromarray(np.uint8(swapped)).save(output_path)  # Convert back to PIL Image for saving
    
    # Apply Real-ESRGAN enhancement if enabled
    if enhance:
        logger.info("Enhancement enabled. Applying Real-ESRGAN.")
        
        # Free up GPU memory before running Real-ESRGAN
        # This helps avoid CUDA out of memory errors
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared GPU cache before enhancement")
        
        try:
            # Construct the command to call inference_realesrgan.py
            # Using --tile 256 to reduce memory usage and avoid CUDA OOM errors
            command = [
                'python',
                'G:/My Drive/scripts/faceoff/external/Real-ESRGAN/inference_realesrgan.py',
                '-n', 'RealESRGAN_x4plus',
                '-i', str(output_path),
                '-o', str(output_dir),
                '--outscale', '4',
                '--tile', '256',
                '--face_enhance'
            ]

            try:
                # Run the command and wait for it to complete
                subprocess.run(command, check=True)
                
                # Real-ESRGAN saves with _out suffix by default
                enhanced_path = output_dir / f"swapped_{Path(dest_path).stem}_out.png"
                
                if enhanced_path.exists():
                    # Replace the original with the enhanced version
                    shutil.move(str(enhanced_path), str(output_path))
                    logger.info(f"Enhanced image saved to {output_path}")
                else:
                    logger.warning(f"Enhanced image not found at {enhanced_path}, using original")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to enhance image using Real-ESRGAN: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN model: {e}")
            raise
    
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
        logger.info("Enhancement enabled. Applying Real-ESRGAN to video frames.")
        
        # Free up GPU memory before running Real-ESRGAN
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared GPU cache before enhancement")
        
        try:
            # Create temporary directories for frames
            temp_frames_dir = output_dir / "temp_video_frames"
            temp_frames_dir.mkdir(exist_ok=True)
            
            enhanced_output_dir = output_dir / "temp_video_enhanced"
            
            # Remove enhanced directory if it exists from previous run
            if enhanced_output_dir.exists():
                import shutil as sh
                sh.rmtree(enhanced_output_dir)
            
            # Extract frames from the video
            video_clip = VideoFileClip(str(output_path))
            frame_count = int(video_clip.fps * video_clip.duration)
            
            logger.info(f"Extracting {frame_count} frames from video for enhancement")
            
            frame_paths = []
            for i, frame in enumerate(video_clip.iter_frames()):
                frame_path = temp_frames_dir / f"frame_{i:06d}.png"
                Image.fromarray(frame).save(frame_path)
                frame_paths.append(frame_path)
            
            video_clip.close()
            logger.info(f"Saved {len(frame_paths)} frames for enhancement")
            
            # Enhance all frames using Real-ESRGAN
            command = [
                'python',
                'G:/My Drive/scripts/faceoff/external/Real-ESRGAN/inference_realesrgan.py',
                '-n', 'RealESRGAN_x4plus',
                '-i', str(temp_frames_dir),
                '-o', str(enhanced_output_dir),
                '--outscale', '4',
                '--tile', '256',
                '--face_enhance'
            ]

            try:
                # Run the command and wait for it to complete
                subprocess.run(command, check=True)
                
                # Load enhanced frames
                enhanced_frames = []
                
                if not enhanced_output_dir.exists():
                    logger.error(f"Enhanced frames directory not found: {enhanced_output_dir}")
                    logger.warning("Returning non-enhanced video")
                else:
                    for i in range(len(frame_paths)):
                        enhanced_frame_path = enhanced_output_dir / f"frame_{i:06d}_out.png"
                        if enhanced_frame_path.exists():
                            enhanced_frames.append(np.array(Image.open(enhanced_frame_path)))
                        else:
                            logger.warning(f"Enhanced frame {i} not found at {enhanced_frame_path}")
                    
                    # Create enhanced video - be lenient with frame count (Real-ESRGAN sometimes produces +/- 1-2 frames)
                    expected_frames = len(frame_paths)
                    actual_frames = len(enhanced_frames)
                    frame_diff = abs(actual_frames - expected_frames)
                    
                    if enhanced_frames and frame_diff <= 2:
                        if frame_diff > 0:
                            logger.warning(f"Frame count mismatch. Expected {expected_frames}, got {actual_frames}. Adjusting...")
                            # If we have too many frames, trim them
                            if actual_frames > expected_frames:
                                enhanced_frames = enhanced_frames[:expected_frames]
                            # If we have too few frames, duplicate the last frame
                            elif actual_frames < expected_frames:
                                while len(enhanced_frames) < expected_frames:
                                    enhanced_frames.append(enhanced_frames[-1])
                        
                        enhanced_clip = ImageSequenceClip(enhanced_frames, fps=fps)
                        
                        # Add audio back
                        if audio:
                            enhanced_clip = enhanced_clip.set_audio(audio)
                        
                        # Overwrite the original video with enhanced version
                        enhanced_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
                        logger.info(f"Enhanced video saved to {output_path} with {len(enhanced_frames)} frames")
                    else:
                        logger.warning(f"Frame count mismatch too large. Expected {expected_frames}, got {actual_frames}")
                        logger.warning("Keeping original video")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to enhance video using Real-ESRGAN: {e}")
                logger.warning("Returning non-enhanced video")
            finally:
                # Clean up temporary frames
                import shutil as sh
                try:
                    sh.rmtree(temp_frames_dir)
                    logger.info("Cleaned up temporary frame files")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary frames: {e}")
                
                try:
                    if enhanced_output_dir.exists():
                        sh.rmtree(enhanced_output_dir)
                        logger.info("Cleaned up enhanced frame files")
                except Exception as e:
                    logger.warning(f"Failed to clean up enhanced frames: {e}")
                    
        except Exception as e:
            logger.error(f"Enhancement initialization failed: {e}")
            logger.warning("Returning non-enhanced video")
    
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

    # Reassemble frames into a GIF first
    output_path = Path(output_dir) / f"swapped_{Path(dest_path).stem}.gif"
    result_images = [Image.fromarray(f) for f in result_frames]  # Ensure frames are in RGB
    result_images[0].save(output_path, save_all=True, append_images=result_images[1:], loop=0, duration=frame_durations)
    logger.info("GIF saved at: %s", output_path)

    # Apply Real-ESRGAN enhancement if enabled
    if enhance:
        logger.info("Enhancement enabled. Applying Real-ESRGAN to GIF frames.")
        
        # Free up GPU memory before running Real-ESRGAN
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared GPU cache before enhancement")
        
        try:
            # Create a temporary directory for enhanced frames
            temp_frames_dir = output_dir / "temp_gif_frames"
            temp_frames_dir.mkdir(exist_ok=True)
            
            enhanced_output_dir = output_dir / "temp_gif_enhanced"
            
            # Remove enhanced directory if it exists from previous run
            if enhanced_output_dir.exists():
                import shutil as sh
                sh.rmtree(enhanced_output_dir)
            
            # Save each frame as a separate image
            frame_paths = []
            for i, frame in enumerate(result_frames):
                frame_path = temp_frames_dir / f"frame_{i:04d}.png"
                Image.fromarray(frame).save(frame_path)
                frame_paths.append(frame_path)
            
            logger.info(f"Saved {len(frame_paths)} frames for enhancement")
            
            # Enhance all frames using Real-ESRGAN
            command = [
                'python',
                'G:/My Drive/scripts/faceoff/external/Real-ESRGAN/inference_realesrgan.py',
                '-n', 'RealESRGAN_x4plus',
                '-i', str(temp_frames_dir),
                '-o', str(enhanced_output_dir),
                '--outscale', '4',
                '--tile', '256',
                '--face_enhance'
            ]

            try:
                # Run the command and wait for it to complete
                subprocess.run(command, check=True)
                
                # Load enhanced frames - Real-ESRGAN saves with _out suffix by default
                enhanced_frames = []
                
                if not enhanced_output_dir.exists():
                    logger.error(f"Enhanced frames directory not found: {enhanced_output_dir}")
                    logger.warning("Returning non-enhanced GIF")
                else:
                    for i in range(len(frame_paths)):
                        # Real-ESRGAN adds _out suffix by default
                        enhanced_frame_path = enhanced_output_dir / f"frame_{i:04d}_out.png"
                        if enhanced_frame_path.exists():
                            enhanced_img = Image.open(enhanced_frame_path)
                            logger.info(f"Loaded enhanced frame {i}: size={enhanced_img.size}")
                            enhanced_frames.append(enhanced_img)
                        else:
                            logger.warning(f"Enhanced frame {i} not found at {enhanced_frame_path}, using original")
                            enhanced_frames.append(Image.fromarray(result_frames[i]))
                    
                    # Create enhanced GIF
                    if enhanced_frames and len(enhanced_frames) == len(result_frames):
                        # Scale durations if frames are upscaled
                        enhanced_frames[0].save(
                            output_path, 
                            save_all=True, 
                            append_images=enhanced_frames[1:], 
                            loop=0, 
                            duration=frame_durations
                        )
                        logger.info(f"Enhanced GIF saved to {output_path} with {len(enhanced_frames)} frames")
                    else:
                        logger.warning(f"Frame count mismatch or no enhanced frames. Expected {len(result_frames)}, got {len(enhanced_frames)}")
                        logger.warning("Keeping original GIF")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to enhance GIF using Real-ESRGAN: {e}")
                logger.warning("Returning non-enhanced GIF")
            finally:
                # Clean up temporary frames
                import shutil as sh
                try:
                    sh.rmtree(temp_frames_dir)
                    logger.info("Cleaned up temporary frame files")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary frames: {e}")
                
                try:
                    if enhanced_output_dir.exists():
                        sh.rmtree(enhanced_output_dir)
                        logger.info("Cleaned up enhanced frame files")
                except Exception as e:
                    logger.warning(f"Failed to clean up enhanced frames: {e}")
                    
        except Exception as e:
            logger.error(f"Enhancement initialization failed: {e}")
            logger.warning("Returning non-enhanced GIF")

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
