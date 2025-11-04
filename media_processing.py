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
import magic

logger = logging.getLogger("FaceOff")

def filter_faces_by_confidence(faces, threshold=0.5):
    """Filter faces by detection confidence threshold."""
    if not faces:
        return faces
    filtered = [f for f in faces if (f.det_score if hasattr(f, 'det_score') else 1.0) >= threshold]
    if len(filtered) < len(faces):
        logger.info("Filtered faces: %d/%d meet confidence threshold %.0f%%", len(filtered), len(faces), threshold * 100)
    return filtered

def _apply_realesrgan_enhancement_multi_gpu(input_paths, output_dir, media_type, device_ids, fps=None, audio=None, frame_durations=None, tile_size=256, outscale=4):
    """
    Apply Real-ESRGAN enhancement using multiple GPUs for parallel processing.
    
    Args:
        input_paths: List of frame paths to enhance
        output_dir: Output directory for enhanced results
        media_type: "video" or "gif"
        device_ids: List of GPU device IDs to use
        fps: Frames per second (for video reconstruction)
        audio: Audio track to add back (for video)
        frame_durations: List of durations for each frame (for GIF)
        tile_size: Tile size for processing
        outscale: Upscaling factor
    
    Returns:
        List of enhanced frames (as PIL Images for GIF or numpy arrays for video)
    """
    if len(device_ids) <= 1:
        # Fall back to single GPU enhancement
        return _apply_realesrgan_enhancement(
            input_paths if isinstance(input_paths, (str, Path)) else input_paths[0].parent,
            output_dir, media_type, fps=fps, audio=audio, 
            frame_durations=frame_durations, tile_size=tile_size, 
            outscale=outscale, gpu_id=device_ids[0] if device_ids else 0
        )
    
    logger.info("🚀 Multi-GPU Enhancement: Distributing %d frames across %d GPUs", len(input_paths), len(device_ids))
    
    # Split frames into chunks for each GPU
    chunks = [[] for _ in device_ids]
    for idx, frame_path in enumerate(input_paths):
        gpu_idx = idx % len(device_ids)
        chunks[gpu_idx].append(frame_path)
    
    # Create temporary directories for each GPU's work
    temp_dirs = []
    enhanced_dirs = []
    for gpu_idx in range(len(device_ids)):
        temp_dir = output_dir / f"temp_{media_type}_gpu{gpu_idx}_frames"
        enhanced_dir = output_dir / f"temp_{media_type}_gpu{gpu_idx}_enhanced"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Remove enhanced directory if it exists
        if enhanced_dir.exists():
            shutil.rmtree(enhanced_dir)
        
        temp_dirs.append(temp_dir)
        enhanced_dirs.append(enhanced_dir)
    
    # Copy frames to their respective GPU directories with sequential numbering
    for gpu_idx, chunk in enumerate(chunks):
        for seq_idx, frame_path in enumerate(chunk):
            dest_path = temp_dirs[gpu_idx] / f"frame_{seq_idx:06d}.png"
            shutil.copy(frame_path, dest_path)
    
    logger.info("Split frames: %s", [len(chunk) for chunk in chunks])
    
    # Run Real-ESRGAN on each GPU in parallel
    def enhance_on_gpu(args):
        gpu_idx, gpu_id, temp_dir, enhanced_dir = args
        
        command = [
            'python',
            'G:/My Drive/scripts/faceoff/external/Real-ESRGAN/inference_realesrgan.py',
            '-n', 'RealESRGAN_x4plus',
            '-i', str(temp_dir),
            '-o', str(enhanced_dir),
            '--outscale', str(outscale),
            '--tile', str(tile_size),
            '--gpu-id', str(gpu_id),
            '--face_enhance'
        ]
        
        logger.info("GPU %d: Enhancing %d frames...", gpu_id, len(chunks[gpu_idx]))
        
        try:
            subprocess.run(command, check=True, capture_output=True)
            logger.info("✅ GPU %d: Enhancement complete", gpu_id)
            return True
        except subprocess.CalledProcessError as e:
            logger.error("GPU %d: Enhancement failed: %s", gpu_id, e)
            return False
    
    # Execute enhancement on all GPUs in parallel
    enhancement_args = [
        (gpu_idx, device_ids[gpu_idx], temp_dirs[gpu_idx], enhanced_dirs[gpu_idx])
        for gpu_idx in range(len(device_ids))
    ]
    
    try:
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            results = list(executor.map(enhance_on_gpu, enhancement_args))
        
        if not all(results):
            logger.error("Some GPU enhancement tasks failed")
            return None
    except Exception as e:
        logger.error("Multi-GPU enhancement failed: %s", e)
        return None
    
    # Collect enhanced frames in original order
    logger.info("Collecting enhanced frames from all GPUs...")
    enhanced_frames = [None] * len(input_paths)
    
    for idx, frame_path in enumerate(input_paths):
        gpu_idx = idx % len(device_ids)
        seq_idx = chunks[gpu_idx].index(frame_path)
        
        enhanced_frame_path = enhanced_dirs[gpu_idx] / f"frame_{seq_idx:06d}_out.png"
        
        if enhanced_frame_path.exists():
            if media_type == "gif":
                enhanced_frames[idx] = Image.open(enhanced_frame_path).copy()
            else:  # video
                enhanced_frames[idx] = np.array(Image.open(enhanced_frame_path))
        else:
            logger.error("Enhanced frame not found: %s", enhanced_frame_path)
            return None
    
    # Clean up temporary directories
    for temp_dir in temp_dirs + enhanced_dirs:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning("Failed to clean up %s: %s", temp_dir, e)
    
    logger.info("✅ Multi-GPU enhancement complete: %d frames enhanced", len(enhanced_frames))
    
    # Return appropriate format based on media type
    if media_type == "video":
        # Reconstruct video with audio
        enhanced_clip = ImageSequenceClip(enhanced_frames, fps=fps)
        if audio:
            enhanced_clip = enhanced_clip.set_audio(audio)
        return enhanced_frames, enhanced_clip
    else:  # gif
        return enhanced_frames


def _apply_realesrgan_enhancement(input_path, output_dir, media_type="image", frame_pattern=None, 
                                   fps=None, audio=None, frame_durations=None, tile_size=256, outscale=4, gpu_id=0):
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
        gpu_id: GPU device ID to use for Real-ESRGAN processing
    
    Returns:
        Path to enhanced output file (for image/video/gif) or None if enhancement failed
    """
    logger.info("Enhancement enabled. Applying Real-ESRGAN on GPU %d to %s.", gpu_id, media_type)
    
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
            '--gpu-id', str(gpu_id),
            '--face_enhance'
        ]
        
        logger.info("Running Real-ESRGAN on GPU %d: %s", gpu_id, ' '.join(command))

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

def _process_image(processor, source_image, dest_path, output_dir, enhance, tile_size=256, outscale=4, face_confidence=0.5, device_ids=None):
    # For single images, we just use the first device (multi-GPU doesn't help here)
    if not device_ids:
        device_ids = [0]
    gpu_id = device_ids[0]
    
    dest_image = Image.open(dest_path).convert("RGB")
    logger.info("inswapper-shape: %s", processor.swapper.input_shape)

    # Validate source_image and dest_image
    source_array = np.array(source_image)
    dest_array = np.array(dest_image)
    logger.info("Source image shape: %s, dtype: %s", source_array.shape, source_array.dtype)
    logger.info("Destination image shape: %s, dtype: %s", dest_array.shape, dest_array.dtype)

    src_faces = processor.get_faces(source_array)
    dst_faces = processor.get_faces(dest_array)
    
    # Apply confidence filtering
    src_faces = filter_faces_by_confidence(src_faces, face_confidence)
    dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)

    # Log detected faces
    logger.info("Source faces detected (after filtering): %s", len(src_faces))
    logger.info("Destination faces detected (after filtering): %s", len(dst_faces))

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
        _apply_realesrgan_enhancement(output_path, output_dir, media_type="image", tile_size=tile_size, outscale=outscale, gpu_id=gpu_id)
    
    return str(output_path), None

def _process_animated(processor, source_image, dest_path, output_dir, enhance, tile_size=256, outscale=4, face_confidence=0.5, device_ids=None):
    if not device_ids:
        device_ids = [0]
    gpu_id = device_ids[0]  # Use first GPU for enhancement
    
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
    
    # Multi-GPU setup: Create processors for each device
    if len(device_ids) > 1:
        logger.info("🚀 Multi-GPU Mode: Setting up %d GPUs: %s", len(device_ids), device_ids)
        processors = [MediaProcessor(device_id=dev_id) for dev_id in device_ids]
        
        # Get source faces using first processor
        src_faces = processors[0].get_faces(source_image)
        src_faces = filter_faces_by_confidence(src_faces, face_confidence)
        logger.info("Source faces detected (after filtering): %s", len(src_faces))
        
        # Store source faces for each processor to avoid repeated detection
        processor_src_faces = []
        for proc in processors:
            proc_src = proc.get_faces(source_image)
            proc_src = filter_faces_by_confidence(proc_src, face_confidence)
            processor_src_faces.append(proc_src)
        
        def process_frame_multi_gpu(args):
            frame_idx, frame = args
            # Distribute frames across GPUs round-robin
            gpu_idx = frame_idx % len(processors)
            proc = processors[gpu_idx]
            proc_src_faces = processor_src_faces[gpu_idx]
            
            dst_faces = proc.get_faces(frame)
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
            swapped = frame.copy()
            for face in dst_faces:
                if proc_src_faces:
                    swapped = proc.swapper.get(swapped, face, proc_src_faces[0], paste_back=True)
            return swapped
        
        # Process ALL frames at once for better GPU distribution
        logger.info("Processing %d frames across %d GPUs...", len(frames), len(device_ids))
        max_workers = len(device_ids) * 4  # 4 workers per GPU for better parallelism
        result = []
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create enumerated list for proper frame ordering
                frame_tasks = [(idx, frame) for idx, frame in enumerate(frames)]
                result = list(executor.map(process_frame_multi_gpu, frame_tasks))
                logger.info("✅ Multi-GPU processing complete: %d frames processed", len(result))
        except Exception as e:
            logger.error("Multi-GPU parallel processing failed, falling back to sequential: %s", e)
            result = [process_frame_multi_gpu((idx, frame)) for idx, frame in enumerate(frames)]
    else:
        logger.info("Single GPU Mode: Using GPU %d", device_ids[0])
        src_faces = processor.get_faces(source_image)
        src_faces = filter_faces_by_confidence(src_faces, face_confidence)
        logger.info("Source faces detected (after filtering): %s", len(src_faces))
        
        # Single GPU processing (original code)
        def process_frame(frame):
            dst_faces = processor.get_faces(frame)
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
            swapped = frame.copy()
            for face in dst_faces:
                swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
            return swapped

        max_workers = 4
        batch_size = 50
        result = []

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i + batch_size]
                    result.extend(executor.map(process_frame, batch))
        except Exception as e:
            logger.error("Parallel processing failed, falling back to sequential processing: %s", e)
            for idx, frame in enumerate(frames):
                result.append(process_frame(frame))

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
            
            frame_paths = []
            for i, frame in enumerate(video_clip.iter_frames()):
                frame_path = temp_frames_dir / f"frame_{i:06d}.png"
                Image.fromarray(frame).save(frame_path)
                frame_paths.append(frame_path)
            
            video_clip.close()
            
            # Enhance frames - use multi-GPU if available
            if len(device_ids) > 1:
                logger.info("Using multi-GPU enhancement with %d GPUs", len(device_ids))
                result = _apply_realesrgan_enhancement_multi_gpu(
                    frame_paths,
                    output_dir,
                    media_type="video",
                    device_ids=device_ids,
                    fps=fps,
                    audio=audio,
                    tile_size=tile_size,
                    outscale=outscale
                )
            else:
                result = _apply_realesrgan_enhancement(
                    temp_frames_dir, 
                    output_dir, 
                    media_type="video",
                    fps=fps,
                    audio=audio,
                    tile_size=tile_size,
                    outscale=outscale,
                    gpu_id=gpu_id
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

def _process_gif(processor, source_image, dest_path, output_dir, enhance, tile_size=256, outscale=4, face_confidence=0.5, device_ids=None):
    if not device_ids:
        device_ids = [0]
    gpu_id = device_ids[0]  # Use first GPU for enhancement
    
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
    result_frames = []
    total_frames = len(frame_paths)
    
    # Multi-GPU setup for GIF processing
    if len(device_ids) > 1:
        logger.info("🚀 Multi-GPU GIF Mode: Setting up %d GPUs: %s", len(device_ids), device_ids)
        processors = [MediaProcessor(device_id=dev_id) for dev_id in device_ids]
        
        # Pre-detect source faces for each processor
        processor_src_faces = []
        for proc in processors:
            proc_src = proc.get_faces(np.array(source_image))
            proc_src = filter_faces_by_confidence(proc_src, face_confidence)
            processor_src_faces.append(proc_src)
        logger.info("Source faces detected (after filtering): %s", len(processor_src_faces[0]))
        
        def process_gif_frame_multi_gpu(args):
            frame_idx, frame_path = args
            # Distribute frames across GPUs round-robin
            gpu_idx = frame_idx % len(processors)
            proc = processors[gpu_idx]
            proc_src_faces = processor_src_faces[gpu_idx]
            
            frame = proc.read_image(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dst_faces = proc.get_faces(frame)
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
            swapped = frame.copy()
            for face in dst_faces:
                if proc_src_faces:
                    swapped = proc.swapper.get(swapped, face, proc_src_faces[0], paste_back=True)
            return swapped
        
        # Process ALL frames at once for better GPU distribution
        logger.info("Processing %d GIF frames across %d GPUs...", len(frame_paths), len(device_ids))
        max_workers = len(device_ids) * 4  # 4 workers per GPU
        frame_tasks = [(idx, path) for idx, path in enumerate(frame_paths)]
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                result_frames = list(executor.map(process_gif_frame_multi_gpu, frame_tasks))
                logger.info("✅ Multi-GPU GIF processing complete: %d frames processed", len(result_frames))
        except Exception as e:
            logger.error("Multi-GPU GIF processing failed, falling back to sequential: %s", e)
            result_frames = [process_gif_frame_multi_gpu((idx, path)) for idx, path in enumerate(frame_paths)]
    else:
        logger.info("Single GPU GIF Mode: Using GPU %d", device_ids[0])
        src_faces = processor.get_faces(np.array(source_image))
        src_faces = filter_faces_by_confidence(src_faces, face_confidence)
        logger.info("Source faces detected (after filtering): %s", len(src_faces))
        
        # Single GPU processing (original sequential code)
        for idx, frame_path in enumerate(frame_paths):
            frame = processor.read_image(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing
            dst_faces = processor.get_faces(frame)
            # Apply confidence filtering to destination faces
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
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
            
            # Save each frame as a separate image and track paths
            frame_paths = []
            for i, frame in enumerate(result_frames):
                frame_path = temp_frames_dir / f"frame_{i:04d}.png"
                Image.fromarray(frame).save(frame_path)
                frame_paths.append(frame_path)
            
            # Enhance frames - use multi-GPU if available
            if len(device_ids) > 1:
                logger.info("Using multi-GPU GIF enhancement with %d GPUs", len(device_ids))
                enhanced_frames = _apply_realesrgan_enhancement_multi_gpu(
                    frame_paths,
                    output_dir,
                    media_type="gif",
                    device_ids=device_ids,
                    frame_durations=frame_durations,
                    tile_size=tile_size,
                    outscale=outscale
                )
            else:
                enhanced_frames = _apply_realesrgan_enhancement(
                    temp_frames_dir,
                    output_dir,
                    media_type="gif",
                    frame_durations=frame_durations,
                    tile_size=tile_size,
                    outscale=outscale,
                    gpu_id=gpu_id
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

def process_media(source_image, dest_path, media_type, output_dir, enhance=False, tile_size=256, outscale=4, face_confidence=0.5, gpu_selection=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Parse GPU selection
    device_ids = []
    if gpu_selection and gpu_selection.startswith("All GPUs"):
        # Use all available GPUs
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            logger.info("Using all %d GPUs for processing", len(device_ids))
    elif gpu_selection and gpu_selection.startswith("GPU"):
        # Extract GPU ID from string like "GPU 0: RTX 4090"
        try:
            gpu_id = int(gpu_selection.split(":")[0].split(" ")[1])
            device_ids = [gpu_id]
            logger.info("Using GPU %d for processing", gpu_id)
        except (ValueError, IndexError):
            logger.warning("Failed to parse GPU selection, using default GPU 0")
            device_ids = [0]
    else:
        # Default to GPU 0
        device_ids = [0] if torch.cuda.is_available() else []
        logger.info("Using default GPU 0 for processing")
    
    logger.info("Initializing MediaProcessor with device(s): %s", device_ids)
    # Initialize processor with the first device in the list (or device 0 as fallback)
    primary_device = device_ids[0] if device_ids else 0
    processor = MediaProcessor(device_id=primary_device)

    if media_type == "image":
        try:
            return _process_image(processor, source_image, dest_path, output_dir, enhance, tile_size, outscale, face_confidence, device_ids)
        except Exception as e:
            logger.error("Image processing failed: %s", e)
            raise
    elif media_type == "video":
        try:
            return _process_animated(processor, source_image, dest_path, output_dir, enhance, tile_size, outscale, face_confidence, device_ids)
        except Exception as e:
            logger.error("Video processing failed: %s", e)
            raise
    elif media_type == "gif":
        try:
            return _process_gif(processor, source_image, dest_path, output_dir, enhance, tile_size, outscale, face_confidence, device_ids)
        except Exception as e:
            logger.error("GIF processing failed: %s", e)
            raise
    else:
        raise ValueError(f"Unsupported media type: {media_type}")
