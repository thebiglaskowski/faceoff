"""
Video processing with face swapping and enhancement.
"""
import logging
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip, ImageSequenceClip
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position, FaceTracker
from core.media_processor import MediaProcessor
from processing.enhancement import enhance_frames_single_gpu, enhance_frames_multi_gpu

logger = logging.getLogger("FaceOff")


def process_video(
    processor: MediaProcessor,
    source_image: np.ndarray,
    dest_path: str,
    output_dir: Path,
    enhance: bool = False,
    tile_size: int = 256,
    outscale: int = 4,
    face_confidence: float = 0.5,
    device_ids: Optional[List[int]] = None,
    face_mappings: Optional[List[Tuple[int, int]]] = None,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0
) -> Tuple[None, Optional[str]]:
    """
    Process video files with face swapping.
    
    Args:
        processor: MediaProcessor instance
        source_image: Source image as numpy array
        dest_path: Path to destination video
        output_dir: Output directory for results
        enhance: Whether to apply Real-ESRGAN enhancement
        tile_size: Tile size for enhancement (128-512, lower = less VRAM)
        outscale: Upscaling factor for enhancement (2 or 4)
        face_confidence: Minimum face detection confidence
        device_ids: List of GPU device IDs
        face_mappings: Optional list of (source_face_idx, dest_face_idx) tuples.
                      If None, uses default (first source face to all destination faces)
        model_name: Real-ESRGAN model to use for enhancement
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts
                      
    Returns:
        Tuple of (None, output_path)
    """
    if not device_ids:
        device_ids = [0]
    
    # Load video and extract audio
    clip = VideoFileClip(dest_path)
    fps = clip.fps
    audio = clip.audio
    
    if audio:
        logger.info("Audio track successfully extracted from the original video.")
    else:
        logger.warning("No audio track found in the original video.")
    
    frames = list(clip.iter_frames())
    logger.info("inswapper-shape: %s, total frames: %d", processor.swapper.input_shape, len(frames))
    
    # Multi-GPU setup
    if len(device_ids) > 1:
        logger.info("🚀 Multi-GPU Video Mode: %d GPUs: %s", len(device_ids), device_ids)
        processors = [MediaProcessor(device_id=dev_id) for dev_id in device_ids]
        
        # Pre-detect source faces for each processor
        processor_src_faces = []
        for proc in processors:
            proc_src = proc.get_faces(source_image)
            proc_src = filter_faces_by_confidence(proc_src, face_confidence)
            proc_src = sort_faces_by_position(proc_src)
            processor_src_faces.append(proc_src)
        
        logger.info("Source faces detected (after filtering): %d", len(processor_src_faces[0]))
        
        # Detect faces in first frame to establish reference positions
        reference_faces = processors[0].get_faces(frames[0])
        reference_faces = filter_faces_by_confidence(reference_faces, face_confidence)
        reference_faces = sort_faces_by_position(reference_faces)
        logger.info("Reference frame has %d faces for tracking", len(reference_faces))
        
        def match_faces_to_reference(current_faces, reference_faces):
            """Match current faces to reference frame using IoU for stable IDs."""
            if not current_faces or not reference_faces:
                return current_faces
            
            from core.face_processor import calculate_iou
            
            matched_faces = [None] * len(reference_faces)
            matched_indices = set()  # Track which current faces have been matched
            
            for ref_idx, ref_face in enumerate(reference_faces):
                best_iou = 0.0
                best_match_idx = -1
                
                for curr_idx, curr_face in enumerate(current_faces):
                    if curr_idx in matched_indices:
                        continue
                    iou = calculate_iou(ref_face.bbox, curr_face.bbox)
                    if iou > best_iou and iou >= 0.3:
                        best_iou = iou
                        best_match_idx = curr_idx
                
                if best_match_idx >= 0:
                    matched_faces[ref_idx] = current_faces[best_match_idx]
                    matched_indices.add(best_match_idx)
            
            # Keep None as placeholders to maintain stable indices!
            stable_faces = matched_faces
            unmatched = [f for idx, f in enumerate(current_faces) if idx not in matched_indices]
            if unmatched:
                stable_faces.extend(sort_faces_by_position(unmatched))
            
            return stable_faces
        
        def process_frame_multi_gpu(args):
            frame_idx, frame = args
            # Distribute frames round-robin across GPUs
            gpu_idx = frame_idx % len(processors)
            proc = processors[gpu_idx]
            proc_src_faces = processor_src_faces[gpu_idx]
            
            dst_faces = proc.get_faces(frame)
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
            
            # Match to reference frame for stable IDs
            dst_faces = match_faces_to_reference(dst_faces, reference_faces)
            
            swapped = frame.copy()
            
            # Apply face mappings or default behavior
            if face_mappings:
                for src_idx, dst_idx in face_mappings:
                    # Check bounds and ensure target face exists (not None from tracking)
                    if src_idx < len(proc_src_faces) and dst_idx < len(dst_faces) and dst_faces[dst_idx] is not None:
                        swapped = proc.swapper.get(swapped, dst_faces[dst_idx], proc_src_faces[src_idx], paste_back=True)
            else:
                # Default: swap first source face to all destination faces
                for face in dst_faces:
                    if proc_src_faces and face is not None:  # Skip None placeholders
                        swapped = proc.swapper.get(swapped, face, proc_src_faces[0], paste_back=True)
            return swapped
        
        # Process all frames with multi-GPU parallelism
        logger.info("Processing %d frames across %d GPUs...", len(frames), len(device_ids))
        max_workers = len(device_ids) * 4  # 4 workers per GPU
        frame_tasks = [(idx, frame) for idx, frame in enumerate(frames)]
        result = []
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                result = list(executor.map(process_frame_multi_gpu, frame_tasks))
                logger.info("✅ Multi-GPU processing complete: %d frames", len(result))
        except Exception as e:
            logger.error("Multi-GPU processing failed, falling back to sequential: %s", e)
            result = [process_frame_multi_gpu((idx, frame)) for idx, frame in enumerate(frames)]
    
    else:
        # Single GPU processing
        logger.info("Single GPU Video Mode: GPU %d", device_ids[0])
        src_faces = processor.get_faces(source_image)
        src_faces = filter_faces_by_confidence(src_faces, face_confidence)
        src_faces = sort_faces_by_position(src_faces)
        logger.info("Source faces detected (after filtering): %d", len(src_faces))
        
        # Initialize face tracker for stable face IDs across frames
        face_tracker = FaceTracker(iou_threshold=0.3)
        
        def process_frame(frame):
            dst_faces = processor.get_faces(frame)
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
            
            # Use face tracker to maintain stable IDs across frames
            dst_faces = face_tracker.track_faces(dst_faces)
            
            swapped = frame.copy()
            
            # Apply face mappings or default behavior
            if face_mappings:
                for src_idx, dst_idx in face_mappings:
                    # Check bounds and ensure target face exists (not None from tracking)
                    if src_idx < len(src_faces) and dst_idx < len(dst_faces) and dst_faces[dst_idx] is not None:
                        swapped = processor.swapper.get(swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True)
            else:
                # Default: swap first source face to all destination faces
                for face in dst_faces:
                    if face is not None:  # Skip None placeholders
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
            logger.error("Parallel processing failed, falling back to sequential: %s", e)
            for frame in frames:
                result.append(process_frame(frame))
    
    # Create processed video with timestamp-based unique name
    import time
    timestamp = int(time.time() * 1000)
    output_path = output_dir / f"swapped_{timestamp}.mp4"
    clip = ImageSequenceClip([np.array(f) for f in result], fps=fps)
    
    # Add audio back
    if audio:
        clip = clip.set_audio(audio)
        logger.info("Audio track successfully added to the processed video.")
    else:
        logger.warning("Processed video will not contain audio.")
    
    clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    
    # Apply enhancement if requested
    if enhance:
        temp_frames_dir = output_dir / "temp_video_frames"
        enhanced_output_dir = output_dir / "temp_video_enhanced"
        
        try:
            # Extract frames for enhancement
            temp_frames_dir.mkdir(exist_ok=True)
            
            if enhanced_output_dir.exists():
                shutil.rmtree(enhanced_output_dir)
            
            video_clip = VideoFileClip(str(output_path))
            logger.info("Extracting frames from video for enhancement")
            
            frame_paths = []
            for i, frame in enumerate(video_clip.iter_frames()):
                frame_path = temp_frames_dir / f"frame_{i:06d}.png"
                Image.fromarray(frame).save(frame_path)
                frame_paths.append(frame_path)
            
            video_clip.close()
            
            # Enhance frames using multi-GPU or single GPU
            if len(device_ids) > 1:
                logger.info("Using multi-GPU enhancement with %d GPUs", len(device_ids))
                result = enhance_frames_multi_gpu(
                    frame_paths,
                    output_dir,
                    media_type="video",
                    device_ids=device_ids,
                    fps=fps,
                    audio=audio,
                    tile_size=tile_size,
                    outscale=outscale,
                    model_name=model_name,
                    denoise_strength=denoise_strength,
                    use_fp32=use_fp32,
                    pre_pad=pre_pad
                )
            else:
                result = enhance_frames_single_gpu(
                    temp_frames_dir,
                    output_dir,
                    media_type="video",
                    fps=fps,
                    audio=audio,
                    tile_size=tile_size,
                    outscale=outscale,
                    gpu_id=device_ids[0] if device_ids else 0,
                    model_name=model_name,
                    denoise_strength=denoise_strength,
                    use_fp32=use_fp32,
                    pre_pad=pre_pad
                )
            
            if result:
                enhanced_frames, enhanced_clip = result
                # Overwrite original with enhanced version
                enhanced_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
                logger.info("Enhanced video saved to %s with %d frames", output_path, len(enhanced_frames))
            else:
                logger.warning("Enhancement failed, keeping original video")
        
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            logger.warning("Keeping original video")
        
        finally:
            # Clean up temporary frames
            for temp_dir in [temp_frames_dir, enhanced_output_dir]:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        logger.info("Cleaned up %s", temp_dir.name)
                except Exception as e:
                    logger.warning("Failed to clean up %s: %s", temp_dir, e)
    
    return None, str(output_path)
