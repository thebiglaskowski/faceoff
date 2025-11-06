"""
GIF processing with face swapping and enhancement.
"""
import cv2
import logging
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position, FaceTracker
from core.media_processor import MediaProcessor
from processing.enhancement import enhance_frames_single_gpu, enhance_frames_multi_gpu

logger = logging.getLogger("FaceOff")


def extract_duration(duration) -> int:
    """
    Extract duration value from various formats.
    
    Args:
        duration: Duration value (can be int, list, array, etc.)
        
    Returns:
        Duration in milliseconds
    """
    try:
        if isinstance(duration, (list, np.ndarray)):
            return int(duration[0]) if len(duration) > 0 else 100
        return int(duration)
    except (TypeError, ValueError):
        logger.warning("Invalid duration value: %s. Defaulting to 100ms.", duration)
        return 100


def process_gif(
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
    Process GIF files with face swapping.
    
    Args:
        processor: MediaProcessor instance
        source_image: Source image as numpy array
        dest_path: Path to destination GIF
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
    gpu_id = device_ids[0]  # Primary GPU for enhancement
    
    # Log face mappings received
    logger.info("process_gif received face_mappings: %s", face_mappings)
    
    # Extract frames and durations from GIF
    frame_paths, frame_durations = processor.process_gif(dest_path, str(output_dir))
    logger.info("Extracted %d frames from GIF.", len(frame_paths))
    
    # Normalize frame durations
    frame_durations = [extract_duration(duration) for duration in frame_durations]
    
    # Process frames with face swapping
    result_frames = []
    
    # Multi-GPU setup
    if len(device_ids) > 1:
        logger.info("🚀 Multi-GPU GIF Mode: %d GPUs: %s", len(device_ids), device_ids)
        processors = [MediaProcessor(device_id=dev_id) for dev_id in device_ids]
        
        # Pre-detect source faces for each processor
        processor_src_faces = []
        for proc in processors:
            proc_src = proc.get_faces(np.array(source_image))
            proc_src = filter_faces_by_confidence(proc_src, face_confidence)
            proc_src = sort_faces_by_position(proc_src)
            processor_src_faces.append(proc_src)
        
        logger.info("Source faces detected (after filtering): %d", len(processor_src_faces[0]))
        
        # Detect faces in first frame to establish reference positions
        first_frame = cv2.imread(str(frame_paths[0]))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        reference_faces = processors[0].get_faces(first_frame)
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
            # Only append truly new unmatched faces at the end
            unmatched = [f for idx, f in enumerate(current_faces) if idx not in matched_indices]
            stable_faces = matched_faces  # Keep None entries for index stability
            if unmatched:
                stable_faces.extend(sort_faces_by_position(unmatched))
            
            logger.debug("Face tracking: %d current -> %d matched, %d None, %d unmatched -> %d total", 
                        len(current_faces), len([f for f in matched_faces if f is not None]),
                        len([f for f in matched_faces if f is None]),
                        len(unmatched), len(stable_faces))
            
            return stable_faces
        
        def process_gif_frame_multi_gpu(args):
            frame_idx, frame_path = args
            # Distribute frames round-robin across GPUs
            gpu_idx = frame_idx % len(processors)
            proc = processors[gpu_idx]
            proc_src_faces = processor_src_faces[gpu_idx]
            
            frame = proc.read_image(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dst_faces = proc.get_faces(frame)
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
            
            # Match to reference frame for stable IDs
            dst_faces = match_faces_to_reference(dst_faces, reference_faces)
            
            swapped = frame.copy()
            
            # Apply face mappings or default behavior
            if face_mappings:
                if frame_idx == 0:  # Log first frame for debugging
                    logger.info("Frame 0: Attempting mapped swaps - mappings=%s, src_faces=%d, dst_faces=%d", 
                               face_mappings, len(proc_src_faces), len(dst_faces))
                for src_idx, dst_idx in face_mappings:
                    # Check bounds and ensure target face exists (not None from tracking)
                    if src_idx < len(proc_src_faces) and dst_idx < len(dst_faces) and dst_faces[dst_idx] is not None:
                        swapped = proc.swapper.get(swapped, dst_faces[dst_idx], proc_src_faces[src_idx], paste_back=True)
                        if frame_idx == 0:
                            logger.info("Frame 0: ✓ Swapped src %d → dst %d", src_idx, dst_idx)
                    else:
                        if frame_idx == 0:
                            face_missing = dst_idx < len(dst_faces) and dst_faces[dst_idx] is None
                            logger.warning("Frame 0: ✗ SKIPPED mapping src %d → dst %d (out of bounds=%s, face_missing=%s, src_count=%d, dst_count=%d)", 
                                         src_idx, dst_idx, dst_idx >= len(dst_faces), face_missing, len(proc_src_faces), len(dst_faces))
            else:
                # Default: swap first source face to all destination faces
                if frame_idx == 0:  # Log first frame for debugging
                    logger.info("Frame 0: Default swap - %d source faces, %d dest faces", len(proc_src_faces), len(dst_faces))
                for face in dst_faces:
                    if proc_src_faces and face is not None:  # Skip None placeholders
                        swapped = proc.swapper.get(swapped, face, proc_src_faces[0], paste_back=True)
            return swapped
        
        # Process all frames with multi-GPU parallelism
        logger.info("Processing %d GIF frames across %d GPUs...", len(frame_paths), len(device_ids))
        max_workers = len(device_ids) * 4  # 4 workers per GPU
        frame_tasks = [(idx, path) for idx, path in enumerate(frame_paths)]
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                result_frames = list(executor.map(process_gif_frame_multi_gpu, frame_tasks))
                logger.info("✅ Multi-GPU GIF processing complete: %d frames", len(result_frames))
        except Exception as e:
            logger.error("Multi-GPU GIF processing failed, falling back to sequential: %s", e)
            result_frames = [process_gif_frame_multi_gpu((idx, path)) for idx, path in enumerate(frame_paths)]
    
    else:
        # Single GPU processing
        logger.info("Single GPU GIF Mode: GPU %d", device_ids[0])
        src_faces = processor.get_faces(np.array(source_image))
        src_faces = filter_faces_by_confidence(src_faces, face_confidence)
        src_faces = sort_faces_by_position(src_faces)
        logger.info("Source faces detected (after filtering): %d", len(src_faces))
        
        # Initialize face tracker for stable face IDs across frames
        face_tracker = FaceTracker(iou_threshold=0.3)
        
        # Sequential processing
        for idx, frame_path in enumerate(frame_paths):
            frame = processor.read_image(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            
            result_frames.append(swapped)
    
    # Reassemble frames into GIF with timestamp-based unique name
    import time
    timestamp = int(time.time() * 1000)
    output_path = Path(output_dir) / f"swapped_{timestamp}.gif"
    result_images = [Image.fromarray(f) for f in result_frames]
    result_images[0].save(
        output_path,
        save_all=True,
        append_images=result_images[1:],
        loop=0,
        duration=frame_durations
    )
    logger.info("GIF saved at: %s", output_path)
    
    # Apply enhancement if requested
    if enhance:
        temp_frames_dir = output_dir / "temp_gif_frames"
        enhanced_output_dir = output_dir / "temp_gif_enhanced"
        
        try:
            # Save frames for enhancement
            temp_frames_dir.mkdir(exist_ok=True)
            
            if enhanced_output_dir.exists():
                shutil.rmtree(enhanced_output_dir)
            
            # Save each frame
            enhancement_frame_paths = []
            for i, frame in enumerate(result_frames):
                frame_path = temp_frames_dir / f"frame_{i:04d}.png"
                Image.fromarray(frame).save(frame_path)
                enhancement_frame_paths.append(frame_path)
            
            # Enhance frames using multi-GPU or single GPU
            if len(device_ids) > 1:
                logger.info("Using multi-GPU GIF enhancement with %d GPUs", len(device_ids))
                enhanced_frames = enhance_frames_multi_gpu(
                    enhancement_frame_paths,
                    output_dir,
                    media_type="gif",
                    device_ids=device_ids,
                    frame_durations=frame_durations,
                    tile_size=tile_size,
                    outscale=outscale,
                    model_name=model_name,
                    denoise_strength=denoise_strength,
                    use_fp32=use_fp32,
                    pre_pad=pre_pad
                )
            else:
                enhanced_frames = enhance_frames_single_gpu(
                    temp_frames_dir,
                    output_dir,
                    media_type="gif",
                    tile_size=tile_size,
                    outscale=outscale,
                    gpu_id=gpu_id,
                    model_name=model_name,
                    denoise_strength=denoise_strength,
                    use_fp32=use_fp32,
                    pre_pad=pre_pad
                )
            
            # Create enhanced GIF if successful
            if enhanced_frames and len(enhanced_frames) == len(result_frames):
                enhanced_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=enhanced_frames[1:],
                    loop=0,
                    duration=frame_durations
                )
                logger.info("Enhanced GIF saved to %s with %d frames", output_path, len(enhanced_frames))
            else:
                logger.warning("Enhancement failed, keeping original GIF")
        
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            logger.warning("Keeping original GIF")
        
        finally:
            # Clean up temporary frames
            for temp_dir in [temp_frames_dir, enhanced_output_dir]:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        logger.info("Cleaned up %s", temp_dir.name)
                except Exception as e:
                    logger.warning("Failed to clean up %s: %s", temp_dir, e)
    
    # Cleanup temporary frames from original GIF extraction
    logger.info("Cleaning up %d temporary frame files...", len(frame_paths))
    cleanup_count = 0
    for frame_path in frame_paths:
        try:
            frame_file = Path(frame_path)
            if frame_file.exists():
                frame_file.unlink()
                cleanup_count += 1
                logger.debug("Deleted temporary frame: %s", frame_path)
            else:
                logger.debug("Frame file already deleted: %s", frame_path)
        except Exception as e:
            logger.warning("Failed to delete temporary frame %s: %s", frame_path, e)
    
    logger.info("Cleaned up %d/%d temporary frame files", cleanup_count, len(frame_paths))
    
    return None, str(output_path)
