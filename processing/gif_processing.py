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

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position
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
    denoise_strength: float = 0.5
) -> Tuple[None, Optional[str]]:
    """
    Process GIF files with face swapping.
    
    Args:
        processor: MediaProcessor instance
        source_image: Source image as numpy array
        dest_path: Path to destination GIF
        output_dir: Output directory for results
        enhance: Whether to apply Real-ESRGAN enhancement
        tile_size: Tile size for enhancement
        outscale: Upscaling factor for enhancement
        face_confidence: Minimum face detection confidence
        device_ids: List of GPU device IDs
        face_mappings: Optional list of (source_face_idx, dest_face_idx) tuples.
                      If None, uses default (first source face to all destination faces)
        model_name: Real-ESRGAN model to use for enhancement
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
                      
    Returns:
        Tuple of (None, output_path)
    """
    if not device_ids:
        device_ids = [0]
    gpu_id = device_ids[0]  # Primary GPU for enhancement
    
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
            dst_faces = sort_faces_by_position(dst_faces)
            swapped = frame.copy()
            
            # Apply face mappings or default behavior
            if face_mappings:
                for src_idx, dst_idx in face_mappings:
                    if src_idx < len(proc_src_faces) and dst_idx < len(dst_faces):
                        swapped = proc.swapper.get(swapped, dst_faces[dst_idx], proc_src_faces[src_idx], paste_back=True)
            else:
                # Default: swap first source face to all destination faces
                for face in dst_faces:
                    if proc_src_faces:
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
        
        # Sequential processing
        for idx, frame_path in enumerate(frame_paths):
            frame = processor.read_image(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dst_faces = processor.get_faces(frame)
            dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
            dst_faces = sort_faces_by_position(dst_faces)
            swapped = frame.copy()
            
            # Apply face mappings or default behavior
            if face_mappings:
                for src_idx, dst_idx in face_mappings:
                    if src_idx < len(src_faces) and dst_idx < len(dst_faces):
                        swapped = processor.swapper.get(swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True)
            else:
                # Default: swap first source face to all destination faces
                for face in dst_faces:
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
                    denoise_strength=denoise_strength
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
                    denoise_strength=denoise_strength
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
