"""
GIF processing with face swapping and enhancement.
"""
import cv2
import logging
import numpy as np
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position, FaceTracker
from core.media_processor import MediaProcessor
from processing.enhancement import enhance_frames_single_gpu, enhance_frames_multi_gpu
from processing.face_restoration import FaceRestorer
from processing.video_processing import process_frames_batch
from processing.resolution_adaptive import ResolutionAdaptiveProcessor
from processing.async_pipeline import AsyncPipeline
from utils.config_manager import config
from utils.temp_manager import get_temp_manager
from utils.progress import get_progress_tracker

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
    pre_pad: int = 0,
    restore_faces: bool = False,
    restoration_weight: float = 0.5,
    adaptive_detection: bool = None,
    detection_scale: float = None,
    use_async_pipeline: bool = True
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
        restore_faces: Whether to apply GFPGAN face restoration after swapping
        restoration_weight: GFPGAN restoration strength (0=original, 1=fully restored)
                      
    Returns:
        Tuple of (None, output_path)
    """
    if not device_ids:
        device_ids = [0]
    gpu_id = device_ids[0]  # Primary GPU for enhancement
    
    # Log face mappings received
    logger.info("process_gif received face_mappings: %s", face_mappings)
    
    # Use temp manager for extracted GIF frames
    temp_manager = get_temp_manager()
    frames_temp_dir = temp_manager.get_temp_dir("gif") / f"frames_{Path(dest_path).stem}"
    frames_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames and durations from GIF
    frame_paths, frame_durations = processor.process_gif(dest_path, str(frames_temp_dir))
    logger.info("Extracted %d frames from GIF to temp directory.", len(frame_paths))
    
    # Normalize frame durations
    frame_durations = [extract_duration(duration) for duration in frame_durations]
    
    # Create progress tracker for all processing stages
    progress = get_progress_tracker()
    
    # Process frames with face swapping
    result_frames = []
    
    # Multi-GPU setup - now with batching support
    if len(device_ids) > 1:
        logger.info("Multi-GPU GIF Mode: %d GPUs: %s (batched processing)", len(device_ids), device_ids)
        
        # Stage 1: Face Detection
        progress.set_stage("🔍 Face Detection")
        progress.log("📸 Detecting faces in source image...")
        
        processors = [MediaProcessor(device_id=dev_id, use_tensorrt=True) for dev_id in device_ids]
        
        # Create a single global lock for all swapper operations to prevent corruption
        # Even though each GPU has its own processor, the inswapper model has shared state
        global_swapper_lock = threading.Lock()
        
        # Pre-detect source faces for each processor
        processor_src_faces = []
        for proc in processors:
            proc_src = proc.get_faces(np.array(source_image))
            proc_src = filter_faces_by_confidence(proc_src, face_confidence)
            proc_src = sort_faces_by_position(proc_src)
            processor_src_faces.append(proc_src)
        
        progress.log(f"✅ Found {len(processor_src_faces[0])} source face(s)")
        logger.info("Source faces detected (after filtering): %d", len(processor_src_faces[0]))
        
        # Initialize resolution-adaptive processor if enabled
        adaptive_processor = None
        if adaptive_detection:
            adaptive_processor = ResolutionAdaptiveProcessor(
                detection_scale=detection_scale
            )
            logger.info("Resolution-adaptive detection enabled (scale=%.2f)", detection_scale)
        
        # Initialize face tracker for stable face IDs across frames
        face_tracker = FaceTracker(iou_threshold=0.3)
        
        # Load all frames into memory
        all_frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        
        # Use config for batch size
        batch_size = config.batch_size
        num_gpus = len(processors)
        
        # Distribute frames to GPUs (round-robin)
        gpu_frame_batches = [[] for _ in range(num_gpus)]
        gpu_frame_indices = [[] for _ in range(num_gpus)]
        
        for idx, frame in enumerate(all_frames):
            gpu_idx = idx % num_gpus
            gpu_frame_batches[gpu_idx].append(frame)
            gpu_frame_indices[gpu_idx].append(idx)
        
        logger.info("Processing %d frames across %d GPUs in batches of %d...", 
                   len(all_frames), num_gpus, batch_size)        
        # Process each GPU's frames in parallel using threads
        def process_gpu_frames(gpu_idx):
            proc = processors[gpu_idx]
            src_faces = processor_src_faces[gpu_idx]
            frames = gpu_frame_batches[gpu_idx]
            
            # Each GPU needs its own face tracker to avoid thread-safety issues
            gpu_face_tracker = FaceTracker(iou_threshold=0.3)
            
            gpu_results = []
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                batch_results = process_frames_batch(
                    proc, batch, src_faces, face_confidence,
                    face_mappings, gpu_face_tracker, adaptive_processor, global_swapper_lock
                )
                gpu_results.extend(batch_results)
            
            return gpu_idx, gpu_results
        
        # Process all GPUs in parallel
        gpu_results_dict = {}
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(process_gpu_frames, gpu_idx) for gpu_idx in range(num_gpus)]
            for future in futures:
                gpu_idx, gpu_frames = future.result()
                gpu_results_dict[gpu_idx] = gpu_frames
        
        # Reconstruct frames in original order
        result_frames = [None] * len(all_frames)
        for gpu_idx in range(num_gpus):
            indices = gpu_frame_indices[gpu_idx]
            frames = gpu_results_dict[gpu_idx]
            for original_idx, frame in zip(indices, frames):
                result_frames[original_idx] = frame
        
        logger.info("Multi-GPU GIF processing complete: %d frames", len(result_frames))
    
    else:
        # Single GPU processing with batching
        logger.info("Single GPU GIF Mode: GPU %d (batched processing)", device_ids[0])
        
        # Stage 1: Face Detection
        progress.set_stage("🔍 Face Detection")
        progress.log("📸 Detecting faces in source image...")
        
        src_faces = processor.get_faces(np.array(source_image))
        src_faces = filter_faces_by_confidence(src_faces, face_confidence)
        src_faces = sort_faces_by_position(src_faces)
        
        progress.log(f"✅ Found {len(src_faces)} source face(s)")
        logger.info("Source faces detected (after filtering): %d", len(src_faces))
        
        # Initialize resolution-adaptive processor if enabled
        adaptive_processor = None
        if adaptive_detection:
            adaptive_processor = ResolutionAdaptiveProcessor(
                detection_scale=detection_scale
            )
            logger.info("Resolution-adaptive detection enabled (scale=%.2f)", detection_scale)
        
        # Load all frames into memory for processing
        frames = []
        for frame_path in frame_paths:
            frame = processor.read_image(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        # Use config defaults if not specified
        if adaptive_detection is None:
            adaptive_detection = config.adaptive_detection_enabled
        if detection_scale is None:
            detection_scale = config.detection_scale
        
        # Choose processing method: async pipeline or traditional batching
        if use_async_pipeline and len(frames) > 10:
            logger.info("Using async pipeline for overlapped processing")
            pipeline = AsyncPipeline(
                processor=processor,
                src_faces=src_faces,
                face_confidence=face_confidence,
                face_mappings=face_mappings,
                adaptive_processor=adaptive_processor
            )
            try:
                result_frames = pipeline.process_frames(frames)
            finally:
                pipeline.shutdown()
        else:
            # Traditional batch processing (fallback or small GIFs)
            logger.info("Using traditional batch processing")
            face_tracker = FaceTracker(iou_threshold=0.3)
            
            # Set stage for traditional processing
            progress.set_stage("Face Swapping")
            
            # Use config for batch size
            batch_size = config.batch_size
            result_frames = []
            total_batches = (len(frames) + batch_size - 1) // batch_size
            logger.info("Processing %d frames in batches of %d...", len(frames), batch_size)
            
            # No lock needed for single-GPU processing (no threading)
            with progress.track(len(frames), "Processing GIF frames", "frame", "Face Swapping") as pbar:
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i + batch_size]
                    batch_results = process_frames_batch(
                        processor, batch, src_faces, face_confidence,
                        face_mappings, face_tracker, adaptive_processor, swapper_lock=None
                    )
                    result_frames.extend(batch_results)
                    pbar.update(len(batch))
                    pbar.set_postfix(batch=f"{i//batch_size + 1}/{total_batches}")
                    
                    if (i + batch_size) % (batch_size * 5) == 0:  # Log every 5 batches
                        logger.info("Processed %d/%d frames", min(i + batch_size, len(frames)), len(frames))
    
    # Apply face restoration if requested (before enhancement)
    if restore_faces:
        progress.set_stage("⚡ Face Restoration")
        progress.log(f"🔧 Applying GFPGAN restoration (weight={restoration_weight:.2f})...")
        logger.info("Applying GFPGAN face restoration (weight=%.2f)...", restoration_weight)
        restorer = FaceRestorer(device_id=gpu_id)
        try:
            restored_frames = []
            with progress.track(len(result_frames), "Restoring faces", "frame") as pbar:
                for idx, frame in enumerate(result_frames):
                    restored = restorer.restore_faces_in_frame(frame, weight=restoration_weight)
                    restored_frames.append(restored)
                    pbar.update(1)
                    if (idx + 1) % 10 == 0:
                        logger.info("Restored %d/%d frames", idx + 1, len(result_frames))
            result_frames = restored_frames
            progress.log("✅ Restoration complete")
            logger.info("Face restoration completed")
        finally:
            restorer.cleanup()
    
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
        progress.set_stage("✨ Enhancement")
        progress.log(f"🎨 Applying Real-ESRGAN enhancement (scale={outscale}x, model={model_name})...")
        
        # Use temp manager for enhancement temp directories
        enhancement_temp_dir = temp_manager.get_temp_dir("gif") / f"enhance_{Path(dest_path).stem}"
        temp_frames_dir = enhancement_temp_dir / "frames"
        enhanced_output_dir = enhancement_temp_dir / "enhanced"
        
        try:
            # Create temp directories
            temp_frames_dir.mkdir(exist_ok=True, parents=True)
            
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
                    enhanced_output_dir,  # Use temp directory instead of output_dir
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
                # Get original frame dimensions from the first frame file
                first_frame_path = frame_paths[0]
                first_frame = cv2.imread(str(first_frame_path))
                if first_frame is not None:
                    original_height, original_width = first_frame.shape[:2]
                    original_size = (original_width, original_height)
                else:
                    original_size = None
                
                enhanced_frames = enhance_frames_single_gpu(
                    temp_frames_dir,
                    enhanced_output_dir,  # Use temp directory instead of output_dir
                    media_type="gif",
                    tile_size=tile_size,
                    outscale=outscale,
                    gpu_id=gpu_id,
                    model_name=model_name,
                    denoise_strength=denoise_strength,
                    use_fp32=use_fp32,
                    pre_pad=pre_pad,
                    maintain_dimensions=True,  # Keep original size
                    original_size=original_size
                )
            
            # Create enhanced GIF if successful
            if enhanced_frames and len(enhanced_frames) == len(result_frames):
                progress.log("✅ Enhancement complete")
                enhanced_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=enhanced_frames[1:],
                    loop=0,
                    duration=frame_durations
                )
                logger.info("Enhanced GIF saved to %s with %d frames", output_path, len(enhanced_frames))
            else:
                progress.log("⚠️  Enhancement failed, keeping original GIF")
                logger.warning("Enhancement failed, keeping original GIF")
        
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            logger.warning("Keeping original GIF")
        
        finally:
            # Clean up enhancement temp directories
            try:
                if enhancement_temp_dir.exists():
                    shutil.rmtree(enhancement_temp_dir)
                    logger.info("Cleaned up enhancement temp directory: %s", enhancement_temp_dir.name)
            except Exception as e:
                logger.warning("Failed to clean up enhancement temp directory: %s", e)
    
    # Cleanup temporary frames from original GIF extraction
    logger.info("Cleaning up temporary GIF frames directory...")
    try:
        if frames_temp_dir.exists():
            shutil.rmtree(frames_temp_dir)
            logger.info("Cleaned up temporary GIF frames: %s", frames_temp_dir)
    except Exception as e:
        logger.warning("Failed to clean up temp frames directory: %s", e)
    
    return None, str(output_path)
