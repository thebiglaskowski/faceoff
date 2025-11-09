"""
Video processing with face swapping and enhancement.
"""
import cv2
import logging
import numpy as np
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip, ImageSequenceClip
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position, FaceTracker
from core.media_processor import MediaProcessor
from processing.enhancement import enhance_frames_single_gpu, enhance_frames_multi_gpu
from processing.face_restoration import FaceRestorer
from processing.resolution_adaptive import ResolutionAdaptiveProcessor
from processing.async_pipeline import AsyncPipeline
from utils.config_manager import config
from utils.temp_manager import get_temp_manager
from utils.progress import get_progress_tracker, create_stage_tracker

logger = logging.getLogger("FaceOff")


def process_frames_batch(
    processor: MediaProcessor,
    frames: List[np.ndarray],
    src_faces: List,
    face_confidence: float,
    face_mappings: Optional[List[Tuple[int, int]]] = None,
    face_tracker: Optional[FaceTracker] = None,
    adaptive_processor: Optional[ResolutionAdaptiveProcessor] = None,
    swapper_lock: Optional[threading.Lock] = None
) -> List[np.ndarray]:
    """
    Process a batch of frames for face swapping (better GPU utilization).
    
    Args:
        processor: MediaProcessor instance
        frames: Batch of frames to process
        src_faces: Source faces (pre-detected)
        face_confidence: Minimum face detection confidence
        face_mappings: Optional face mapping rules
        face_tracker: Face tracker for maintaining stable IDs
        adaptive_processor: Optional resolution-adaptive processor for faster detection
        swapper_lock: Optional threading lock for thread-safe swapper access
        
    Returns:
        List of processed frames with swapped faces
    """
    # Batch detect faces in all frames (with optional resolution-adaptive detection)
    if adaptive_processor:
        all_dst_faces = [adaptive_processor.detect_faces_adaptive(processor, frame) for frame in frames]
    else:
        all_dst_faces = processor.get_faces_batch(frames)
    
    # Filter by confidence
    all_dst_faces = [filter_faces_by_confidence(faces, face_confidence) for faces in all_dst_faces]
    
    # Track faces if tracker provided
    if face_tracker:
        all_dst_faces = [face_tracker.track_faces(faces) for faces in all_dst_faces]
    
    # Prepare batch data for swapping
    results = []
    for frame, dst_faces in zip(frames, all_dst_faces):
        swapped = frame.copy()
        
        # Apply face mappings or default behavior
        if face_mappings:
            for src_idx, dst_idx in face_mappings:
                if src_idx < len(src_faces) and dst_idx < len(dst_faces) and dst_faces[dst_idx] is not None:
                    # Use lock if provided to prevent concurrent access to swapper
                    if swapper_lock:
                        with swapper_lock:
                            swapped = processor.swapper.get(swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True)
                    else:
                        swapped = processor.swapper.get(swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True)
        else:
            # Default: swap first source face to all destination faces
            for face in dst_faces:
                if src_faces and face is not None:
                    # Use lock if provided for Multi-GPU processing (currently disabled for videos/GIFs)
                    if swapper_lock:
                        with swapper_lock:
                            swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
                    else:
                        swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
        
        results.append(swapped)
    
    return results


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
    pre_pad: int = 0,
    restore_faces: bool = False,
    restoration_weight: float = 0.5,
    adaptive_detection: bool = None,
    detection_scale: float = None,
    use_async_pipeline: bool = True
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
    
    # Create progress tracker for all processing stages
    progress = get_progress_tracker()
    
    # Initialize resolution-adaptive processor if enabled
    adaptive_processor = None
    if adaptive_detection:
        adaptive_processor = ResolutionAdaptiveProcessor(
            detection_scale=detection_scale
        )
        logger.info("Resolution-adaptive detection enabled (scale=%.2f)", detection_scale)
    
    # Multi-GPU setup - now with batching support
    if len(device_ids) > 1:
        logger.info("Multi-GPU Video Mode: %d GPUs: %s (batched processing)", len(device_ids), device_ids)
        
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
            proc_src = proc.get_faces(source_image)
            proc_src = filter_faces_by_confidence(proc_src, face_confidence)
            proc_src = sort_faces_by_position(proc_src)
            processor_src_faces.append(proc_src)
        
        progress.log(f"✅ Found {len(processor_src_faces[0])} source face(s)")
        logger.info("Source faces detected (after filtering): %d", len(processor_src_faces[0]))
        
        # Initialize face tracker for stable face IDs across frames
        face_tracker = FaceTracker(iou_threshold=0.3)
        
        # Use config for batch size
        batch_size = config.batch_size
        num_gpus = len(processors)
        
        # Distribute frames to GPUs (round-robin)
        gpu_frame_batches = [[] for _ in range(num_gpus)]
        gpu_frame_indices = [[] for _ in range(num_gpus)]
        
        for idx, frame in enumerate(frames):
            gpu_idx = idx % num_gpus
            gpu_frame_batches[gpu_idx].append(frame)
            gpu_frame_indices[gpu_idx].append(idx)
        
        logger.info("Processing %d frames across %d GPUs in batches of %d...", 
                   len(frames), num_gpus, batch_size)        
        # Process each GPU's frames in parallel using threads
        def process_gpu_frames(gpu_idx):
            proc = processors[gpu_idx]
            src_faces = processor_src_faces[gpu_idx]
            gpu_frames = gpu_frame_batches[gpu_idx]
            
            # Each GPU needs its own face tracker to avoid thread-safety issues
            gpu_face_tracker = FaceTracker(iou_threshold=0.3)
            
            gpu_results = []
            for i in range(0, len(gpu_frames), batch_size):
                batch = gpu_frames[i:i + batch_size]
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
        result = [None] * len(frames)
        for gpu_idx in range(num_gpus):
            indices = gpu_frame_indices[gpu_idx]
            gpu_frames = gpu_results_dict[gpu_idx]
            for original_idx, frame in zip(indices, gpu_frames):
                result[original_idx] = frame
        
        logger.info("Multi-GPU Video processing complete: %d frames", len(result))
    
    else:
        # Single GPU processing with frame batching
        logger.info("Single GPU Video Mode: GPU %d (batched processing)", device_ids[0])
        
        # Stage 1: Face Detection
        progress.set_stage("🔍 Face Detection")
        progress.log("📸 Detecting faces in source image...")
        
        src_faces = processor.get_faces(source_image)
        src_faces = filter_faces_by_confidence(src_faces, face_confidence)
        src_faces = sort_faces_by_position(src_faces)
        
        progress.log(f"✅ Found {len(src_faces)} source face(s)")
        logger.info("Source faces detected (after filtering): %d", len(src_faces))
        
        # Use config defaults if not specified
        if adaptive_detection is None:
            adaptive_detection = config.adaptive_detection_enabled
        if detection_scale is None:
            detection_scale = config.detection_scale
        
        # Initialize resolution-adaptive processor if enabled
        adaptive_processor = None
        if adaptive_detection:
            logger.info("Resolution-adaptive processing enabled (scale=%.2f)", detection_scale)
            adaptive_processor = ResolutionAdaptiveProcessor(detection_scale=detection_scale)
        
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
                result = pipeline.process_frames(frames)
            finally:
                pipeline.shutdown()
        else:
            # Traditional batch processing (fallback or small videos)
            logger.info("Using traditional batch processing")
            face_tracker = FaceTracker(iou_threshold=0.3)
            
            # Use config for batch size
            batch_size = config.batch_size
            result = []
            
            # Set stage for traditional processing
            progress.set_stage("Face Swapping")
            
            total_batches = (len(frames) + batch_size - 1) // batch_size
            logger.info("Processing %d frames in batches of %d...", len(frames), batch_size)
            
            with progress.track(len(frames), "Processing video frames", "frame", "Face Swapping") as pbar:
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i + batch_size]
                    batch_results = process_frames_batch(
                        processor, batch, src_faces, face_confidence, 
                        face_mappings, face_tracker, adaptive_processor
                    )
                    result.extend(batch_results)
                    pbar.update(len(batch))
                    pbar.set_postfix(batch=f"{i//batch_size + 1}/{total_batches}")
                    
                    if (i + batch_size) % (batch_size * 10) == 0:  # Log every 10 batches
                        logger.info("Processed %d/%d frames", min(i + batch_size, len(frames)), len(frames))
    
    # Apply face restoration if requested (before creating video)
    if restore_faces:
        progress.set_stage("⚡ Face Restoration")
        progress.log(f"🔧 Applying GFPGAN restoration (weight={restoration_weight:.2f})...")
        logger.info("Applying GFPGAN face restoration (weight=%.2f)...", restoration_weight)
        restorer = FaceRestorer(device_id=device_ids[0] if device_ids else 0)
        try:
            restored_result = []
            with progress.track(len(result), "Restoring faces", "frame") as pbar:
                for idx, frame in enumerate(result):
                    restored = restorer.restore_faces_in_frame(frame, weight=restoration_weight)
                    restored_result.append(restored)
                    pbar.update(1)
                    if (idx + 1) % 50 == 0:
                        logger.info("Restored %d/%d frames", idx + 1, len(result))
            result = restored_result
            progress.log("✅ Restoration complete")
            logger.info("Face restoration completed")
        finally:
            restorer.cleanup()
    
    # Create processed video with timestamp-based unique name
    import time
    timestamp = int(time.time() * 1000)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use absolute path and ensure it's properly formatted for Windows
    output_path = output_dir.resolve() / f"swapped_{timestamp}.mp4"
    logger.info(f"Output video path: {output_path}")
    
    # Ensure all frames are proper numpy arrays with correct dtype and shape
    processed_frames = []
    for idx, f in enumerate(result):
        frame = np.array(f)
        # Ensure uint8 dtype
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        # Ensure RGB format (H, W, 3)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Verify frame shape
        if idx == 0:
            logger.info(f"First frame shape: {frame.shape}, dtype: {frame.dtype}")
        
        # Ensure contiguous array (required by FFmpeg)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        processed_frames.append(frame)
    
    logger.info(f"Creating video from {len(processed_frames)} frames at {fps} fps")
    
    # Try to create video clip - if it fails, we'll use temp files approach
    try:
        clip = ImageSequenceClip(processed_frames, fps=fps)
        
        # Add audio back
        if audio:
            clip = clip.set_audio(audio)
            logger.info("Audio track successfully added to the processed video.")
        else:
            logger.warning("Processed video will not contain audio.")
        
        # Write video with explicit settings for Windows/high-resolution compatibility
        clip.write_videofile(
            str(output_path), 
            codec="libx264",
            audio_codec="aac" if audio else None,
            preset='medium',
            ffmpeg_params=['-pix_fmt', 'yuv420p'],  # Explicit pixel format for compatibility
            verbose=False,
            logger=None
        )
        logger.info(f"Video successfully written to {output_path}")
        
    except Exception as e:
        logger.error(f"MoviePy ImageSequenceClip failed: {e}")
        logger.info("Falling back to temp file approach...")
        
        # Fallback: Save frames to temp directory and use FFmpeg directly
        temp_manager = get_temp_manager()
        temp_frames_dir = temp_manager.get_temp_dir("video") / f"fallback_{Path(dest_path).stem}"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save all frames as images
            logger.info(f"Saving {len(processed_frames)} frames to temp directory...")
            for idx, frame in enumerate(processed_frames):
                frame_path = temp_frames_dir / f"frame_{idx:06d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Use FFmpeg directly to create video
            import subprocess
            logger.info("Creating video with FFmpeg...")
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', str(temp_frames_dir / 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
            ]
            
            if audio:
                # Extract audio from original video
                audio_path = temp_frames_dir / "audio.aac"
                audio.write_audiofile(str(audio_path), codec='aac', verbose=False, logger=None)
                cmd.extend(['-i', str(audio_path), '-c:a', 'aac', '-b:a', '192k'])
            
            cmd.append(str(output_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            logger.info(f"Video successfully created with FFmpeg: {output_path}")
            
        finally:
            # Cleanup temp directory
            if temp_frames_dir.exists():
                shutil.rmtree(temp_frames_dir)
                logger.debug(f"Cleaned up temp frames directory")
    
    # Apply enhancement if requested
    if enhance:
        progress.set_stage("✨ Enhancement")
        progress.log(f"🎨 Applying Real-ESRGAN enhancement (scale={outscale}x, model={model_name})...")
        
        # Use temp manager for enhancement temp directories
        temp_manager = get_temp_manager()
        enhancement_temp_dir = temp_manager.get_temp_dir("video") / f"enhance_{Path(output_path).stem}"
        temp_frames_dir = enhancement_temp_dir / "frames"
        enhanced_output_dir = enhancement_temp_dir / "enhanced"
        
        try:
            # Extract frames for enhancement
            temp_frames_dir.mkdir(exist_ok=True, parents=True)
            
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
                progress.log("✅ Enhancement complete")
                enhanced_frames, enhanced_clip = result
                # Overwrite original with enhanced version
                enhanced_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
                logger.info("Enhanced video saved to %s with %d frames", output_path, len(enhanced_frames))
            else:
                progress.log("⚠️  Enhancement failed, keeping original video")
                logger.warning("Enhancement failed, keeping original video")
        
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            logger.warning("Keeping original video")
        
        finally:
            # Clean up enhancement temp directories
            try:
                if enhancement_temp_dir.exists():
                    shutil.rmtree(enhancement_temp_dir)
                    logger.info("Cleaned up enhancement temp directory: %s", enhancement_temp_dir.name)
            except Exception as e:
                logger.warning("Failed to clean up enhancement temp directory: %s", e)
    
    return None, str(output_path)
