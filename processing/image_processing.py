"""
Image processing with face swapping and enhancement.
"""
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position
from processing.enhancement import enhance_image_single_gpu
from processing.face_restoration import FaceRestorer
from utils.progress import get_progress_tracker

logger = logging.getLogger("FaceOff")


def process_image(
    processor,
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
    restoration_weight: float = 0.5
) -> Tuple[Optional[str], None]:
    """
    Process a single image with face swapping.
    
    Args:
        processor: MediaProcessor instance
        source_image: Source image as numpy array
        dest_path: Path to destination image
        output_dir: Output directory for results
        enhance: Whether to apply Real-ESRGAN enhancement
        tile_size: Tile size for enhancement (128-512, lower = less VRAM)
        outscale: Upscaling factor for enhancement (2 or 4)
        face_confidence: Minimum face detection confidence
        device_ids: List of GPU device IDs (only first is used for images)
        face_mappings: Optional list of (source_idx, dest_idx) tuples
        model_name: Real-ESRGAN model to use for enhancement
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts
        face_mappings: Optional list of (source_face_idx, dest_face_idx) tuples.
                      If None, uses default (first source face to all destination faces)
                      
    Returns:
        Tuple of (output_path, None)
    """
    # For single images, only use first GPU
    if not device_ids:
        device_ids = [0]
    gpu_id = device_ids[0]
    
    dest_image = Image.open(dest_path).convert("RGB")
    logger.info("inswapper-shape: %s", processor.swapper.input_shape)
    
    # Get progress tracker
    progress = get_progress_tracker()
    
    # Validate images
    source_array = np.array(source_image)
    dest_array = np.array(dest_image)
    logger.info("Source image shape: %s, dtype: %s", source_array.shape, source_array.dtype)
    logger.info("Destination image shape: %s, dtype: %s", dest_array.shape, dest_array.dtype)
    
    # Stage 1: Face Detection
    progress.set_stage("🔍 Face Detection")
    progress.log("📸 Detecting faces in source and target images...")
    
    # Detect and filter faces
    src_faces = processor.get_faces(source_array)
    dst_faces = processor.get_faces(dest_array)
    
    src_faces = filter_faces_by_confidence(src_faces, face_confidence)
    dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
    
    # Sort faces for consistent ordering
    src_faces = sort_faces_by_position(src_faces)
    dst_faces = sort_faces_by_position(dst_faces)
    
    progress.log(f"✅ Found {len(src_faces)} source face(s), {len(dst_faces)} target face(s)")
    logger.info("Source faces detected (after filtering): %d", len(src_faces))
    logger.info("Destination faces detected (after filtering): %d", len(dst_faces))
    
    if not src_faces:
        raise ValueError("No faces detected in the source image.")
    if not dst_faces:
        raise ValueError("No faces detected in the destination image.")
    
    # Stage 2: Face Swapping
    progress.set_stage("🔄 Face Swapping")
    
    # Prepare for face swapping
    swapped = np.array(dest_image.copy(), dtype=np.uint8)
    swapped = np.ascontiguousarray(swapped)
    
    # Apply face swapping using mappings or default behavior
    if face_mappings:
        progress.log(f"🎯 Using custom face mappings: {face_mappings}")
        logger.info("Using face mappings: %s", face_mappings)
        with progress.track(len(face_mappings), "Swapping faces", "face") as pbar:
            for src_idx, dst_idx in face_mappings:
                if src_idx >= len(src_faces):
                    msg = f"Source face index {src_idx} out of range (only {len(src_faces)} faces)"
                    progress.log(f"⚠️  {msg}")
                    logger.warning(msg)
                    pbar.update(1)
                    continue
                if dst_idx >= len(dst_faces):
                    msg = f"Destination face index {dst_idx} out of range (only {len(dst_faces)} faces)"
                    progress.log(f"⚠️  {msg}")
                    logger.warning(msg)
                    pbar.update(1)
                    continue
                try:
                    swapped = processor.swapper.get(swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True)
                    logger.info("Swapped source face %d → destination face %d", src_idx, dst_idx)
                except Exception as e:
                    logger.error("Error swapping face %d → %d: %s", src_idx, dst_idx, e)
                pbar.update(1)
    else:
        # Default: swap first source face to all destination faces
        progress.log(f"🔁 Swapping all faces ({len(dst_faces)} face(s))...")
        with progress.track(len(dst_faces), "Swapping faces", "face") as pbar:
            for face in dst_faces:
                try:
                    swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
                except Exception as e:
                    logger.error("Error during face swapping: %s", e)
                    raise
                pbar.update(1)
    
    progress.log("✅ Face swapping complete")
    
    # Stage 3: Face Restoration (optional)
    if restore_faces:
        progress.set_stage("⚡ Face Restoration")
        progress.log(f"🔧 Applying GFPGAN restoration (weight={restoration_weight:.2f})...")
        logger.info("Applying GFPGAN face restoration (weight=%.2f)...", restoration_weight)
        restorer = FaceRestorer(device_id=gpu_id)
        try:
            swapped = restorer.restore_faces_in_frame(swapped, weight=restoration_weight)
            progress.log("✅ Restoration complete")
            logger.info("Face restoration completed")
        finally:
            restorer.cleanup()
    
    # Save swapped image with timestamp-based unique name
    import time
    timestamp = int(time.time() * 1000)
    output_path = output_dir / f"swapped_{timestamp}.png"
    Image.fromarray(np.uint8(swapped)).save(output_path)
    
    # Stage 4: Enhancement (optional)
    if enhance:
        progress.set_stage("✨ Enhancement")
        progress.log(f"🎨 Applying Real-ESRGAN enhancement (scale={outscale}x, model={model_name})...")
        enhance_image_single_gpu(output_path, output_dir, tile_size, outscale, gpu_id, model_name, denoise_strength, use_fp32, pre_pad)
        progress.log("✅ Enhancement complete")
    
    return str(output_path), None
