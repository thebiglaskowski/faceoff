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
    pre_pad: int = 0
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
    
    # Validate images
    source_array = np.array(source_image)
    dest_array = np.array(dest_image)
    logger.info("Source image shape: %s, dtype: %s", source_array.shape, source_array.dtype)
    logger.info("Destination image shape: %s, dtype: %s", dest_array.shape, dest_array.dtype)
    
    # Detect and filter faces
    src_faces = processor.get_faces(source_array)
    dst_faces = processor.get_faces(dest_array)
    
    src_faces = filter_faces_by_confidence(src_faces, face_confidence)
    dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
    
    # Sort faces for consistent ordering
    src_faces = sort_faces_by_position(src_faces)
    dst_faces = sort_faces_by_position(dst_faces)
    
    logger.info("Source faces detected (after filtering): %d", len(src_faces))
    logger.info("Destination faces detected (after filtering): %d", len(dst_faces))
    
    if not src_faces:
        raise ValueError("No faces detected in the source image.")
    if not dst_faces:
        raise ValueError("No faces detected in the destination image.")
    
    # Prepare for face swapping
    swapped = np.array(dest_image.copy(), dtype=np.uint8)
    swapped = np.ascontiguousarray(swapped)
    
    # Apply face swapping using mappings or default behavior
    if face_mappings:
        logger.info("Using face mappings: %s", face_mappings)
        for src_idx, dst_idx in face_mappings:
            if src_idx >= len(src_faces):
                logger.warning("Source face index %d out of range (only %d faces)", src_idx, len(src_faces))
                continue
            if dst_idx >= len(dst_faces):
                logger.warning("Destination face index %d out of range (only %d faces)", dst_idx, len(dst_faces))
                continue
            try:
                swapped = processor.swapper.get(swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True)
                logger.info("Swapped source face %d → destination face %d", src_idx, dst_idx)
            except Exception as e:
                logger.error("Error swapping face %d → %d: %s", src_idx, dst_idx, e)
    else:
        # Default: swap first source face to all destination faces
        for face in dst_faces:
            try:
                swapped = processor.swapper.get(swapped, face, src_faces[0], paste_back=True)
            except Exception as e:
                logger.error("Error during face swapping: %s", e)
                raise
    
    # Save swapped image with timestamp-based unique name
    import time
    timestamp = int(time.time() * 1000)
    output_path = output_dir / f"swapped_{timestamp}.png"
    Image.fromarray(np.uint8(swapped)).save(output_path)
    
    # Apply enhancement if requested
    if enhance:
        enhance_image_single_gpu(output_path, output_dir, tile_size, outscale, gpu_id, model_name, denoise_strength, use_fp32, pre_pad)
    
    return str(output_path), None
