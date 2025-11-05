"""
Media processing orchestrator - unified entry point for all media types.
"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from core.media_processor import MediaProcessor
from core.gpu_manager import GPUManager
from processing.image_processing import process_image
from processing.video_processing import process_video
from processing.gif_processing import process_gif

logger = logging.getLogger("FaceOff")


def parse_gpu_selection(gpu_selection: Optional[str]) -> List[int]:
    """
    Parse GPU selection string into list of device IDs.
    
    Args:
        gpu_selection: String like "GPU 0: RTX 4090" or "All GPUs: 2 devices"
        
    Returns:
        List of GPU device IDs
    """
    device_ids = []
    
    if gpu_selection and gpu_selection.startswith("All GPUs"):
        # Use all available GPUs
        if GPUManager.is_available():
            device_ids = list(range(GPUManager.get_device_count()))
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
        device_ids = [0] if GPUManager.is_available() else []
        logger.info("Using default GPU 0 for processing")
    
    return device_ids


def process_media(
    source_image: np.ndarray,
    dest_path: str,
    media_type: str,
    output_dir: str,
    enhance: bool = False,
    tile_size: int = 256,
    outscale: int = 4,
    face_confidence: float = 0.5,
    gpu_selection: Optional[str] = None,
    face_mappings: Optional[List[Tuple[int, int]]] = None,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5
) -> Tuple[Optional[str], Optional[str]]:
    """
    Process media with face swapping and optional enhancement.
    
    Args:
        source_image: Source image as numpy array
        dest_path: Path to destination media file
        media_type: Type of media ("image", "video", or "gif")
        output_dir: Output directory for results
        enhance: Whether to apply Real-ESRGAN enhancement
        tile_size: Tile size for enhancement
        outscale: Upscaling factor for enhancement
        face_confidence: Minimum confidence threshold for face detection
        gpu_selection: GPU selection string
        face_mappings: List of (source_idx, target_idx) tuples for face mapping
        model_name: Real-ESRGAN model to use for enhancement
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        
    Returns:
        Tuple of (result_path, log_message)
        media_type: "image", "video", or "gif"
        output_dir: Output directory for results
        enhance: Whether to apply Real-ESRGAN enhancement
        tile_size: Tile size for enhancement (smaller = less VRAM, slower)
        outscale: Upscaling factor (2 or 4)
        face_confidence: Minimum face detection confidence (0.0-1.0)
        gpu_selection: GPU selection string (e.g., "GPU 0: RTX 4090")
        face_mappings: Optional list of (source_face_idx, dest_face_idx) tuples.
                      If None, swaps first source face to all destination faces.
                      Example: [(0, 0), (1, 1)] = source 0 → dest 0, source 1 → dest 1
                      
    Returns:
        Tuple of (image_output_path, video_output_path)
        - For images: (path, None)
        - For videos/GIFs: (None, path)
        
    Raises:
        ValueError: If unsupported media type
        RuntimeError: If processing fails
    """
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Parse GPU selection
    device_ids = parse_gpu_selection(gpu_selection)
    
    # Initialize primary processor
    logger.info("Initializing MediaProcessor with device(s): %s", device_ids)
    primary_device = device_ids[0] if device_ids else 0
    processor = MediaProcessor(device_id=primary_device)
    
    # Route to appropriate processing function
    try:
        if media_type == "image":
            return process_image(
                processor, source_image, dest_path, output_path,
                enhance, tile_size, outscale, face_confidence,
                device_ids, face_mappings, model_name, denoise_strength
            )
        elif media_type == "video":
            return process_video(
                processor, source_image, dest_path, output_path,
                enhance, tile_size, outscale, face_confidence,
                device_ids, face_mappings, model_name, denoise_strength
            )
        elif media_type == "gif":
            return process_gif(
                processor, source_image, dest_path, output_path,
                enhance, tile_size, outscale, face_confidence,
                device_ids, face_mappings, model_name, denoise_strength
            )
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
    
    except Exception as e:
        logger.error("%s processing failed: %s", media_type.capitalize(), e)
        raise
