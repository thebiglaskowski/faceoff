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
from utils.config_manager import config
from utils.error_handler import ErrorHandler, FriendlyError
from utils.compression import compress_media_file

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
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0,
    restore_faces: bool = False,
    restoration_weight: float = 0.5
) -> Tuple[Optional[str], Optional[str]]:
    """
    Process media with face swapping and optional enhancement.
    
    Args:
        source_image: Source image as numpy array
        dest_path: Path to destination media file
        media_type: Type of media ("image", "video", or "gif")
        output_dir: Output directory for results
        enhance: Whether to apply Real-ESRGAN enhancement
        tile_size: Tile size for enhancement (128-512, lower = less VRAM)
        outscale: Upscaling factor for enhancement (2 or 4)
        face_confidence: Minimum confidence threshold for face detection
        gpu_selection: GPU selection string
        face_mappings: List of (source_idx, target_idx) tuples for face mapping
        model_name: Real-ESRGAN model to use for enhancement
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts
        
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
    # Prepare output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create media-type specific subdirectories
    image_output_dir = output_path / "image"
    gif_output_dir = output_path / "gif"
    video_output_dir = output_path / "video"
    
    image_output_dir.mkdir(exist_ok=True)
    gif_output_dir.mkdir(exist_ok=True)
    video_output_dir.mkdir(exist_ok=True)
    
    # Determine the specific output directory based on media type
    if media_type == "image":
        specific_output_path = image_output_dir
    elif media_type == "gif":
        specific_output_path = gif_output_dir
    elif media_type == "video":
        specific_output_path = video_output_dir
    else:
        specific_output_path = output_path  # Fallback to root outputs
    
    # Parse GPU selection
    device_ids = parse_gpu_selection(gpu_selection)
    
    # TEMPORARY: Force single-GPU for videos/GIFs due to inswapper model threading issues
    # Multi-GPU causes model corruption in face swapping even with locks
    if len(device_ids) > 1 and media_type in ["video", "gif"]:
        logger.warning("Multi-GPU not supported for %s due to model threading limitations. Using single GPU: %d", 
                      media_type.upper(), device_ids[0])
        logger.warning("Face detection will still benefit from optimizations (batching, adaptive resolution)")
        device_ids = [device_ids[0]]
    
    # Log face mappings received
    logger.info("process_media received face_mappings: %s", face_mappings)
    
    # Initialize primary processor with automatic TensorRT (will fallback to CUDA if not available)
    logger.info("Initializing MediaProcessor with device(s): %s (TensorRT auto-detected)", device_ids)
    primary_device = device_ids[0] if device_ids else 0
    # CRITICAL: Use config settings for TensorRT and optimization
    # ONNX optimization is disabled in config - it creates corrupted model files during Multi-GPU runs
    # TensorRT is always attempted if available (automatic optimization for face detection)
    processor = MediaProcessor(
        device_id=primary_device, 
        use_tensorrt=config.tensorrt_enabled, 
        optimize_models=False  # Never optimize - causes corruption
    )
    
    # Route to appropriate processing function
    try:
        result_path = None
        video_path = None
        
        if media_type == "image":
            result_path, video_path = process_image(
                processor, source_image, dest_path, specific_output_path,
                enhance, tile_size, outscale, face_confidence,
                device_ids, face_mappings, model_name, denoise_strength,
                use_fp32, pre_pad, restore_faces, restoration_weight
            )
        elif media_type == "video":
            result_path, video_path = process_video(
                processor, source_image, dest_path, specific_output_path,
                enhance, tile_size, outscale, face_confidence,
                device_ids, face_mappings, model_name, denoise_strength,
                use_fp32, pre_pad, restore_faces, restoration_weight,
                use_async_pipeline=True
            )
        elif media_type == "gif":
            result_path, video_path = process_gif(
                processor, source_image, dest_path, specific_output_path,
                enhance, tile_size, outscale, face_confidence,
                device_ids, face_mappings, model_name, denoise_strength,
                use_fp32, pre_pad, restore_faces, restoration_weight,
                use_async_pipeline=True
            )
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Log what we got back from processing
        logger.info(f"Processing returned: result_path={result_path}, video_path={video_path}")
        
        # Compress the final output (check both result_path and video_path)
        output_to_compress = result_path or video_path
        if output_to_compress:
            try:
                # Pretty section header for compression
                print("\n" + "=" * 60)
                print("  📦 Compression")
                print("=" * 60)
                
                logger.info("�️ Compressing final output...")
                success, message, stats = compress_media_file(output_to_compress, media_type)
                if success:
                    logger.info(f"✅ Compression complete: {message}")
                else:
                    logger.warning(f"⚠️ Compression failed: {message}")
            except Exception as comp_error:
                logger.error(f"❌ Compression error: {comp_error}", exc_info=True)
        
        return result_path, video_path
    
    except FriendlyError:
        # Re-raise friendly errors as-is
        raise
    except Exception as e:
        # Convert technical errors to friendly errors
        context = {
            'media_type': media_type,
            'enhance': enhance,
            'tile_size': tile_size,
            'outscale': outscale,
            'restore_faces': restore_faces,
            'use_fp32': use_fp32
        }
        friendly_error = ErrorHandler.handle_error(e, context)
        logger.error("%s processing failed: %s", media_type.capitalize(), e)
        raise friendly_error
