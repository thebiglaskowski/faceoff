"""
Media processing orchestrator - unified entry point for all media types.
"""
import logging
import numpy as np
from dataclasses import dataclass
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
from utils.output_metadata import save_output_metadata

logger = logging.getLogger("FaceOff")


@dataclass
class ProcessOptions:
    """Parameters passed to process_media for face swapping and enhancement."""
    source_image: np.ndarray
    dest_path: str
    media_type: str
    output_dir: str
    enhance: bool = False
    tile_size: int = 256
    outscale: int = 4
    face_confidence: float = 0.5
    gpu_selection: Optional[str] = None
    face_mappings: Optional[List[Tuple[int, int]]] = None
    model_name: str = "RealESRGAN_x4plus"
    denoise_strength: float = 0.5
    use_fp32: bool = False
    pre_pad: int = 0
    restore_faces: bool = False
    restoration_weight: float = 0.5
    enhancement_model: str = "RealESRGAN"
    restoration_model: str = "GFPGAN"
    tensorrt_fp16: bool = True
    model_display: Optional[str] = None


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


def process_media(opts: ProcessOptions) -> Tuple[Optional[str], Optional[str]]:
    """
    Process media with face swapping and optional enhancement.

    Args:
        opts: ProcessOptions dataclass with all processing parameters.

    Returns:
        Tuple of (image_output_path, video_output_path)
        - For images: (path, None)
        - For videos/GIFs: (None, path)

    Raises:
        ValueError: If unsupported media type
        RuntimeError: If processing fails
    """
    # Prepare output directory structure
    output_path = Path(opts.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create media-type specific subdirectories
    image_output_dir = output_path / "image"
    gif_output_dir = output_path / "gif"
    video_output_dir = output_path / "video"
    
    image_output_dir.mkdir(exist_ok=True)
    gif_output_dir.mkdir(exist_ok=True)
    video_output_dir.mkdir(exist_ok=True)
    
    # Determine the specific output directory based on media type
    if opts.media_type == "image":
        specific_output_path = image_output_dir
    elif opts.media_type == "gif":
        specific_output_path = gif_output_dir
    elif opts.media_type == "video":
        specific_output_path = video_output_dir
    else:
        specific_output_path = output_path  # Fallback to root outputs
    
    # Parse GPU selection
    device_ids = parse_gpu_selection(opts.gpu_selection)
    
    if opts.media_type == "image" and len(device_ids) > 1:
        logger.info(
            "Image processing uses single GPU %d (%d GPUs available)",
            device_ids[0],
            len(device_ids),
        )
        device_ids = [device_ids[0]]

    # Multi-GPU face swapping for videos/GIFs uses the model pool
    if len(device_ids) > 1 and opts.media_type in ["video", "gif"]:
        if config.multi_gpu_video_enabled:
            logger.info("Multi-GPU enabled for %s with %d GPUs (using model pool)",
                       opts.media_type.upper(), len(device_ids))
        else:
            logger.info("Multi-GPU disabled in config for %s. Using single GPU: %d",
                       opts.media_type.upper(), device_ids[0])
            device_ids = [device_ids[0]]
    
    # Log face mappings received
    logger.info("process_media received face_mappings: %s", opts.face_mappings)
    
    # Initialize primary processor with automatic TensorRT (will fallback to CUDA if not available)
    logger.info("Initializing MediaProcessor with device(s): %s (TensorRT auto-detected)", device_ids)
    primary_device = device_ids[0] if device_ids else 0
    # CRITICAL: Use config settings for TensorRT and optimization
    # ONNX optimization is disabled in config - it creates corrupted model files during Multi-GPU runs
    # TensorRT is always attempted if available (automatic optimization for face detection)
    processor = MediaProcessor(
        device_id=primary_device,
        use_tensorrt=config.tensorrt_enabled,
        optimize_models=False,
        tensorrt_fp16=opts.tensorrt_fp16,
    )
    
    # Route to appropriate processing function
    try:
        result_path = None
        video_path = None
        
        if opts.media_type == "image":
            result_path, video_path = process_image(
                processor, opts.source_image, opts.dest_path, specific_output_path,
                opts.enhance, opts.tile_size, opts.outscale, opts.face_confidence,
                device_ids, opts.face_mappings, opts.model_name, opts.denoise_strength,
                opts.use_fp32, opts.pre_pad, opts.restore_faces, opts.restoration_weight,
                opts.enhancement_model, opts.restoration_model
            )
        elif opts.media_type == "video":
            result_path, video_path = process_video(
                processor, opts.source_image, opts.dest_path, specific_output_path,
                opts.enhance, opts.tile_size, opts.outscale, opts.face_confidence,
                device_ids, opts.face_mappings, opts.model_name, opts.denoise_strength,
                opts.use_fp32, opts.pre_pad, opts.restore_faces, opts.restoration_weight,
                enhancement_model=opts.enhancement_model,
                restoration_model=opts.restoration_model
            )
        elif opts.media_type == "gif":
            result_path, video_path = process_gif(
                processor, opts.source_image, opts.dest_path, specific_output_path,
                opts.enhance, opts.tile_size, opts.outscale, opts.face_confidence,
                device_ids, opts.face_mappings, opts.model_name, opts.denoise_strength,
                opts.use_fp32, opts.pre_pad, opts.restore_faces, opts.restoration_weight,
                enhancement_model=opts.enhancement_model,
                restoration_model=opts.restoration_model
            )
        else:
            raise ValueError(f"Unsupported media type: {opts.media_type}")
        
        # Log what we got back from processing
        logger.info(f"Processing returned: result_path={result_path}, video_path={video_path}")
        
        # Compress the final output (check both result_path and video_path)
        output_to_compress = result_path or video_path
        if output_to_compress:
            try:
                logger.info("📦 Compressing final output...")
                success, message, stats = compress_media_file(output_to_compress, opts.media_type)
                if success:
                    logger.info(f"✅ Compression complete: {message}")
                else:
                    logger.warning(f"⚠️ Compression failed: {message}")
            except Exception as comp_error:
                logger.error(f"❌ Compression error: {comp_error}", exc_info=True)

        if output_to_compress:
            save_output_metadata(output_to_compress, opts)
        
        return result_path, video_path
    
    except FriendlyError:
        # Re-raise friendly errors as-is
        raise
    except Exception as e:
        # Convert technical errors to friendly errors
        context = {
            'media_type': opts.media_type,
            'enhance': opts.enhance,
            'tile_size': opts.tile_size,
            'outscale': opts.outscale,
            'restore_faces': opts.restore_faces,
            'use_fp32': opts.use_fp32
        }
        friendly_error = ErrorHandler.handle_error(e, context)
        logger.error("%s processing failed: %s", opts.media_type.capitalize(), e)
        raise friendly_error
