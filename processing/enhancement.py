"""
Image and video enhancement using Real-ESRGAN and GFPGAN.

This module consolidates enhancement functionality from:
- enhancement_utils.py (EnhancementProcessor class)
- media_processing.py (Real-ESRGAN CLI wrapper functions)
"""
import gc
import logging
import numpy as np
import shutil
import subprocess
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union
from moviepy.editor import ImageSequenceClip
import torch

logger = logging.getLogger("FaceOff")

# Import Real-ESRGAN and GFPGAN
try:
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
    MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Real-ESRGAN or GFPGAN not available. Enhancement will be disabled.")
    MODELS_AVAILABLE = False


class EnhancementProcessor:
    """
    Image enhancement using Real-ESRGAN with integrated GFPGAN face enhancement.
    
    This class provides a Python API for enhancement (in-memory processing).
    For batch processing of frames, use the CLI wrapper functions instead.
    """
    
    def __init__(self, device: str = "cuda", tile_size: int = 256, outscale: int = 4):
        """
        Initialize enhancement processor.
        
        Args:
            device: Device to use ("cuda" or "cpu")
            tile_size: Tile size for processing (smaller = less VRAM, slower)
            outscale: Upscaling factor (2 or 4)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tile_size = tile_size
        self.outscale = outscale
        self.realesrganer = None
        self.gfpganer = None
        
        if MODELS_AVAILABLE:
            self._init_real_esrgan()
            self._init_gfpgan()
            
            # Set integrated face enhancement on RealESRGANer
            if self.realesrganer and self.gfpganer:
                self.realesrganer.face_enhance = True
                self.realesrganer.face_enhancer = self.gfpganer
                logger.info("Integrated face enhancement enabled")
    
    def _init_real_esrgan(self) -> None:
        """Initialize Real-ESRGAN model."""
        try:
            weight_path = "models/realesrgan/weights/RealESRGAN_x4plus.pth"
            if not Path(weight_path).exists():
                raise FileNotFoundError(f"RealESRGAN weight not found: {weight_path}")
            
            logger.info("Loading Real-ESRGAN from %s (device: %s)", weight_path, self.device)
            self.realesrganer = RealESRGANer(
                scale=4,
                model_path=weight_path,
                tile=self.tile_size,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            logger.info("Real-ESRGAN initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Real-ESRGAN: %s", e)
            self.realesrganer = None
    
    def _init_gfpgan(self) -> None:
        """Initialize GFPGAN model for face restoration."""
        try:
            weight_path = "models/gfpgan/weights/GFPGANv1.4.pth"
            if not Path(weight_path).exists():
                raise FileNotFoundError(f"GFPGAN weight not found: {weight_path}")
            
            logger.info("Loading GFPGAN from %s", weight_path)
            self.gfpganer = GFPGANer(
                model_path=weight_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            logger.info("GFPGAN initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize GFPGAN: %s", e)
            self.gfpganer = None
    
    def enhance(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhance a single image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Enhanced image as numpy array, or None if enhancement failed
        """
        if not self.realesrganer:
            logger.error("Real-ESRGAN not initialized")
            return None
        
        try:
            enhanced_image, _ = self.realesrganer.enhance(image, outscale=self.outscale)
            return enhanced_image
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            return None


def apply_realesrgan_cli(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    tile_size: int = 256,
    outscale: int = 4,
    gpu_id: int = 0,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5
) -> bool:
    """
    Apply Real-ESRGAN enhancement using the CLI wrapper.
    
    This is the recommended method for batch processing (videos/GIFs)
    as it's more memory-efficient than the Python API.
    
    Args:
        input_path: Path to input file or directory
        output_dir: Output directory for enhanced results
        tile_size: Tile size for processing
        outscale: Upscaling factor
        gpu_id: GPU device ID
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        
    Returns:
        True if successful, False otherwise
    """
    # Clear GPU cache before enhancement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    command = [
        'python',
        'G:/My Drive/scripts/faceoff/external/Real-ESRGAN/inference_realesrgan.py',
        '-n', model_name,
        '-i', str(input_path),
        '-o', str(output_dir),
        '--outscale', str(outscale),
        '--tile', str(tile_size),
        '--gpu-id', str(gpu_id),
        '--face_enhance'
    ]
    
    # Add denoise strength only for models that support it
    if model_name == "realesr-general-x4v3":
        command.extend(['--denoise_strength', str(denoise_strength)])
    
    logger.info("Running Real-ESRGAN (model=%s) on GPU %d: tile=%d, outscale=%d", 
                model_name, gpu_id, tile_size, outscale)
    if model_name == "realesr-general-x4v3":
        logger.info("Denoise strength: %.2f", denoise_strength)
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        logger.info("✅ Enhancement complete on GPU %d", gpu_id)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Enhancement failed on GPU %d: %s", gpu_id, e)
        return False


def enhance_image_single_gpu(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    tile_size: int = 256,
    outscale: int = 4,
    gpu_id: int = 0,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5
) -> Optional[Path]:
    """
    Enhance a single image file using Real-ESRGAN.
    
    Args:
        input_path: Path to input image
        output_dir: Output directory
        tile_size: Tile size for processing
        outscale: Upscaling factor
        gpu_id: GPU device ID
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        
    Returns:
        Path to enhanced image, or None if failed
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not apply_realesrgan_cli(input_path, output_dir, tile_size, outscale, gpu_id, model_name, denoise_strength):
        return None
    
    # Real-ESRGAN saves with _out suffix
    enhanced_path = output_dir / f"{input_path.stem}_out.png"
    
    if enhanced_path.exists():
        # Replace original with enhanced version
        shutil.move(str(enhanced_path), str(input_path))
        logger.info("Enhanced image saved to %s", input_path)
        return input_path
    else:
        logger.warning("Enhanced image not found at %s", enhanced_path)
        return None


def enhance_frames_single_gpu(
    frames_dir: Union[str, Path],
    output_dir: Union[str, Path],
    media_type: str,
    fps: Optional[float] = None,
    audio = None,
    tile_size: int = 256,
    outscale: int = 4,
    gpu_id: int = 0,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5
) -> Optional[Union[Tuple[List, ImageSequenceClip], List[Image.Image]]]:
    """
    Enhance video or GIF frames using single GPU.
    
    Args:
        frames_dir: Directory containing frames
        output_dir: Output directory
        media_type: "video" or "gif"
        fps: Frames per second (for video)
        audio: Audio track (for video)
        tile_size: Tile size for processing
        outscale: Upscaling factor
        gpu_id: GPU device ID
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        
    Returns:
        For video: (frames, enhanced_clip)
        For GIF: list of PIL Images
        None if failed
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    enhanced_dir = output_dir / f"temp_{media_type}_enhanced"
    
    # Remove old enhanced directory
    if enhanced_dir.exists():
        shutil.rmtree(enhanced_dir)
    
    # Run enhancement
    if not apply_realesrgan_cli(frames_dir, enhanced_dir, tile_size, outscale, gpu_id, model_name, denoise_strength):
        return None
    
    # Load enhanced frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    expected_frames = len(frame_files)
    enhanced_frames = []
    
    for i in range(expected_frames):
        # Frame naming differs between video and GIF
        if media_type == "video":
            enhanced_path = enhanced_dir / f"frame_{i:06d}_out.png"
        else:  # gif
            enhanced_path = enhanced_dir / f"frame_{i:04d}_out.png"
        
        if enhanced_path.exists():
            if media_type == "gif":
                enhanced_frames.append(Image.open(enhanced_path).copy())
            else:  # video
                enhanced_frames.append(np.array(Image.open(enhanced_path)))
        else:
            logger.error("Enhanced frame not found: %s", enhanced_path)
            return None
    
    # Handle frame count mismatch (Real-ESRGAN sometimes produces +/- 1-2 frames)
    actual_frames = len(enhanced_frames)
    frame_diff = abs(actual_frames - expected_frames)
    
    if frame_diff > 2:
        logger.error("Frame count mismatch too large. Expected %d, got %d", expected_frames, actual_frames)
        return None
    
    if frame_diff > 0:
        logger.warning("Frame count mismatch. Expected %d, got %d. Adjusting...", expected_frames, actual_frames)
        if actual_frames > expected_frames:
            enhanced_frames = enhanced_frames[:expected_frames]
        elif actual_frames < expected_frames:
            while len(enhanced_frames) < expected_frames:
                enhanced_frames.append(enhanced_frames[-1])
    
    # Return format depends on media type
    if media_type == "video":
        enhanced_clip = ImageSequenceClip(enhanced_frames, fps=fps)
        if audio:
            enhanced_clip = enhanced_clip.set_audio(audio)
        logger.info("Enhanced video: %d frames", len(enhanced_frames))
        return enhanced_frames, enhanced_clip
    else:  # gif
        logger.info("Enhanced GIF: %d frames", len(enhanced_frames))
        return enhanced_frames


def enhance_frames_multi_gpu(
    frame_paths: List[Path],
    output_dir: Path,
    media_type: str,
    device_ids: List[int],
    fps: Optional[float] = None,
    audio = None,
    frame_durations: Optional[List[int]] = None,
    tile_size: int = 256,
    outscale: int = 4,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5
) -> Optional[Union[Tuple[List, ImageSequenceClip], List[Image.Image]]]:
    """
    Enhance frames using multiple GPUs for parallel processing.
    
    Args:
        frame_paths: List of frame paths to enhance
        output_dir: Output directory
        media_type: "video" or "gif"
        device_ids: List of GPU device IDs
        fps: Frames per second (for video)
        audio: Audio track (for video)
        frame_durations: Frame durations (for GIF)
        tile_size: Tile size for processing
        outscale: Upscaling factor
        model_name: Real-ESRGAN model to use
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        
    Returns:
        For video: (frames, enhanced_clip)
        For GIF: list of PIL Images
        None if failed
    """
    if len(device_ids) <= 1:
        # Fall back to single GPU
        return enhance_frames_single_gpu(
            frames_dir=frame_paths[0].parent,
            output_dir=output_dir,
            media_type=media_type,
            fps=fps,
            audio=audio,
            tile_size=tile_size,
            outscale=outscale,
            gpu_id=device_ids[0] if device_ids else 0,
            model_name=model_name,
            denoise_strength=denoise_strength
        )
    
    logger.info("🚀 Multi-GPU Enhancement: %d frames across %d GPUs", len(frame_paths), len(device_ids))
    
    # Split frames into chunks for each GPU
    chunks = [[] for _ in device_ids]
    for idx, frame_path in enumerate(frame_paths):
        gpu_idx = idx % len(device_ids)
        chunks[gpu_idx].append(frame_path)
    
    logger.info("Split frames: %s", [len(chunk) for chunk in chunks])
    
    # Create temporary directories for each GPU
    temp_dirs = []
    enhanced_dirs = []
    for gpu_idx in range(len(device_ids)):
        temp_dir = output_dir / f"temp_{media_type}_gpu{gpu_idx}_frames"
        enhanced_dir = output_dir / f"temp_{media_type}_gpu{gpu_idx}_enhanced"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        if enhanced_dir.exists():
            shutil.rmtree(enhanced_dir)
        
        temp_dirs.append(temp_dir)
        enhanced_dirs.append(enhanced_dir)
    
    # Copy frames to GPU directories with sequential numbering
    for gpu_idx, chunk in enumerate(chunks):
        for seq_idx, frame_path in enumerate(chunk):
            dest_path = temp_dirs[gpu_idx] / f"frame_{seq_idx:06d}.png"
            shutil.copy(frame_path, dest_path)
    
    # Run enhancement on each GPU in parallel
    def enhance_on_gpu(args):
        gpu_idx, gpu_id, temp_dir, enhanced_dir = args
        logger.info("GPU %d: Enhancing %d frames...", gpu_id, len(chunks[gpu_idx]))
        success = apply_realesrgan_cli(temp_dir, enhanced_dir, tile_size, outscale, gpu_id, model_name, denoise_strength)
        if success:
            logger.info("✅ GPU %d: Enhancement complete", gpu_id)
        return success
    
    enhancement_args = [
        (gpu_idx, device_ids[gpu_idx], temp_dirs[gpu_idx], enhanced_dirs[gpu_idx])
        for gpu_idx in range(len(device_ids))
    ]
    
    try:
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            results = list(executor.map(enhance_on_gpu, enhancement_args))
        
        if not all(results):
            logger.error("Some GPU enhancement tasks failed")
            return None
    except Exception as e:
        logger.error("Multi-GPU enhancement failed: %s", e)
        return None
    
    # Collect enhanced frames in original order
    logger.info("Collecting enhanced frames from all GPUs...")
    enhanced_frames = [None] * len(frame_paths)
    
    for idx, frame_path in enumerate(frame_paths):
        gpu_idx = idx % len(device_ids)
        seq_idx = chunks[gpu_idx].index(frame_path)
        enhanced_frame_path = enhanced_dirs[gpu_idx] / f"frame_{seq_idx:06d}_out.png"
        
        if enhanced_frame_path.exists():
            if media_type == "gif":
                enhanced_frames[idx] = Image.open(enhanced_frame_path).copy()
            else:  # video
                enhanced_frames[idx] = np.array(Image.open(enhanced_frame_path))
        else:
            logger.error("Enhanced frame not found: %s", enhanced_frame_path)
            return None
    
    # Clean up temporary directories
    for temp_dir in temp_dirs + enhanced_dirs:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning("Failed to clean up %s: %s", temp_dir, e)
    
    logger.info("✅ Multi-GPU enhancement complete: %d frames", len(enhanced_frames))
    
    # Return appropriate format
    if media_type == "video":
        enhanced_clip = ImageSequenceClip(enhanced_frames, fps=fps)
        if audio:
            enhanced_clip = enhanced_clip.set_audio(audio)
        return enhanced_frames, enhanced_clip
    else:  # gif
        return enhanced_frames
