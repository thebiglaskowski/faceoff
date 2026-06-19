"""
Image processing with face swapping and enhancement.
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position
from core.media_processor import MediaProcessor
from processing.enhancement import enhance_image_single_gpu
from processing.face_restoration import FaceRestorer
from utils.progress import get_progress_tracker

logger = logging.getLogger("FaceOff")


def _get_restorer(restoration_model: str, gpu_id: int, restoration_weight: float = 0.5):
    """
    Get appropriate face restorer based on model selection.

    Args:
        restoration_model: "GFPGAN" or "CodeFormer"
        gpu_id: GPU device ID
        restoration_weight: Restoration weight (fidelity for CodeFormer)

    Returns:
        Tuple of (restorer_instance, restore_function)
    """
    if restoration_model == "CodeFormer":
        from processing.codeformer_restoration import CodeFormerRestorer

        restorer = CodeFormerRestorer(device_id=gpu_id)
        return restorer, lambda img, w: restorer.restore_faces_in_frame(
            img, fidelity_weight=w
        )
    else:
        # Default to GFPGAN
        restorer = FaceRestorer(device_id=gpu_id)
        return restorer, lambda img, w: restorer.restore_faces_in_frame(img, weight=w)


class ImageProcessor:
    """High-level wrapper for image face-swap processing."""

    def __init__(self, device_id: int = 0):
        import logging
        from core.media_processor import MediaProcessor

        self.logger = logging.getLogger("FaceOff")
        self.processor = MediaProcessor(device_id=device_id)

    def read_image(self, file_path: str) -> np.ndarray:
        """Read image file as numpy array."""
        import cv2

        return cv2.imread(str(file_path), cv2.IMREAD_COLOR)

    def swap_face(self, image: np.ndarray, target_face, source_face) -> np.ndarray:
        """Swap a face using the underlying processor."""
        return self.processor.swapper.get(
            image, target_face, source_face, paste_back=True
        )


def _apply_enhancement(
    output_path: Path,
    output_dir: Path,
    enhancement_model: str,
    model_name: str,
    tile_size: int,
    outscale: int,
    gpu_id: int,
    denoise_strength: float,
    use_fp32: bool,
    pre_pad: int,
) -> Optional[Path]:
    """
    Apply enhancement using the selected model.

    Args:
        output_path: Path to image to enhance
        output_dir: Output directory
        enhancement_model: "RealESRGAN" or "SwinIR"
        model_name: Specific model variant
        tile_size: Tile size for processing
        outscale: Upscaling factor
        gpu_id: GPU device ID
        denoise_strength: Denoise strength
        use_fp32: Use FP32 precision
        pre_pad: Pre-padding size

    Returns:
        Path to enhanced image or None
    """
    if enhancement_model == "SwinIR":
        from processing.swinir_enhancement import (
            enhance_image_swinir_file,
            SWINIR_MODELS,
        )

        # Map model_name to SwinIR equivalent if needed
        if model_name.startswith("Swin"):
            swinir_model = model_name
        else:
            # Default to RealWorld model for best results
            swinir_model = "Swin2SR_RealWorld_x4"

        return enhance_image_swinir_file(
            output_path, output_dir, model_name=swinir_model, gpu_id=gpu_id
        )
    elif enhancement_model == "HAT":
        from processing.hat_enhancement import enhance_image_hat_file, DEFAULT_HAT_MODEL

        if model_name.startswith("HAT_"):
            hat_model = model_name
        else:
            hat_model = DEFAULT_HAT_MODEL

        return enhance_image_hat_file(
            output_path, output_dir, model_name=hat_model, gpu_id=gpu_id
        )
    else:
        # Default to Real-ESRGAN
        return enhance_image_single_gpu(
            output_path,
            output_dir,
            tile_size,
            outscale,
            gpu_id,
            model_name,
            denoise_strength,
            use_fp32,
            pre_pad,
        )


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
    restoration_weight: float = 0.5,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN",
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
        model_name: Enhancement model variant (Real-ESRGAN or SwinIR specific)
        denoise_strength: Denoise strength (0-1, only for realesr-general-x4v3)
        use_fp32: Use FP32 precision instead of FP16
        pre_pad: Pre-padding size to reduce edge artifacts
        restore_faces: Whether to apply face restoration
        restoration_weight: Face restoration weight (0-1)
        enhancement_model: Enhancement framework ("RealESRGAN" or "SwinIR")
        restoration_model: Face restoration model ("GFPGAN" or "CodeFormer")

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
    logger.info(
        "Source image shape: %s, dtype: %s", source_array.shape, source_array.dtype
    )
    logger.info(
        "Destination image shape: %s, dtype: %s", dest_array.shape, dest_array.dtype
    )

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

    progress.log(
        f"✅ Found {len(src_faces)} source face(s), {len(dst_faces)} target face(s)"
    )
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
    # Ensure 3D image for arithmetic (handle possible grayscale 2D inputs)
    if swapped.ndim == 2:
        swapped = np.stack([swapped] * 3, axis=-1)

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
                    swapped = processor.swapper.get(
                        swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True
                    )
                    logger.info(
                        "Swapped source face %d → destination face %d", src_idx, dst_idx
                    )
                except Exception as e:
                    logger.error("Error swapping face %d → %d: %s", src_idx, dst_idx, e)
                pbar.update(1)
    else:
        # Default: swap first source face to all destination faces (batched ONNX)
        progress.log(f"🔁 Swapping all faces ({len(dst_faces)} face(s))...")
        swapped = processor.swap_face_batch(swapped, dst_faces, src_faces[0])

    progress.log("✅ Face swapping complete")

    # Stage 3: Face Restoration (optional)
    if restore_faces:
        progress.set_stage("⚡ Face Restoration")
        progress.log(
            f"🔧 Applying {restoration_model} restoration (weight={restoration_weight:.2f})..."
        )
        logger.info(
            "Applying %s face restoration (weight=%.2f)...",
            restoration_model,
            restoration_weight,
        )
        restorer, restore_fn = _get_restorer(
            restoration_model, gpu_id, restoration_weight
        )
        try:
            swapped = restore_fn(swapped, restoration_weight)
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
        display_model = (
            model_name if enhancement_model == "RealESRGAN" else enhancement_model
        )
        progress.log(
            f"🎨 Applying {enhancement_model} enhancement (scale={outscale}x, model={display_model})..."
        )
        enhanced_path = _apply_enhancement(
            output_path,
            output_dir,
            enhancement_model,
            model_name,
            tile_size,
            outscale,
            gpu_id,
            denoise_strength,
            use_fp32,
            pre_pad,
        )
        progress.log("✅ Enhancement complete")
        # Return the enhanced image path if enhancement succeeded
        if enhanced_path is not None:
            output_path = enhanced_path

    return str(output_path), None
