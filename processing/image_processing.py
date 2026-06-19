"""
Image processing with face swapping and enhancement.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from core.face_processor import filter_faces_by_confidence, sort_faces_by_position
from processing.in_memory_enhancement import InMemoryEnhancer
from processing.restoration_session import RestorationSession
from utils.memory_manager import prepare_for_enhancement
from utils.progress import get_progress_tracker

logger = logging.getLogger("FaceOff")


class ImageProcessor:
    """High-level wrapper for image face-swap processing."""

    def __init__(self, device_id: int = 0):
        from core.media_processor import MediaProcessor

        self.logger = logging.getLogger("FaceOff")
        self.processor = MediaProcessor(device_id=device_id)

    def read_image(self, file_path: str) -> np.ndarray:
        import cv2

        return cv2.imread(str(file_path), cv2.IMREAD_COLOR)

    def swap_face(self, image: np.ndarray, target_face, source_face) -> np.ndarray:
        return self.processor.swapper.get(
            image, target_face, source_face, paste_back=True
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
    """Process a single image with face swapping (in-memory restore/enhance)."""
    if not device_ids:
        device_ids = [0]
    gpu_id = device_ids[0]

    dest_image = Image.open(dest_path).convert("RGB")
    logger.info("inswapper-shape: %s", processor.swapper.input_shape)

    progress = get_progress_tracker()
    source_array = np.array(source_image)
    dest_array = np.array(dest_image)

    progress.set_stage("🔍 Face Detection")
    progress.log("📸 Detecting faces in source and target images...")

    src_faces = processor.get_faces(source_array)
    dst_faces = processor.get_faces(dest_array)
    src_faces = filter_faces_by_confidence(src_faces, face_confidence)
    dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
    src_faces = sort_faces_by_position(src_faces)
    dst_faces = sort_faces_by_position(dst_faces)

    progress.log(
        f"✅ Found {len(src_faces)} source face(s), {len(dst_faces)} target face(s)"
    )

    if not src_faces:
        raise ValueError("No faces detected in the source image.")
    if not dst_faces:
        raise ValueError("No faces detected in the destination image.")

    from utils.validation import validate_face_mappings_or_raise

    validate_face_mappings_or_raise(face_mappings, len(src_faces), len(dst_faces))

    progress.set_stage("🔄 Face Swapping")
    swapped = np.ascontiguousarray(dest_array.astype(np.uint8))
    if swapped.ndim == 2:
        swapped = np.stack([swapped] * 3, axis=-1)

    if face_mappings:
        progress.log(f"🎯 Using custom face mappings: {face_mappings}")
        swaps_applied = 0
        with progress.track(len(face_mappings), "Swapping faces", "face") as pbar:
            for src_idx, dst_idx in face_mappings:
                if src_idx >= len(src_faces) or dst_idx >= len(dst_faces):
                    pbar.update(1)
                    continue
                swapped = processor.swapper.get(
                    swapped, dst_faces[dst_idx], src_faces[src_idx], paste_back=True
                )
                swaps_applied += 1
                pbar.update(1)
        if swaps_applied == 0:
            raise ValueError(
                "Face mappings did not match any detected faces. "
                "Re-detect faces on this target and update mappings."
            )
    else:
        progress.log(f"🔁 Swapping all faces ({len(dst_faces)} face(s))...")
        swapped = processor.swap_face_batch(swapped, dst_faces, src_faces[0])

    progress.log("✅ Face swapping complete")

    restoration_session: Optional[RestorationSession] = None
    try:
        if restore_faces:
            progress.set_stage("⚡ Face Restoration")
            restoration_session = RestorationSession(
                restoration_model, gpu_id, restoration_weight
            )
            swapped = restoration_session.restore_rgb_frame(swapped)
            progress.log("✅ Restoration complete")

        if enhance:
            processor.release_gpu_models()
            prepare_for_enhancement(gpu_id, device_ids=device_ids)
            progress.set_stage("✨ Enhancement")
            progress.log(
                f"🎨 Applying {enhancement_model} enhancement (scale={outscale}x)..."
            )
            enhancer = InMemoryEnhancer(
                enhancement_model,
                model_name,
                device_ids,
                tile_size,
                outscale,
                denoise_strength,
                use_fp32,
                pre_pad,
                face_enhance=not restore_faces,
            )
            swapped = enhancer.enhance_rgb_image(swapped)
            progress.log("✅ Enhancement complete")

        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"swapped_{timestamp}.png"
        Image.fromarray(np.uint8(swapped)).save(output_path)
        return str(output_path), None
    finally:
        if restoration_session:
            restoration_session.close()