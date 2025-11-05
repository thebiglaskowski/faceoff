"""
Face detection, extraction, and processing utilities.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
from utils.constants import DEFAULT_FACE_CONFIDENCE

logger = logging.getLogger("FaceOff")


def filter_faces_by_confidence(faces, threshold: float = DEFAULT_FACE_CONFIDENCE):
    """
    Filter faces by detection confidence threshold.
    
    Args:
        faces: List of detected faces
        threshold: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Filtered list of faces
    """
    if not faces:
        return faces
    filtered = [f for f in faces if (f.det_score if hasattr(f, 'det_score') else 1.0) >= threshold]
    if len(filtered) < len(faces):
        logger.info(
            "Filtered faces: %d/%d meet confidence threshold %.0f%%",
            len(filtered), len(faces), threshold * 100
        )
    return filtered


def sort_faces_by_position(faces):
    """
    Sort faces consistently by position (left-to-right, top-to-bottom).
    This ensures face indices remain consistent across frames.
    
    Args:
        faces: List of face objects with bbox attribute
        
    Returns:
        Sorted list of faces
    """
    if not faces:
        return faces
    # Sort by (x, y) coordinates of top-left corner of bounding box
    return sorted(faces, key=lambda f: (f.bbox[0], f.bbox[1]))


class FaceProcessor:
    """Manages face detection and extraction operations."""
    
    def __init__(self, device_id: int = 0, confidence: float = DEFAULT_FACE_CONFIDENCE):
        """
        Initialize face processor.
        
        Args:
            device_id: GPU device ID
            confidence: Minimum face detection confidence
        """
        from core.media_processor import MediaProcessor
        
        self.device_id = device_id
        self.confidence = confidence
        self._processor = MediaProcessor(device_id=device_id)
        logger.info("FaceProcessor initialized on device %d", device_id)
    
    def detect_faces_info(self, image_path: str) -> str:
        """
        Detect faces in image and return information string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Formatted string with face count and confidence scores
        """
        try:
            # Load and detect faces
            img = self._processor.read_image(image_path)
            all_faces = self._processor.get_faces(img)
            
            # Filter faces by confidence threshold
            faces = filter_faces_by_confidence(all_faces, self.confidence)
            
            if not faces:
                filtered_count = len(all_faces) - len(faces)
                if filtered_count > 0:
                    return (
                        f"⚠️ No faces meet confidence threshold {self.confidence:.0%} "
                        f"(filtered out {filtered_count} low-confidence detection(s))"
                    )
                return "⚠️ No faces detected in source image"
            
            # Sort faces for consistent ordering
            faces = sort_faces_by_position(faces)
            
            # Format face information
            info_lines = [
                f"✅ Detected {len(faces)} face(s) in source image "
                f"(threshold: {self.confidence:.0%}):"
            ]
            for i, face in enumerate(faces, 1):
                confidence = face.det_score if hasattr(face, 'det_score') else 1.0
                age = face.age if hasattr(face, 'age') else "Unknown"
                gender = (
                    "Male" if hasattr(face, 'gender') and face.gender == 1
                    else "Female" if hasattr(face, 'gender')
                    else "Unknown"
                )
                info_lines.append(
                    f"  Face {i}: Confidence {confidence:.1%}, "
                    f"Age ~{age}, Gender: {gender}"
                )
            
            return "\n".join(info_lines)
            
        except Exception as e:
            logger.error("Face detection failed: %s", e)
            return f"❌ Face detection failed: {str(e)}"
    
    def detect_and_extract_faces(self, image_path: str) -> Tuple[List[Image.Image], str]:
        """
        Detect faces and extract face thumbnails for preview.
        Faces are sorted by position (left-to-right, top-to-bottom) for consistency.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (face_images, info_text)
        """
        try:
            img = self._processor.read_image(image_path)
            all_faces = self._processor.get_faces(img)
            
            # Filter by confidence
            faces = filter_faces_by_confidence(all_faces, self.confidence)
            
            if not faces:
                return [], "⚠️ No faces detected"
            
            # Sort faces by position to match processing order
            faces = sort_faces_by_position(faces)
            
            # Extract face thumbnails
            face_images = []
            for face in faces:
                # Get bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Add padding
                padding = 20
                h, w = img.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                # Crop face
                face_crop = img[y1:y2, x1:x2]
                
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_images.append(face_pil)
            
            info_text = f"✅ Detected {len(faces)} face(s) (sorted left→right, top→bottom)"
            return face_images, info_text
            
        except Exception as e:
            logger.error("Face extraction failed: %s", e)
            return [], f"❌ Error: {str(e)}"


class FaceMappingManager:
    """Manages face mapping state for multi-face swapping."""
    
    def __init__(self):
        """Initialize face mapping manager."""
        self._mappings: List[Tuple[int, int]] = []
    
    def add(self, source_idx: int, target_idx: int) -> None:
        """Add a face mapping."""
        self._mappings.append((source_idx, target_idx))
        logger.info("Added face mapping: Source %d → Target %d", source_idx, target_idx)
    
    def clear(self) -> None:
        """Clear all face mappings."""
        self._mappings = []
        logger.info("Cleared all face mappings")
    
    def get(self) -> Optional[List[Tuple[int, int]]]:
        """Get current face mappings, or None if empty."""
        return self._mappings if self._mappings else None
    
    def get_display_text(self) -> str:
        """Get formatted display text for current mappings."""
        if not self._mappings:
            return "No mappings"
        return "\n".join([
            f"Source Face {s} → Target Face {t}"
            for s, t in self._mappings
        ])
    
    def count(self) -> int:
        """Get number of mappings."""
        return len(self._mappings)
