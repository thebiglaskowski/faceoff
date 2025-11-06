"""
Face detection, extraction, and processing utilities.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
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


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Extract coordinates - convert to float to handle numpy arrays
    x1_min, y1_min, x1_max, y1_max = float(bbox1[0]), float(bbox1[1]), float(bbox1[2]), float(bbox1[3])
    x2_min, y2_min, x2_max, y2_max = float(bbox2[0]), float(bbox2[1]), float(bbox2[2]), float(bbox2[3])
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return float(inter_area / union_area) if union_area > 0 else 0.0


class FaceTracker:
    """
    Tracks faces across frames using bounding box overlap (IoU).
    Assigns stable IDs to faces even when detection order changes.
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize face tracker.
        
        Args:
            iou_threshold: Minimum IoU to consider faces as matching (default 0.3)
        """
        self.previous_faces = []
        self.iou_threshold = iou_threshold
        logger.debug("FaceTracker initialized with IoU threshold %.2f", iou_threshold)
    
    def reset(self):
        """Reset tracker state (call between videos/GIFs)."""
        self.previous_faces = []
    
    def track_faces(self, current_faces):
        """
        Match current faces to previous frame using IoU overlap.
        Returns faces reordered to maintain stable indices.
        
        Args:
            current_faces: List of detected faces in current frame
            
        Returns:
            Reordered list of faces with stable IDs
        """
        if not current_faces:
            self.previous_faces = []
            return []
        
        # First frame: use positional sorting as baseline
        if not self.previous_faces:
            sorted_faces = sort_faces_by_position(current_faces)
            self.previous_faces = sorted_faces
            return sorted_faces
        
        # Match current faces to previous faces using IoU
        matched_faces = [None] * len(self.previous_faces)
        matched_indices = set()  # Track which current faces have been matched
        
        # For each previous face, find best match in current frame
        for prev_idx, prev_face in enumerate(self.previous_faces):
            best_iou = 0.0
            best_match_idx = -1
            
            for curr_idx, curr_face in enumerate(current_faces):
                if curr_idx in matched_indices:
                    continue
                iou = calculate_iou(prev_face.bbox, curr_face.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match_idx = curr_idx
            
            if best_match_idx >= 0:
                matched_faces[prev_idx] = current_faces[best_match_idx]
                matched_indices.add(best_match_idx)
        
        # Keep None as placeholders to maintain stable indices!
        stable_faces = matched_faces
        
        # Append any new faces (didn't match previous frame) sorted by position
        unmatched_current = [f for idx, f in enumerate(current_faces) if idx not in matched_indices]
        if unmatched_current:
            new_faces = sort_faces_by_position(unmatched_current)
            stable_faces.extend(new_faces)
        
        # Update previous faces for next frame
        self.previous_faces = stable_faces
        
        return stable_faces


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
            for idx, face in enumerate(faces):
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
                
                # Add index number overlay
                draw = ImageDraw.Draw(face_pil)
                
                # Calculate font size based on image size
                img_width = face_pil.width
                font_size = max(24, int(img_width * 0.15))  # Scale with image, min 24px
                
                # Try to load a nice font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                
                # Draw text with background for better visibility
                text = str(idx)
                
                # Get text bounding box
                try:
                    bbox_text = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                except:
                    # Fallback for older Pillow versions
                    text_width, text_height = draw.textsize(text, font=font)
                
                # Position in top-left corner with padding
                padding_text = 8
                x = padding_text
                y = padding_text
                
                # Draw semi-transparent background rectangle
                bg_rect = [
                    x - 4,
                    y - 4,
                    x + text_width + 4,
                    y + text_height + 4
                ]
                draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
                
                # Draw white text
                draw.text((x, y), text, fill=(255, 255, 255), font=font)
                
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
