"""
Face detection, extraction, and processing utilities.
"""
import logging
import threading
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from utils.config_manager import config

logger = logging.getLogger("FaceOff")


def filter_faces_by_confidence(faces, threshold: float = None):
    """
    Filter faces by detection confidence threshold.
    
    Args:
        faces: List of detected faces
        threshold: Minimum confidence threshold (0.0-1.0), uses config default if None
        
    Returns:
        Filtered list of faces
    """
    if threshold is None:
        threshold = config.face_confidence_threshold
    
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


def normalized_face_embedding(face) -> Optional[np.ndarray]:
    """Return L2-normalized recognition embedding for a detected face, if present."""
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None
    return vec / norm


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Cosine similarity for unit-normalized embeddings."""
    return float(np.dot(embedding_a, embedding_b))


class FaceTracker:
    """
    Tracks faces across frames using IoU within a shot and recognition
    embeddings (buffalo_l) when bbox overlap fails at scene cuts.
    """

    def __init__(
        self,
        iou_threshold: Optional[float] = None,
        embedding_threshold: Optional[float] = None,
        embedding_enabled: Optional[bool] = None,
        embedding_ema: Optional[float] = None,
    ):
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else config.iou_threshold
        )
        self.embedding_threshold = (
            embedding_threshold
            if embedding_threshold is not None
            else config.embedding_similarity_threshold
        )
        self.embedding_enabled = (
            embedding_enabled
            if embedding_enabled is not None
            else config.embedding_tracking_enabled
        )
        self.embedding_ema = (
            embedding_ema if embedding_ema is not None else config.embedding_ema_alpha
        )
        self.previous_faces: List = []
        self._slot_embeddings: List[Optional[np.ndarray]] = []
        logger.debug(
            "FaceTracker initialized (IoU=%.2f, embedding=%s, sim=%.2f)",
            self.iou_threshold,
            self.embedding_enabled,
            self.embedding_threshold,
        )

    def reset(self):
        """Reset tracker state (call between videos/GIFs)."""
        self.previous_faces = []
        self._slot_embeddings = []

    def _ensure_slot_count(self, count: int) -> None:
        while len(self._slot_embeddings) < count:
            self._slot_embeddings.append(None)

    def _update_slot_embedding(self, slot_idx: int, face) -> None:
        emb = normalized_face_embedding(face)
        if emb is None:
            return
        self._ensure_slot_count(slot_idx + 1)
        prev = self._slot_embeddings[slot_idx]
        if prev is None:
            self._slot_embeddings[slot_idx] = emb
            return
        alpha = self.embedding_ema
        merged = (1.0 - alpha) * prev + alpha * emb
        norm = float(np.linalg.norm(merged))
        self._slot_embeddings[slot_idx] = merged / norm if norm > 0 else merged

    def _match_by_iou(
        self,
        current_faces: List,
        matched_faces: List,
        matched_indices: set,
    ) -> None:
        for prev_idx, prev_face in enumerate(self.previous_faces):
            if prev_face is None:
                continue
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

    def _match_by_embedding(
        self,
        current_faces: List,
        matched_faces: List,
        matched_indices: set,
    ) -> None:
        for prev_idx in range(len(self.previous_faces)):
            if matched_faces[prev_idx] is not None:
                continue
            if prev_idx >= len(self._slot_embeddings):
                continue
            anchor = self._slot_embeddings[prev_idx]
            if anchor is None:
                continue
            best_sim = 0.0
            best_match_idx = -1
            for curr_idx, curr_face in enumerate(current_faces):
                if curr_idx in matched_indices:
                    continue
                curr_emb = normalized_face_embedding(curr_face)
                if curr_emb is None:
                    continue
                sim = cosine_similarity(anchor, curr_emb)
                if sim > best_sim and sim >= self.embedding_threshold:
                    best_sim = sim
                    best_match_idx = curr_idx
            if best_match_idx >= 0:
                matched_faces[prev_idx] = current_faces[best_match_idx]
                matched_indices.add(best_match_idx)

    def track_faces(self, current_faces):
        """
        Match current faces to previous frame slots.
        IoU handles motion within a shot; embeddings re-identify after cuts.
        """
        if not current_faces:
            self.previous_faces = []
            return []

        if not self.previous_faces:
            sorted_faces = sort_faces_by_position(current_faces)
            self.previous_faces = sorted_faces
            self._slot_embeddings = [
                normalized_face_embedding(face) for face in sorted_faces
            ]
            return sorted_faces

        matched_faces: List = [None] * len(self.previous_faces)
        matched_indices: set = set()

        self._match_by_iou(current_faces, matched_faces, matched_indices)
        if self.embedding_enabled:
            self._match_by_embedding(current_faces, matched_faces, matched_indices)

        stable_faces = matched_faces

        unmatched_current = [
            face for idx, face in enumerate(current_faces) if idx not in matched_indices
        ]
        if unmatched_current:
            new_faces = sort_faces_by_position(unmatched_current)
            stable_faces.extend(new_faces)
            for face in new_faces:
                self._slot_embeddings.append(normalized_face_embedding(face))

        for slot_idx, face in enumerate(matched_faces):
            if face is not None:
                self._update_slot_embedding(slot_idx, face)

        self.previous_faces = stable_faces
        return stable_faces


class FaceProcessor:
    """Manages face detection and extraction operations."""
    
    def __init__(self, device_id: int = 0, confidence: float = None):
        """
        Initialize face processor.
        
        Args:
            device_id: GPU device ID
            confidence: Minimum face detection confidence, uses config default if None
        """
        from core.media_processor import MediaProcessor
        
        self.device_id = device_id
        self.confidence = confidence if confidence is not None else config.face_confidence_threshold
        # CRITICAL: Disable ONNX optimization - it creates corrupted model files
        self._processor = MediaProcessor(
            device_id=device_id,
            use_tensorrt=False,
            optimize_models=False,
        )
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
    """
    Thread-safe manager for face mapping state.

    Manages face mapping state for multi-face swapping with proper
    synchronization for concurrent access.
    """

    def __init__(self):
        """Initialize face mapping manager with thread lock."""
        self._mappings: List[Tuple[int, int]] = []
        self._lock = threading.Lock()

    def add(self, source_idx: int, target_idx: int) -> None:
        """Add a face mapping (thread-safe)."""
        with self._lock:
            self._mappings.append((source_idx, target_idx))
        logger.info("Added face mapping: Source %d → Target %d", source_idx, target_idx)

    def clear(self) -> None:
        """Clear all face mappings (thread-safe)."""
        with self._lock:
            self._mappings = []
        logger.info("Cleared all face mappings")

    def get(self) -> Optional[List[Tuple[int, int]]]:
        """Get current face mappings, or None if empty (thread-safe)."""
        with self._lock:
            return self._mappings.copy() if self._mappings else None

    def get_display_text(self) -> str:
        """Get formatted display text for current mappings (thread-safe)."""
        with self._lock:
            if not self._mappings:
                return "No mappings"
            return "\n".join([
                f"Source Face {s} → Target Face {t}"
                for s, t in self._mappings
            ])

    def count(self) -> int:
        """Get number of mappings (thread-safe)."""
        with self._lock:
            return len(self._mappings)
