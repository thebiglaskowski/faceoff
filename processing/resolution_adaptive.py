"""
Resolution-adaptive processing for optimized face detection and swapping.

Detects faces at lower resolution for speed, then processes at native resolution
for quality. This provides significant speedups for high-resolution content.
"""
import cv2
import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger("FaceOff")


class ResolutionAdaptiveProcessor:
    """
    Handles resolution-adaptive face detection and processing.
    
    Detects faces at lower resolution for speed, then maps coordinates
    back to native resolution for high-quality swapping.
    """
    
    def __init__(self, detection_scale: float = 0.5, min_resolution: int = 640):
        """
        Initialize resolution-adaptive processor.
        
        Args:
            detection_scale: Scale factor for face detection (0.25-1.0)
                           0.5 = detect at half resolution (2x faster)
                           0.25 = detect at quarter resolution (4x faster)
            min_resolution: Minimum resolution for detection (don't scale below this)
        """
        self.detection_scale = np.clip(detection_scale, 0.25, 1.0)
        self.min_resolution = min_resolution
        logger.info("Resolution-adaptive processing enabled (scale=%.2f)", self.detection_scale)
    
    def should_downscale(self, frame_shape: Tuple[int, int]) -> bool:
        """
        Determine if frame should be downscaled for detection.
        
        Args:
            frame_shape: (height, width) of frame
            
        Returns:
            True if frame is large enough to benefit from downscaling
        """
        height, width = frame_shape[:2]
        min_dim = min(height, width)
        
        # Only downscale if resulting resolution is above minimum
        scaled_min_dim = min_dim * self.detection_scale
        return scaled_min_dim >= self.min_resolution
    
    def downscale_for_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Downscale frame for faster face detection.
        
        Args:
            frame: Original frame
            
        Returns:
            Tuple of (downscaled_frame, actual_scale_factor)
        """
        if not self.should_downscale(frame.shape):
            return frame, 1.0
        
        height, width = frame.shape[:2]
        new_width = int(width * self.detection_scale)
        new_height = int(height * self.detection_scale)
        
        # Ensure we don't go below minimum resolution
        min_dim = min(new_width, new_height)
        if min_dim < self.min_resolution:
            scale_factor = self.min_resolution / min_dim
            new_width = int(new_width * scale_factor)
            new_height = int(new_height * scale_factor)
            actual_scale = (new_width / width)
        else:
            actual_scale = self.detection_scale
        
        downscaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        logger.debug(
            "Downscaled for detection: %dx%d -> %dx%d (%.2fx)",
            width, height, new_width, new_height, actual_scale
        )
        
        return downscaled, actual_scale
    
    def scale_face_coordinates(self, faces: List, scale_factor: float) -> List:
        """
        Scale face bounding box coordinates back to original resolution.
        
        Args:
            faces: List of detected faces from downscaled frame
            scale_factor: Scale factor used for downscaling
            
        Returns:
            List of faces with coordinates scaled to original resolution
        """
        if scale_factor == 1.0:
            return faces
        
        inverse_scale = 1.0 / scale_factor
        
        for face in faces:
            # Scale bounding box
            if hasattr(face, 'bbox'):
                face.bbox = face.bbox * inverse_scale
            
            # Scale keypoints (landmarks)
            if hasattr(face, 'kps'):
                face.kps = face.kps * inverse_scale
            
            # Scale landmark points
            if hasattr(face, 'landmark_2d_106'):
                face.landmark_2d_106 = face.landmark_2d_106 * inverse_scale
            
            if hasattr(face, 'landmark_3d_68'):
                face.landmark_3d_68 = face.landmark_3d_68 * inverse_scale
        
        return faces
    
    def detect_faces_adaptive(
        self,
        processor,
        frame: np.ndarray
    ) -> List:
        """
        Detect faces using resolution-adaptive approach.
        
        Args:
            processor: MediaProcessor instance with face detection
            frame: Original resolution frame
            
        Returns:
            List of faces with coordinates at original resolution
        """
        # Downscale for detection
        downscaled, scale_factor = self.downscale_for_detection(frame)
        
        # Detect faces at lower resolution
        faces = processor.get_faces(downscaled)
        
        # Scale coordinates back to original resolution
        faces = self.scale_face_coordinates(faces, scale_factor)
        
        return faces
    
    def extract_face_region(
        self,
        frame: np.ndarray,
        face,
        padding: float = 0.3
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract face region from frame with padding.
        
        Args:
            frame: Original frame
            face: Face object with bbox
            padding: Padding around face (0.3 = 30% larger region)
            
        Returns:
            Tuple of (face_region, (x1, y1, x2, y2))
        """
        height, width = frame.shape[:2]
        
        # Get bounding box
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Add padding
        face_width = x2 - x1
        face_height = y2 - y1
        pad_w = int(face_width * padding)
        pad_h = int(face_height * padding)
        
        # Expand with padding, clamped to frame boundaries
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)
        
        # Extract region
        face_region = frame[y1:y2, x1:x2].copy()
        
        return face_region, (x1, y1, x2, y2)
    
    def paste_face_region(
        self,
        frame: np.ndarray,
        face_region: np.ndarray,
        coords: Tuple[int, int, int, int],
        blend_edges: bool = True
    ) -> np.ndarray:
        """
        Paste processed face region back into frame.
        
        Args:
            frame: Original frame
            face_region: Processed face region
            coords: (x1, y1, x2, y2) coordinates where to paste
            blend_edges: Apply edge blending for seamless integration
            
        Returns:
            Frame with pasted face region
        """
        x1, y1, x2, y2 = coords
        region_height, region_width = face_region.shape[:2]
        
        # Ensure sizes match
        if region_height != (y2 - y1) or region_width != (x2 - x1):
            face_region = cv2.resize(
                face_region,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LINEAR
            )
        
        result = frame.copy()
        
        if blend_edges:
            # Create feathered mask for smooth blending
            mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
            feather = min(20, min(y2 - y1, x2 - x1) // 10)
            
            if feather > 0:
                # Apply Gaussian blur to create soft edges
                mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
                mask = np.expand_dims(mask, axis=2)
                
                # Blend using mask
                result[y1:y2, x1:x2] = (
                    face_region * mask + 
                    result[y1:y2, x1:x2] * (1 - mask)
                ).astype(np.uint8)
            else:
                result[y1:y2, x1:x2] = face_region
        else:
            result[y1:y2, x1:x2] = face_region
        
        return result


def detect_and_swap_adaptive(
    processor,
    adaptive_processor: ResolutionAdaptiveProcessor,
    frame: np.ndarray,
    src_faces: List,
    face_confidence: float,
    face_mappings: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """
    Perform face detection and swapping using resolution-adaptive approach.
    
    Args:
        processor: MediaProcessor instance
        adaptive_processor: ResolutionAdaptiveProcessor instance
        frame: Original frame
        src_faces: Source faces to swap in
        face_confidence: Minimum detection confidence
        face_mappings: Optional face mapping rules
        
    Returns:
        Frame with swapped faces
    """
    from core.face_processor import filter_faces_by_confidence
    
    # Detect faces at lower resolution
    dst_faces = adaptive_processor.detect_faces_adaptive(processor, frame)
    dst_faces = filter_faces_by_confidence(dst_faces, face_confidence)
    
    if not dst_faces:
        return frame
    
    result = frame.copy()
    
    # Process each face
    if face_mappings:
        for src_idx, dst_idx in face_mappings:
            if src_idx < len(src_faces) and dst_idx < len(dst_faces):
                face = dst_faces[dst_idx]
                # Swap directly on full frame (insightface handles region extraction)
                result = processor.swapper.get(result, face, src_faces[src_idx], paste_back=True)
    else:
        # Default: swap first source to all destinations
        for face in dst_faces:
            if src_faces:
                result = processor.swapper.get(result, face, src_faces[0], paste_back=True)
    
    return result
