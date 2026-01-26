"""
Unit tests for core/face_processor.py

Tests:
- filter_faces_by_confidence
- sort_faces_by_position
- calculate_iou
- FaceTracker
- FaceMappingManager
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestFilterFacesByConfidence:
    """Tests for filter_faces_by_confidence function."""

    def test_filter_empty_list(self, reset_config):
        """Should return empty list for empty input."""
        from core.face_processor import filter_faces_by_confidence

        result = filter_faces_by_confidence([])
        assert result == []

    def test_filter_all_above_threshold(self, reset_config, mock_faces):
        """Should keep all faces above threshold."""
        from core.face_processor import filter_faces_by_confidence

        # Set all faces to high confidence
        for face in mock_faces:
            face.det_score = 0.9

        result = filter_faces_by_confidence(mock_faces, threshold=0.5)
        assert len(result) == len(mock_faces)

    def test_filter_some_below_threshold(self, reset_config, mock_faces):
        """Should filter out faces below threshold."""
        from core.face_processor import filter_faces_by_confidence

        # Set varying confidence
        mock_faces[0].det_score = 0.9
        mock_faces[1].det_score = 0.4  # Below threshold
        mock_faces[2].det_score = 0.6

        result = filter_faces_by_confidence(mock_faces, threshold=0.5)
        assert len(result) == 2

    def test_filter_all_below_threshold(self, reset_config, mock_faces):
        """Should return empty when all faces below threshold."""
        from core.face_processor import filter_faces_by_confidence

        for face in mock_faces:
            face.det_score = 0.3

        result = filter_faces_by_confidence(mock_faces, threshold=0.5)
        assert len(result) == 0

    def test_filter_uses_config_default(self, temp_config, reset_config):
        """Should use config default when threshold not provided."""
        from core.face_processor import filter_faces_by_confidence

        # Reload config to use temp_config
        from utils.config_manager import Config
        Config._instance = None
        config = Config()

        mock_face = MagicMock()
        mock_face.det_score = 0.4  # Below default 0.5

        result = filter_faces_by_confidence([mock_face])
        assert len(result) == 0

    def test_filter_handles_missing_det_score(self, reset_config):
        """Should treat faces without det_score as 1.0 confidence."""
        from core.face_processor import filter_faces_by_confidence

        mock_face = MagicMock(spec=[])  # No det_score attribute
        result = filter_faces_by_confidence([mock_face], threshold=0.5)
        assert len(result) == 1


class TestSortFacesByPosition:
    """Tests for sort_faces_by_position function."""

    def test_sort_empty_list(self, reset_config):
        """Should return empty list for empty input."""
        from core.face_processor import sort_faces_by_position

        result = sort_faces_by_position([])
        assert result == []

    def test_sort_by_x_position(self, reset_config, mock_faces):
        """Should sort faces left-to-right by x coordinate."""
        from core.face_processor import sort_faces_by_position

        # Set specific positions
        mock_faces[0].bbox = np.array([200, 50, 300, 150])  # Right
        mock_faces[1].bbox = np.array([50, 50, 150, 150])   # Left
        mock_faces[2].bbox = np.array([125, 50, 225, 150])  # Middle

        result = sort_faces_by_position(mock_faces)

        assert result[0].bbox[0] == 50   # Left first
        assert result[1].bbox[0] == 125  # Middle second
        assert result[2].bbox[0] == 200  # Right last

    def test_sort_by_y_when_x_equal(self, reset_config):
        """Should sort by y coordinate when x is equal."""
        from core.face_processor import sort_faces_by_position

        faces = []
        for y in [150, 50, 100]:
            face = MagicMock()
            face.bbox = np.array([100, y, 200, y + 100])
            faces.append(face)

        result = sort_faces_by_position(faces)

        assert result[0].bbox[1] == 50   # Top first
        assert result[1].bbox[1] == 100  # Middle second
        assert result[2].bbox[1] == 150  # Bottom last


class TestCalculateIoU:
    """Tests for calculate_iou function."""

    def test_perfect_overlap(self, reset_config):
        """Should return 1.0 for identical boxes."""
        from core.face_processor import calculate_iou

        bbox = [0, 0, 100, 100]
        result = calculate_iou(bbox, bbox)
        assert result == 1.0

    def test_no_overlap(self, reset_config):
        """Should return 0.0 for non-overlapping boxes."""
        from core.face_processor import calculate_iou

        bbox1 = [0, 0, 50, 50]
        bbox2 = [100, 100, 150, 150]
        result = calculate_iou(bbox1, bbox2)
        assert result == 0.0

    def test_partial_overlap(self, reset_config):
        """Should calculate correct IoU for partial overlap."""
        from core.face_processor import calculate_iou

        bbox1 = [0, 0, 100, 100]      # Area = 10000
        bbox2 = [50, 50, 150, 150]    # Area = 10000
        # Intersection: [50,50] to [100,100] = 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 = 0.1428...

        result = calculate_iou(bbox1, bbox2)
        assert 0.14 < result < 0.15

    def test_one_inside_other(self, reset_config):
        """Should handle one box completely inside another."""
        from core.face_processor import calculate_iou

        bbox1 = [0, 0, 100, 100]     # Area = 10000
        bbox2 = [25, 25, 75, 75]     # Area = 2500, completely inside bbox1
        # Intersection = 2500
        # Union = 10000 (just the larger box since small is inside)
        # IoU = 2500/10000 = 0.25

        result = calculate_iou(bbox1, bbox2)
        assert result == 0.25

    def test_handles_numpy_arrays(self, reset_config):
        """Should work with numpy array inputs."""
        from core.face_processor import calculate_iou

        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([0, 0, 100, 100])
        result = calculate_iou(bbox1, bbox2)
        assert result == 1.0

    def test_zero_area_box(self, reset_config):
        """Should return 0.0 for zero-area boxes."""
        from core.face_processor import calculate_iou

        bbox1 = [0, 0, 0, 0]  # Zero area
        bbox2 = [0, 0, 100, 100]
        result = calculate_iou(bbox1, bbox2)
        assert result == 0.0


class TestFaceTracker:
    """Tests for FaceTracker class."""

    def test_tracker_init(self, reset_config):
        """Should initialize with default IoU threshold."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker()
        assert tracker.iou_threshold == 0.3
        assert tracker.previous_faces == []

    def test_tracker_custom_threshold(self, reset_config):
        """Should accept custom IoU threshold."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker(iou_threshold=0.5)
        assert tracker.iou_threshold == 0.5

    def test_track_empty_faces(self, reset_config):
        """Should return empty list for empty input."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker()
        result = tracker.track_faces([])
        assert result == []
        assert tracker.previous_faces == []

    def test_track_first_frame(self, reset_config, mock_faces):
        """Should sort faces by position on first frame."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker()
        result = tracker.track_faces(mock_faces)

        # Should be sorted by position
        assert len(result) == len(mock_faces)
        assert tracker.previous_faces == result

    def test_track_maintains_ids(self, reset_config):
        """Should maintain face IDs across frames when faces move slightly."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker(iou_threshold=0.3)

        # Frame 1: face at position A
        face1 = MagicMock()
        face1.bbox = np.array([100, 100, 200, 200])

        # Frame 2: face moved slightly (high IoU)
        face2 = MagicMock()
        face2.bbox = np.array([110, 105, 210, 205])

        result1 = tracker.track_faces([face1])
        result2 = tracker.track_faces([face2])

        # Should maintain order
        assert len(result1) == 1
        assert len(result2) == 1

    def test_track_handles_new_face(self, reset_config):
        """Should add new faces when they appear."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker()

        # Frame 1: one face
        face1 = MagicMock()
        face1.bbox = np.array([50, 50, 150, 150])

        # Frame 2: two faces (new face appears)
        face2a = MagicMock()
        face2a.bbox = np.array([55, 55, 155, 155])  # Slightly moved face1
        face2b = MagicMock()
        face2b.bbox = np.array([300, 50, 400, 150])  # New face

        result1 = tracker.track_faces([face1])
        result2 = tracker.track_faces([face2a, face2b])

        assert len(result1) == 1
        assert len(result2) >= 2  # Could be 2 or more with placeholders

    def test_reset_clears_state(self, reset_config):
        """Should clear previous faces on reset."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker()

        face = MagicMock()
        face.bbox = np.array([50, 50, 150, 150])
        tracker.track_faces([face])

        assert len(tracker.previous_faces) > 0

        tracker.reset()
        assert tracker.previous_faces == []


class TestFaceMappingManager:
    """Tests for FaceMappingManager class."""

    def test_init_empty(self, reset_config):
        """Should initialize with empty mappings."""
        from core.face_processor import FaceMappingManager

        manager = FaceMappingManager()
        assert manager.get() is None
        assert manager.count() == 0

    def test_add_mapping(self, reset_config):
        """Should add face mappings correctly."""
        from core.face_processor import FaceMappingManager

        manager = FaceMappingManager()
        manager.add(0, 1)

        mappings = manager.get()
        assert mappings == [(0, 1)]
        assert manager.count() == 1

    def test_add_multiple_mappings(self, reset_config):
        """Should support multiple mappings."""
        from core.face_processor import FaceMappingManager

        manager = FaceMappingManager()
        manager.add(0, 0)
        manager.add(1, 1)
        manager.add(0, 2)

        mappings = manager.get()
        assert len(mappings) == 3
        assert manager.count() == 3

    def test_clear_mappings(self, reset_config):
        """Should clear all mappings."""
        from core.face_processor import FaceMappingManager

        manager = FaceMappingManager()
        manager.add(0, 1)
        manager.add(1, 0)

        manager.clear()

        assert manager.get() is None
        assert manager.count() == 0

    def test_get_display_text_empty(self, reset_config):
        """Should return 'No mappings' when empty."""
        from core.face_processor import FaceMappingManager

        manager = FaceMappingManager()
        assert manager.get_display_text() == "No mappings"

    def test_get_display_text_with_mappings(self, reset_config):
        """Should format mappings for display."""
        from core.face_processor import FaceMappingManager

        manager = FaceMappingManager()
        manager.add(0, 1)
        manager.add(1, 0)

        text = manager.get_display_text()
        assert "Source Face 0" in text
        assert "Target Face 1" in text
        assert "Source Face 1" in text
        assert "Target Face 0" in text


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_iou_with_float_coordinates(self, reset_config):
        """Should handle floating point coordinates."""
        from core.face_processor import calculate_iou

        bbox1 = [10.5, 20.3, 50.7, 80.9]
        bbox2 = [10.5, 20.3, 50.7, 80.9]
        result = calculate_iou(bbox1, bbox2)
        assert result == 1.0

    def test_tracker_with_none_in_previous(self, reset_config):
        """Tracker should handle None values in previous faces."""
        from core.face_processor import FaceTracker

        tracker = FaceTracker()

        # Manually set previous faces with None
        mock_face = MagicMock()
        mock_face.bbox = np.array([50, 50, 150, 150])
        tracker.previous_faces = [None, mock_face]

        new_face = MagicMock()
        new_face.bbox = np.array([55, 55, 155, 155])

        # Should not raise
        result = tracker.track_faces([new_face])
        assert len(result) >= 1
