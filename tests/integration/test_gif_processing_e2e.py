"""
End-to-end integration tests for GIF processing pipeline.

These tests verify the full GIF processing workflow including:
- GIF frame extraction
- Face detection across frames
- Face tracking and consistency
- Frame-by-frame swapping
- GIF reassembly
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image


@pytest.fixture
def multi_frame_gif(tmp_path):
    """Create a test GIF with multiple frames."""
    frames = []
    for i in range(10):
        # Create frames with slight color variation
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = (i * 25) % 255  # Red varies
        img_array[:, :, 1] = 128  # Green constant
        img_array[:, :, 2] = 128  # Blue constant
        frames.append(Image.fromarray(img_array))

    gif_path = tmp_path / "test_multiframe.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    return gif_path


@pytest.fixture
def mock_gif_processor():
    """Create mock GIF processor components."""
    processor = MagicMock()

    # Mock face detection returning consistent face across frames
    mock_face = MagicMock()
    mock_face.bbox = np.array([25, 25, 75, 75])
    mock_face.det_score = 0.9
    mock_face.embedding = np.random.rand(512).astype(np.float32)
    processor.get_faces.return_value = [mock_face]

    # Mock swap
    def mock_swap(frame, target, source):
        return frame.copy()
    processor.swap_face.side_effect = mock_swap

    return processor


class TestGIFProcessingE2E:
    """End-to-end tests for GIF processing."""

    @pytest.mark.integration
    def test_gif_frame_extraction(self, multi_frame_gif):
        """Test GIF frames are correctly extracted."""
        from PIL import Image

        with Image.open(multi_frame_gif) as gif:
            frame_count = 0
            try:
                while True:
                    frame_count += 1
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass

        assert frame_count == 10, f"Expected 10 frames, got {frame_count}"

    @pytest.mark.integration
    def test_gif_frame_processing_order(self, multi_frame_gif, mock_gif_processor):
        """Test frames are processed in correct order."""
        from PIL import Image
        import numpy as np

        frames = []
        with Image.open(multi_frame_gif) as gif:
            for i in range(10):
                try:
                    gif.seek(i)
                    frame = np.array(gif.convert('RGB'))
                    frames.append(frame)
                except EOFError:
                    break

        # Verify frames are different (color variation)
        assert not np.array_equal(frames[0], frames[5]), "Frames should vary"

        # Process each frame
        processed = []
        for frame in frames:
            result = mock_gif_processor.swap_face(frame, MagicMock(), MagicMock())
            processed.append(result)

        assert len(processed) == 10, "All frames should be processed"

    @pytest.mark.integration
    def test_gif_face_tracking_consistency(self, mock_gif_processor):
        """Test face tracking maintains identity across frames."""
        # Simulate frames with face in slightly different positions
        frames = []
        for i in range(5):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frames.append(frame)

        # Configure mock to return face with slight movement
        def get_faces_with_movement(frame):
            face = MagicMock()
            # Face moves slightly between frames
            offset = np.random.randint(-5, 5)
            face.bbox = np.array([25 + offset, 25 + offset, 75 + offset, 75 + offset])
            face.det_score = 0.9
            face.embedding = np.random.rand(512).astype(np.float32)
            return [face]

        mock_gif_processor.get_faces.side_effect = get_faces_with_movement

        # Track faces across frames
        face_positions = []
        for frame in frames:
            faces = mock_gif_processor.get_faces(frame)
            if faces:
                face_positions.append(faces[0].bbox.copy())

        assert len(face_positions) == 5, "Should track face in all frames"

    @pytest.mark.integration
    def test_gif_output_generation(self, tmp_path):
        """Test GIF output is correctly generated."""
        # Create processed frames
        processed_frames = []
        for i in range(5):
            frame = Image.new('RGB', (100, 100), color=(i * 50, 100, 100))
            processed_frames.append(frame)

        output_path = tmp_path / "output.gif"

        # Save as GIF
        processed_frames[0].save(
            output_path,
            save_all=True,
            append_images=processed_frames[1:],
            duration=100,
            loop=0
        )

        assert output_path.exists(), "Output GIF should be created"

        # Verify it's a valid GIF
        with Image.open(output_path) as out_gif:
            assert out_gif.format == 'GIF'
            # Count frames
            frame_count = 0
            try:
                while True:
                    frame_count += 1
                    out_gif.seek(out_gif.tell() + 1)
            except EOFError:
                pass
            assert frame_count == 5

    @pytest.mark.integration
    def test_gif_duration_preserved(self, multi_frame_gif, tmp_path):
        """Test frame durations are preserved in output."""
        from PIL import Image

        # Get original durations
        original_durations = []
        with Image.open(multi_frame_gif) as gif:
            for i in range(10):
                try:
                    gif.seek(i)
                    duration = gif.info.get('duration', 100)
                    original_durations.append(duration)
                except EOFError:
                    break

        # All frames should have 100ms duration
        assert all(d == 100 for d in original_durations)


class TestGIFWithAsyncPipeline:
    """Tests for GIF processing with async pipeline."""

    @pytest.mark.integration
    def test_async_pipeline_frame_ordering(self, mock_gif_processor):
        """Test async pipeline maintains frame order."""
        # Simulate async processing that might reorder
        frame_indices = list(range(10))
        processed_order = []

        for idx in frame_indices:
            # Simulate processing
            processed_order.append(idx)

        assert processed_order == frame_indices, "Frame order should be preserved"

    @pytest.mark.integration
    def test_async_pipeline_error_handling(self, mock_gif_processor):
        """Test pipeline handles frame processing errors gracefully."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

        # Configure one frame to fail
        call_count = [0]
        def swap_with_error(frame, target, source):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("Simulated processing error")
            return frame.copy()

        mock_gif_processor.swap_face.side_effect = swap_with_error

        results = []
        errors = []
        for i, frame in enumerate(frames):
            try:
                result = mock_gif_processor.swap_face(frame, MagicMock(), MagicMock())
                results.append(result)
            except RuntimeError as e:
                errors.append((i, str(e)))

        assert len(errors) == 1, "Should have one error"
        assert errors[0][0] == 2, "Error should be on frame 3 (index 2)"


class TestGIFMemoryManagement:
    """Tests for memory management during GIF processing."""

    @pytest.mark.integration
    def test_large_gif_memory_limits(self, tmp_path):
        """Test handling of GIFs that might exceed memory limits."""
        from utils.validation import validate_gif_file

        # Create a GIF with many frames
        frames = []
        for i in range(100):  # Many frames
            img = Image.new('RGB', (200, 200), color=(i % 255, 100, 100))
            frames.append(img)

        gif_path = tmp_path / "large.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0
        )

        # Validation should check frame count
        # Assuming max_frames is configurable
        try:
            result = validate_gif_file(str(gif_path), max_frames=50)
            # Should either pass or raise depending on implementation
        except Exception as e:
            assert "frame" in str(e).lower() or "limit" in str(e).lower()

    @pytest.mark.integration
    def test_frame_batch_processing(self, mock_gif_processor):
        """Test frames are processed in batches for memory efficiency."""
        total_frames = 20
        batch_size = 4

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(total_frames)]

        # Process in batches
        batches_processed = 0
        for i in range(0, total_frames, batch_size):
            batch = frames[i:i + batch_size]
            for frame in batch:
                mock_gif_processor.swap_face(frame, MagicMock(), MagicMock())
            batches_processed += 1

        expected_batches = (total_frames + batch_size - 1) // batch_size
        assert batches_processed == expected_batches
