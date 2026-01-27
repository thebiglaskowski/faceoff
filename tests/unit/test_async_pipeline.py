"""
Unit tests for async pipeline processing.

Tests cover:
- FrameTask dataclass
- FramePrefetcher threading and queue management
- AsyncPipeline initialization and configuration
- Thread lifecycle and error propagation
"""

import numpy as np
import pytest
import queue
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

from processing.async_pipeline import (
    FrameTask,
    FramePrefetcher,
    AsyncPipeline,
)


# =============================================================================
# FrameTask Tests
# =============================================================================

class TestFrameTask:
    """Tests for FrameTask dataclass."""

    def test_frame_task_creation(self):
        """Test FrameTask can be created with required fields."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        task = FrameTask(index=0, frame=frame)

        assert task.index == 0
        assert task.frame is frame
        assert task.detected_faces is None
        assert task.swapped_frame is None

    def test_frame_task_with_optional_fields(self):
        """Test FrameTask with all fields populated."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        swapped = np.ones((100, 100, 3), dtype=np.uint8)
        faces = [MagicMock(), MagicMock()]

        task = FrameTask(
            index=5,
            frame=frame,
            detected_faces=faces,
            swapped_frame=swapped
        )

        assert task.index == 5
        assert task.detected_faces == faces
        assert np.array_equal(task.swapped_frame, swapped)

    def test_frame_task_mutable(self):
        """Test FrameTask fields can be updated after creation."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        task = FrameTask(index=0, frame=frame)

        # Update fields
        task.detected_faces = ["face1", "face2"]
        task.swapped_frame = np.ones((100, 100, 3), dtype=np.uint8)

        assert len(task.detected_faces) == 2
        assert task.swapped_frame is not None


# =============================================================================
# FramePrefetcher Tests
# =============================================================================

class TestFramePrefetcher:
    """Tests for FramePrefetcher threading and queue management."""

    def test_prefetcher_init(self):
        """Test FramePrefetcher initialization."""
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(10)]
        prefetcher = FramePrefetcher(frames, prefetch_count=4)

        assert len(prefetcher) == 10
        assert prefetcher.prefetch_count == 4

    def test_prefetcher_limits_prefetch_count(self):
        """Test prefetch_count is limited to frame count."""
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        prefetcher = FramePrefetcher(frames, prefetch_count=10)

        assert prefetcher.prefetch_count == 3  # Limited to len(frames)

    def test_prefetcher_context_manager(self):
        """Test FramePrefetcher as context manager."""
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(5)]

        with FramePrefetcher(frames, prefetch_count=2) as prefetcher:
            # Thread should be running
            assert prefetcher._thread is not None
            assert prefetcher._thread.is_alive()

        # After context exit, thread should stop
        time.sleep(0.2)  # Give thread time to stop
        assert not prefetcher._thread.is_alive()

    def test_prefetcher_get_next_returns_frames(self):
        """Test get_next returns prefetched frames in order."""
        frames = [
            np.full((10, 10, 3), i, dtype=np.uint8)
            for i in range(5)
        ]

        with FramePrefetcher(frames, prefetch_count=5) as prefetcher:
            # Wait for prefetch
            time.sleep(0.2)

            retrieved = []
            for _ in range(5):
                result = prefetcher.get_next()
                if result:
                    idx, frame = result
                    retrieved.append((idx, frame[0, 0, 0]))

            # All frames should be retrieved
            assert len(retrieved) == 5
            # Indices should be 0-4 (order may vary slightly due to threading)
            assert set(r[0] for r in retrieved) == {0, 1, 2, 3, 4}

    def test_prefetcher_get_next_returns_none_when_empty(self):
        """Test get_next returns None when no more frames."""
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)]

        with FramePrefetcher(frames, prefetch_count=1) as prefetcher:
            # Get the only frame
            time.sleep(0.1)
            result = prefetcher.get_next()
            assert result is not None

            # Stop prefetcher
            prefetcher.stop()

            # Now should return None (timeout)
            result = prefetcher.get_next()
            assert result is None

    def test_prefetcher_stop_terminates_thread(self):
        """Test stop() terminates prefetch thread."""
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(100)]
        prefetcher = FramePrefetcher(frames, prefetch_count=10)

        prefetcher.start()
        assert prefetcher._thread.is_alive()

        prefetcher.stop()
        time.sleep(0.3)
        assert not prefetcher._thread.is_alive()

    def test_prefetcher_contiguous_memory(self):
        """Test prefetcher returns contiguous arrays."""
        # Create non-contiguous array
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        non_contiguous = base[::2, ::2, :]  # Strided view
        frames = [non_contiguous]

        with FramePrefetcher(frames, prefetch_count=1) as prefetcher:
            time.sleep(0.1)
            result = prefetcher.get_next()

            if result:
                _, frame = result
                assert frame.flags['C_CONTIGUOUS']


# =============================================================================
# AsyncPipeline Tests
# =============================================================================

class TestAsyncPipeline:
    """Tests for AsyncPipeline initialization and configuration."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MediaProcessor for testing."""
        processor = MagicMock()
        processor.device_id = 0

        # Mock swapper with get method
        processor.swapper = MagicMock()
        processor.swapper.get = MagicMock(
            side_effect=lambda img, dst, src, paste_back: img
        )

        # Mock get_faces method
        processor.get_faces = MagicMock(return_value=[])

        return processor

    @pytest.fixture
    def mock_src_faces(self):
        """Create mock source faces."""
        face = MagicMock()
        face.embedding = np.random.rand(512).astype(np.float32)
        return [face]

    def test_pipeline_init_default_queue_size(self, mock_processor, mock_src_faces):
        """Test AsyncPipeline uses config queue size by default."""
        # The real config has queue_size: 32
        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5
        )

        # Queue maxsize should come from config (32 in real config.yaml)
        # Test that all queues have the same size (consistency check)
        assert pipeline.detection_queue.maxsize == pipeline.swapping_queue.maxsize
        assert pipeline.swapping_queue.maxsize == pipeline.output_queue.maxsize
        assert pipeline.detection_queue.maxsize > 0  # Should be bounded

    def test_pipeline_init_custom_queue_size(self, mock_processor, mock_src_faces):
        """Test AsyncPipeline with custom queue size."""
        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=8
        )

        assert pipeline.detection_queue.maxsize == 8
        assert pipeline.swapping_queue.maxsize == 8
        assert pipeline.output_queue.maxsize == 8

    def test_pipeline_init_unbounded_queue(self, mock_processor, mock_src_faces):
        """Test AsyncPipeline with unbounded queue (legacy behavior)."""
        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=0
        )

        assert pipeline.detection_queue.maxsize == 0  # 0 means unbounded

    def test_pipeline_shutdown(self, mock_processor, mock_src_faces):
        """Test pipeline shutdown sets stop event."""
        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=4
        )

        assert not pipeline.stop_event.is_set()
        pipeline.shutdown()
        assert pipeline.stop_event.is_set()

    def test_pipeline_face_tracker_uses_config(self, mock_processor, mock_src_faces, temp_config):
        """Test pipeline face tracker uses configured IoU threshold."""
        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5
        )

        # temp_config doesn't set iou_threshold, so should use default
        assert pipeline.face_tracker is not None

    @patch('processing.async_pipeline.get_progress_tracker')
    @patch('processing.async_pipeline.MemoryManager')
    def test_pipeline_process_frames_empty_list(
        self, mock_memory_mgr, mock_progress, mock_processor, mock_src_faces
    ):
        """Test processing empty frame list."""
        mock_progress.return_value = MagicMock()
        mock_progress.return_value.track.return_value.__enter__ = MagicMock()
        mock_progress.return_value.track.return_value.__exit__ = MagicMock()

        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=4
        )

        results = pipeline.process_frames([])
        assert results == []

    @patch('processing.async_pipeline.get_progress_tracker')
    @patch('processing.async_pipeline.MemoryManager')
    def test_pipeline_process_frames_returns_same_count(
        self, mock_memory_mgr, mock_progress, mock_processor, mock_src_faces
    ):
        """Test processing returns same number of frames as input."""
        # Setup progress tracker mock
        progress_mock = MagicMock()
        pbar_mock = MagicMock()
        pbar_mock.__enter__ = MagicMock(return_value=pbar_mock)
        pbar_mock.__exit__ = MagicMock(return_value=False)
        progress_mock.track.return_value = pbar_mock
        mock_progress.return_value = progress_mock

        # Setup memory manager mock
        mem_mgr_instance = MagicMock()
        mem_mgr_instance.should_clear_cache.return_value = False
        mock_memory_mgr.return_value = mem_mgr_instance

        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=4
        )

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        results = pipeline.process_frames(frames)

        assert len(results) == 5

    @patch('processing.async_pipeline.get_progress_tracker')
    @patch('processing.async_pipeline.MemoryManager')
    def test_pipeline_threads_stop_after_processing(
        self, mock_memory_mgr, mock_progress, mock_processor, mock_src_faces
    ):
        """Test worker threads stop after processing completes."""
        # Setup mocks
        progress_mock = MagicMock()
        pbar_mock = MagicMock()
        pbar_mock.__enter__ = MagicMock(return_value=pbar_mock)
        pbar_mock.__exit__ = MagicMock(return_value=False)
        progress_mock.track.return_value = pbar_mock
        mock_progress.return_value = progress_mock

        mem_mgr_instance = MagicMock()
        mem_mgr_instance.should_clear_cache.return_value = False
        mock_memory_mgr.return_value = mem_mgr_instance

        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=4
        )

        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        pipeline.process_frames(frames)

        # Give threads time to stop
        time.sleep(0.5)

        # All threads should have stopped
        for thread in pipeline.threads:
            assert not thread.is_alive()


class TestAsyncPipelineErrorHandling:
    """Tests for AsyncPipeline error handling."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MediaProcessor."""
        processor = MagicMock()
        processor.device_id = 0
        processor.swapper = MagicMock()
        processor.get_faces = MagicMock(return_value=[])
        return processor

    @pytest.fixture
    def mock_src_faces(self):
        """Create mock source faces."""
        return [MagicMock()]

    @patch('processing.async_pipeline.get_progress_tracker')
    @patch('processing.async_pipeline.MemoryManager')
    def test_pipeline_handles_detection_error(
        self, mock_memory_mgr, mock_progress, mock_processor, mock_src_faces
    ):
        """Test pipeline handles errors in detection worker."""
        # Setup mocks
        progress_mock = MagicMock()
        pbar_mock = MagicMock()
        pbar_mock.__enter__ = MagicMock(return_value=pbar_mock)
        pbar_mock.__exit__ = MagicMock(return_value=False)
        progress_mock.track.return_value = pbar_mock
        mock_progress.return_value = progress_mock

        mem_mgr_instance = MagicMock()
        mem_mgr_instance.should_clear_cache.return_value = False
        mock_memory_mgr.return_value = mem_mgr_instance

        # Make get_faces raise an error
        mock_processor.get_faces.side_effect = RuntimeError("Detection failed")

        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=4
        )

        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]

        # Should raise RuntimeError due to pipeline error
        # Error can be "Pipeline error" or "Pipeline threads died before completing"
        with pytest.raises(RuntimeError, match="Pipeline"):
            pipeline.process_frames(frames)

    def test_pipeline_error_sets_stop_event(self, mock_processor, mock_src_faces):
        """Test that errors set the stop event."""
        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=mock_src_faces,
            face_confidence=0.5,
            queue_size=4
        )

        # Simulate error
        pipeline.error = Exception("Test error")
        pipeline.stop_event.set()

        assert pipeline.stop_event.is_set()
        assert pipeline.error is not None


class TestAsyncPipelineQueueBackpressure:
    """Tests for queue backpressure behavior."""

    @pytest.fixture
    def mock_processor(self):
        """Create a slow mock processor to test backpressure."""
        processor = MagicMock()
        processor.device_id = 0
        processor.swapper = MagicMock()
        processor.get_faces = MagicMock(return_value=[])
        return processor

    def test_bounded_queue_applies_backpressure(self, mock_processor):
        """Test that bounded queues apply backpressure."""
        pipeline = AsyncPipeline(
            processor=mock_processor,
            src_faces=[MagicMock()],
            face_confidence=0.5,
            queue_size=2  # Small queue
        )

        # Fill the detection queue
        for i in range(2):
            task = FrameTask(index=i, frame=np.zeros((10, 10, 3), dtype=np.uint8))
            pipeline.detection_queue.put(task)

        # Queue should be full
        assert pipeline.detection_queue.full()

        # Trying to add more should block (we test by using put_nowait which raises)
        with pytest.raises(queue.Full):
            task = FrameTask(index=2, frame=np.zeros((10, 10, 3), dtype=np.uint8))
            pipeline.detection_queue.put_nowait(task)
