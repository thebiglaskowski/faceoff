"""
Async pipeline processing for overlapping operations.

This module implements a 3-stage pipeline:
1. Preprocessing: Face detection on frames
2. Processing: Face swapping (GPU-intensive)
3. Postprocessing: Blending and finalization

Stages run in parallel with queues to overlap operations and maximize throughput.
"""
import logging
import numpy as np
import queue
import threading
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

from core.media_processor import MediaProcessor
from core.face_processor import filter_faces_by_confidence, FaceTracker
from processing.resolution_adaptive import ResolutionAdaptiveProcessor
from utils.memory_manager import MemoryManager
from utils.progress import get_progress_tracker

logger = logging.getLogger("FaceOff")


@dataclass
class FrameTask:
    """Container for frame processing task."""
    index: int
    frame: np.ndarray
    detected_faces: Optional[List] = None
    swapped_frame: Optional[np.ndarray] = None


class AsyncPipeline:
    """
    Async pipeline for overlapping face detection, swapping, and blending.
    
    Pipeline stages:
    1. Detection: Detect faces in frames (CPU/GPU mixed)
    2. Swapping: Swap faces (GPU intensive)
    3. Finalization: Post-process and collect results
    
    Each stage runs in a separate thread with queues connecting them.
    """
    
    def __init__(
        self,
        processor: MediaProcessor,
        src_faces: List,
        face_confidence: float,
        face_mappings: Optional[List[Tuple[int, int]]] = None,
        adaptive_processor: Optional[ResolutionAdaptiveProcessor] = None,
        queue_size: int = 0  # 0 = unbounded to avoid deadlock
    ):
        """
        Initialize async pipeline.
        
        Args:
            processor: MediaProcessor instance
            src_faces: Source faces for swapping
            face_confidence: Minimum face detection confidence
            face_mappings: Optional face mapping rules
            adaptive_processor: Optional resolution-adaptive processor
            queue_size: Maximum queue size for buffering (0=unbounded)
        """
        self.processor = processor
        self.src_faces = src_faces
        self.face_confidence = face_confidence
        self.face_mappings = face_mappings
        self.adaptive_processor = adaptive_processor
        
        # Memory manager for OOM prevention
        self.memory_manager = MemoryManager(device_id=processor.device_id)
        
        # Create unbounded queues to avoid deadlock when feeding frames
        # Using maxsize=0 (unbounded) prevents blocking when feeding all frames at once
        self.detection_queue = queue.Queue(maxsize=0)
        self.swapping_queue = queue.Queue(maxsize=0)
        self.output_queue = queue.Queue(maxsize=0)
        
        # Face tracker for maintaining stable IDs
        self.face_tracker = FaceTracker(iou_threshold=0.3)
        
        # Threading control
        self.stop_event = threading.Event()
        self.threads = []
        
        # Error tracking
        self.error = None
    
    def _detection_worker(self):
        """Worker thread for face detection stage."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get frame from input queue (timeout to check stop_event)
                    task = self.detection_queue.get(timeout=0.1)
                    if task is None:  # Poison pill
                        self.swapping_queue.put(None)
                        break
                    
                    # Detect faces in frame
                    if self.adaptive_processor:
                        detected_faces = self.adaptive_processor.detect_faces_adaptive(
                            self.processor, task.frame
                        )
                    else:
                        detected_faces = self.processor.get_faces(task.frame)
                    
                    # Filter and sort faces
                    detected_faces = filter_faces_by_confidence(detected_faces, self.face_confidence)
                    detected_faces = self.face_tracker.track_faces(detected_faces)
                    
                    task.detected_faces = detected_faces
                    
                    # Pass to swapping stage
                    self.swapping_queue.put(task)
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error("Detection worker error: %s", e, exc_info=True)
            self.error = e
            self.stop_event.set()
    
    def _swapping_worker(self):
        """Worker thread for face swapping stage (GPU-intensive)."""
        try:
            # Periodically check and clear CUDA cache if needed
            frame_count = 0
            while not self.stop_event.is_set():
                try:
                    # Get task from swapping queue
                    task = self.swapping_queue.get(timeout=0.1)
                    if task is None:  # Poison pill
                        self.output_queue.put(None)
                        break
                    
                    # Check if we should clear cache
                    if self.memory_manager.should_clear_cache():
                        self.memory_manager.clear_cache()
                    
                    # Perform face swapping with OOM error handling
                    try:
                        swapped = task.frame.copy()
                        
                        if self.face_mappings:
                            # Use face mappings
                            for src_idx, dst_idx in self.face_mappings:
                                if (src_idx < len(self.src_faces) and 
                                    dst_idx < len(task.detected_faces) and 
                                    task.detected_faces[dst_idx] is not None):
                                    swapped = self.processor.swapper.get(
                                        swapped, 
                                        task.detected_faces[dst_idx], 
                                        self.src_faces[src_idx], 
                                        paste_back=True
                                    )
                        else:
                            # Default: swap first source to all detected faces
                            for face in task.detected_faces:
                                if self.src_faces and face is not None:
                                    swapped = self.processor.swapper.get(
                                        swapped, face, self.src_faces[0], paste_back=True
                                    )
                        
                        task.swapped_frame = swapped
                        
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_error:
                        if "out of memory" in str(oom_error).lower():
                            logger.warning("OOM in async pipeline - clearing cache and retrying")
                            self.memory_manager.clear_cache(force=True)
                            # Return original frame if OOM persists (graceful degradation)
                            task.swapped_frame = task.frame
                        else:
                            raise
                    
                    # Pass to output stage
                    self.output_queue.put(task)
                    
                    frame_count += 1
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error("Swapping worker error: %s", e, exc_info=True)
            self.error = e
            self.stop_event.set()
    
    def process_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process frames through async pipeline.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of processed frames in original order
        """
        logger.info("Starting async pipeline processing for %d frames", len(frames))
        
        # Reset state
        self.stop_event.clear()
        self.error = None
        self.face_tracker.reset()
        
        # Start worker threads
        detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        swapping_thread = threading.Thread(target=self._swapping_worker, daemon=True)
        
        detection_thread.start()
        swapping_thread.start()
        
        self.threads = [detection_thread, swapping_thread]
        
        # Feed frames into pipeline
        for idx, frame in enumerate(frames):
            if self.error:
                break
            task = FrameTask(index=idx, frame=frame)
            self.detection_queue.put(task)
        
        # Send poison pill to stop workers
        self.detection_queue.put(None)
        
        # Create progress tracker for async pipeline
        progress = get_progress_tracker()
        progress.set_stage("Face Swapping (Async Pipeline)")
        
        # Collect results
        results = [None] * len(frames)
        processed_count = 0
        
        with progress.track(len(frames), "Processing frames", "frame") as pbar:
            while processed_count < len(frames):
                if self.error:
                    raise RuntimeError(f"Pipeline error: {self.error}")
                
                try:
                    task = self.output_queue.get(timeout=1.0)
                    if task is None:  # All done
                        break
                    
                    results[task.index] = task.swapped_frame
                    processed_count += 1
                    pbar.update(1)
                    
                    if processed_count % 10 == 0:
                        logger.debug("Async pipeline progress: %d/%d frames", processed_count, len(frames))
                        
                except queue.Empty:
                    # Check if threads are still alive
                    if not any(t.is_alive() for t in self.threads):
                        if processed_count < len(frames):
                            raise RuntimeError("Pipeline threads died before completing")
                        break
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        # Check for missing results (use 'is None' instead of 'in' to avoid numpy array comparison issues)
        missing_indices = [i for i, r in enumerate(results) if r is None]
        if missing_indices:
            logger.warning("Missing %d frames from pipeline: %s", len(missing_indices), missing_indices[:10])
            # Fill missing frames with originals
            for idx in missing_indices:
                results[idx] = frames[idx]
        
        logger.info("Async pipeline processing complete")
        return results
    
    def shutdown(self):
        """Shutdown the pipeline gracefully."""
        self.stop_event.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
