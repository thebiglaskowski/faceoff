"""
Unified batched face-swap processing for streaming and legacy paths.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch

from core.face_processor import FaceTracker, filter_faces_by_confidence
from core.gpu_detection import detect_faces_from_gpu
from core.gpu_frame import ChunkFrameBuffer
from core.media_processor import MediaProcessor
from core.model_pool import GPUModelInstance
from processing.gpu_scheduler import assign_frames_to_gpus, frame_pixel_weights
from processing.resolution_adaptive import ResolutionAdaptiveProcessor
from processing.workload_profile import WorkloadProfile, flag as profile_flag
from utils.config_manager import config

logger = logging.getLogger("FaceOff")


def _gpu_paste_enabled(
    frame_buffer: Optional[ChunkFrameBuffer],
    face_mappings: Optional[List[Tuple[int, int]]],
    swap_gpu: Optional[GPUModelInstance],
    profile: Optional[WorkloadProfile] = None,
) -> bool:
    return bool(
        frame_buffer is not None
        and profile_flag(profile, "frame_retention", config.gpu_frame_retention_enabled)
        and profile_flag(profile, "paste_on_gpu", config.gpu_paste_on_gpu)
        and not face_mappings
        and swap_gpu is not None
    )


def _resolve_swap_gpu(
    processor: Optional[MediaProcessor],
    gpu_instance: Optional[GPUModelInstance],
) -> Optional[GPUModelInstance]:
    if gpu_instance is not None:
        return gpu_instance
    if isinstance(processor, MediaProcessor):
        processor._ensure_bound()
        return processor._gpu
    return None


def _build_detect_fn(
    adaptive_processor: Optional[ResolutionAdaptiveProcessor],
    processor: Optional[MediaProcessor],
    gpu_instance: Optional[GPUModelInstance],
):
    if adaptive_processor and processor:
        return lambda f: adaptive_processor.detect_faces_adaptive(processor, f)
    if adaptive_processor and gpu_instance:
        return lambda f: adaptive_processor.detect_faces_adaptive_gpu(gpu_instance, f)
    if gpu_instance:
        return lambda f: gpu_instance.get_faces(f)
    return lambda f: processor.get_faces(f)


def _gpu_detection_enabled(
    frame_buffer: Optional[ChunkFrameBuffer],
    swap_gpu: Optional[GPUModelInstance],
    profile: Optional[WorkloadProfile] = None,
) -> bool:
    return bool(
        frame_buffer is not None
        and profile_flag(profile, "frame_retention", config.gpu_frame_retention_enabled)
        and profile_flag(profile, "detection_on_gpu", config.gpu_detection_on_gpu)
        and swap_gpu is not None
        and torch.cuda.is_available()
    )


def _detect_faces_for_batch(
    frames: List[np.ndarray],
    detect_fn,
    face_confidence: float,
    face_tracker: Optional[FaceTracker],
    *,
    frame_buffer: Optional[ChunkFrameBuffer] = None,
    frame_indices: Optional[List[int]] = None,
    swap_gpu: Optional[GPUModelInstance] = None,
    adaptive_processor: Optional[ResolutionAdaptiveProcessor] = None,
    profile: Optional[WorkloadProfile] = None,
) -> list:
    use_gpu_det = _gpu_detection_enabled(frame_buffer, swap_gpu, profile)
    if frame_indices is None:
        frame_indices = list(range(len(frames)))

    if use_gpu_det and frame_buffer is not None:
        all_dst_faces = []
        for local_idx, frame in enumerate(frames):
            buf_idx = (
                frame_indices[local_idx]
                if local_idx < len(frame_indices)
                else local_idx
            )
            try:
                faces = detect_faces_from_gpu(
                    swap_gpu,
                    frame_buffer.frame_gpu(buf_idx),
                    adaptive_processor,
                )
            except Exception as exc:
                logger.debug("GPU detection fallback for frame %d: %s", buf_idx, exc)
                faces = detect_fn(frame)
            all_dst_faces.append(faces)
    else:
        workers = max(1, config.workers_per_gpu) if len(frames) > 1 else 1
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                all_dst_faces = list(executor.map(detect_fn, frames))
        else:
            all_dst_faces = [detect_fn(frame) for frame in frames]
    all_dst_faces = [
        filter_faces_by_confidence(faces, face_confidence) for faces in all_dst_faces
    ]
    if face_tracker:
        all_dst_faces = [face_tracker.track_faces(faces) for faces in all_dst_faces]
    return all_dst_faces


def _swap_frame(
    frame: np.ndarray,
    dst_faces: list,
    src_faces: list,
    face_mappings: Optional[List[Tuple[int, int]]],
    processor: Optional[MediaProcessor] = None,
    gpu_instance: Optional[GPUModelInstance] = None,
    *,
    frame_buffer: Optional[ChunkFrameBuffer] = None,
    frame_index: Optional[int] = None,
    gpu_paste: bool = False,
):
    swap_gpu = _resolve_swap_gpu(processor, gpu_instance)
    swap_fn = swap_gpu.swap_face if swap_gpu else processor.swap_face
    needs_copy = bool(face_mappings) or not src_faces
    swapped = frame.copy() if needs_copy else frame
    frame_gpu = None
    if gpu_paste and frame_buffer is not None and frame_index is not None:
        frame_gpu = frame_buffer.frame_gpu(frame_index)

    if face_mappings:
        for src_idx, dst_idx in face_mappings:
            if (
                src_idx < len(src_faces)
                and dst_idx < len(dst_faces)
                and dst_faces[dst_idx] is not None
            ):
                swapped = swap_fn(swapped, dst_faces[dst_idx], src_faces[src_idx])
    elif src_faces:
        valid_faces = [f for f in dst_faces if f is not None]
        if valid_faces:
            if swap_gpu is not None:
                use_iobind = config.gpu_frame_retention_enabled
                swapped = swap_gpu.swap_face_batch(
                    swapped,
                    valid_faces,
                    src_faces[0],
                    use_iobinding=use_iobind,
                    frame_gpu=frame_gpu,
                    paste_on_gpu=gpu_paste,
                )
            elif processor is not None:
                swapped = processor.swap_face_batch(swapped, valid_faces, src_faces[0])
            else:
                for face in valid_faces:
                    swapped = swap_fn(swapped, face, src_faces[0])
    return swapped


def process_frames_batch(
    frames: List[np.ndarray],
    src_faces: list,
    face_confidence: float,
    face_mappings: Optional[List[Tuple[int, int]]] = None,
    face_tracker: Optional[FaceTracker] = None,
    adaptive_processor: Optional[ResolutionAdaptiveProcessor] = None,
    processor: Optional[MediaProcessor] = None,
    gpu_instance: Optional[GPUModelInstance] = None,
    frame_buffer: Optional[ChunkFrameBuffer] = None,
    frame_indices: Optional[List[int]] = None,
    defer_download: bool = False,
    profile: Optional[WorkloadProfile] = None,
    gpu_paste_override: Optional[bool] = None,
) -> List[np.ndarray]:
    """Process a batch of frames on a single GPU with batched multi-face swap."""
    if frame_buffer is not None and profile_flag(
        profile, "frame_retention", config.gpu_frame_retention_enabled
    ):
        try:
            frame_buffer.upload()
        except Exception as exc:
            logger.debug("Chunk GPU upload skipped: %s", exc)

    swap_gpu = _resolve_swap_gpu(processor, gpu_instance)
    if gpu_paste_override is not None:
        gpu_paste = gpu_paste_override
    else:
        gpu_paste = _gpu_paste_enabled(frame_buffer, face_mappings, swap_gpu, profile)
    if frame_indices is None and frame_buffer is not None:
        frame_indices = list(range(len(frames)))

    detect_fn = _build_detect_fn(adaptive_processor, processor, gpu_instance)
    all_dst_faces = _detect_faces_for_batch(
        frames,
        detect_fn,
        face_confidence,
        face_tracker,
        frame_buffer=frame_buffer,
        frame_indices=frame_indices,
        swap_gpu=swap_gpu,
        adaptive_processor=adaptive_processor,
        profile=profile,
    )

    results = []
    for local_idx, (frame, dst_faces) in enumerate(zip(frames, all_dst_faces)):
        buf_idx = (
            frame_indices[local_idx]
            if frame_indices is not None and local_idx < len(frame_indices)
            else local_idx
        )
        swapped = _swap_frame(
            frame,
            dst_faces,
            src_faces,
            face_mappings,
            processor=processor,
            gpu_instance=gpu_instance,
            frame_buffer=frame_buffer,
            frame_index=buf_idx,
            gpu_paste=gpu_paste,
        )
        if isinstance(swapped, torch.Tensor):
            if frame_buffer is not None:
                frame_buffer.update_frame_gpu(buf_idx, swapped)
                swapped = None
            else:
                swapped = swapped.detach().cpu().numpy()
        results.append(swapped)

    if (
        gpu_paste
        and not defer_download
        and frame_buffer is not None
        and frame_buffer.has_gpu_batch()
        and len(frames) == len(frame_buffer.frames)
    ):
        return frame_buffer.download_all()

    return [r if r is not None else frames[i] for i, r in enumerate(results)]


def process_chunk_multi_gpu(
    frames: List[np.ndarray],
    src_faces: list,
    device_ids: List[int],
    gpu_instances: list,
    face_confidence: float,
    face_mappings: Optional[List[Tuple[int, int]]] = None,
    face_tracker: Optional[FaceTracker] = None,
    adaptive_processor: Optional[ResolutionAdaptiveProcessor] = None,
    batch_size: int = 8,
    frame_buffer: Optional[ChunkFrameBuffer] = None,
    defer_download: bool = False,
    profile: Optional[WorkloadProfile] = None,
) -> List[np.ndarray]:
    """Process a chunk across multiple GPUs with VRAM-weighted frame assignment."""
    if len(device_ids) <= 1:
        tracker = face_tracker or FaceTracker(iou_threshold=config.iou_threshold)
        return process_frames_batch(
            frames,
            src_faces,
            face_confidence,
            face_mappings,
            face_tracker=tracker,
            adaptive_processor=adaptive_processor,
            gpu_instance=gpu_instances[0],
            frame_buffer=frame_buffer,
            defer_download=defer_download,
            profile=profile,
        )

    swap_gpu = gpu_instances[0] if gpu_instances else None
    # GPU paste writes into a single shared ChunkFrameBuffer on the primary GPU.
    # Secondary GPUs cannot update that buffer, so multi-GPU jobs must use CPU paste.
    gpu_paste = _gpu_paste_enabled(frame_buffer, face_mappings, swap_gpu, profile) and len(
        device_ids
    ) == 1
    if gpu_paste and frame_buffer is not None:
        try:
            frame_buffer.upload()
        except Exception as exc:
            logger.debug("Chunk GPU upload skipped: %s", exc)
            gpu_paste = False

    assignments = assign_frames_to_gpus(
        len(frames),
        device_ids,
        frame_weights=frame_pixel_weights(frames),
    )
    results: List[Optional[np.ndarray]] = [None] * len(frames)
    lock = threading.Lock()

    def _worker(dev_id: int, indices: List[int]) -> None:
        if not indices:
            return
        gpu_inst = next(g for g in gpu_instances if g.device_id == dev_id)
        tracker = FaceTracker(iou_threshold=config.iou_threshold)
        subset = [frames[i] for i in indices]
        chunk_buf = frame_buffer if dev_id == device_ids[0] else None
        for start in range(0, len(subset), batch_size):
            batch = subset[start : start + batch_size]
            batch_indices = indices[start : start + batch_size]
            processed = process_frames_batch(
                batch,
                src_faces,
                face_confidence,
                face_mappings,
                face_tracker=tracker,
                adaptive_processor=adaptive_processor,
                gpu_instance=gpu_inst,
                frame_buffer=chunk_buf,
                frame_indices=batch_indices,
                defer_download=defer_download,
                profile=profile,
                gpu_paste_override=gpu_paste,
            )
            if not gpu_paste:
                with lock:
                    for idx, frame in zip(batch_indices, processed):
                        results[idx] = frame

    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = [
            executor.submit(_worker, dev_id, assignments.get(dev_id, []))
            for dev_id in device_ids
        ]
        for fut in futures:
            fut.result()

    if (
        gpu_paste
        and not defer_download
        and frame_buffer is not None
        and frame_buffer.has_gpu_batch()
    ):
        return frame_buffer.download_all()

    return [r if r is not None else frames[i] for i, r in enumerate(results)]