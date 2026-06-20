"""
Chunked streaming pipeline for video and GIF face-swap processing.
"""
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from core.face_processor import (
    filter_faces_by_confidence,
    sort_faces_by_position,
    FaceTracker,
)
from core.gpu_frame import ChunkFrameBuffer
from core.media_processor import MediaProcessor
from core.model_pool import get_model_pool
from processing.frame_batch import process_chunk_multi_gpu, process_frames_batch
from processing.in_memory_enhancement import InMemoryEnhancer
from processing.resolution_adaptive import ResolutionAdaptiveProcessor
from processing.restoration_session import RestorationSession
from processing.workload_profile import (
    WorkloadProfile,
    flag as profile_flag,
    log_workload_profile,
    resolve_workload_profile,
)
from utils import video_io
from utils.config_manager import config
from utils.memory_manager import MemoryManager, prepare_for_enhancement
from utils.progress import get_progress_tracker
from utils.temp_manager import get_temp_manager

logger = logging.getLogger("FaceOff")


def _stack_rgb_frames(frames: List[np.ndarray]) -> np.ndarray:
    """Stack HxWx3 uint8 frames into contiguous NHWC batch for pinned encode."""
    arrays: List[np.ndarray] = []
    for i, frame in enumerate(frames):
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"frame {i}: expected HxWx3 uint8, got shape {arr.shape}"
            )
        arrays.append(arr)
    return np.ascontiguousarray(np.stack(arrays, axis=0))


def _write_chunk_frames(
    writer: video_io.StreamingVideoWriter,
    processed: List[np.ndarray],
    *,
    use_pinned_encode: bool,
    frame_buffer: Optional[ChunkFrameBuffer],
    gpu_paste_active: bool,
) -> None:
    """Encode a processed chunk, using pinned batch write only when safe."""
    if not processed:
        return

    if (
        use_pinned_encode
        and gpu_paste_active
        and frame_buffer is not None
        and frame_buffer.has_gpu_batch()
        and len(processed) == len(frame_buffer.frames)
    ):
        contiguous = frame_buffer.download_contiguous()
        writer.write_frames_pinned(contiguous, len(processed))
        return

    if use_pinned_encode and len(processed) > 1:
        try:
            contiguous = _stack_rgb_frames(processed)
            writer.write_frames_pinned(contiguous, contiguous.shape[0])
            return
        except ValueError as exc:
            logger.warning("Pinned encode unavailable (%s) — per-frame fallback", exc)

    writer.write_frames(processed)


@dataclass
class StreamingContext:
    media_type: str
    dest_path: str
    output_dir: Path
    fps: float
    width: int
    height: int
    frame_durations: Optional[List[int]] = None
    audio_path: Optional[str] = None


def _detect_source_faces(
    processor: MediaProcessor,
    source_image: np.ndarray,
    face_confidence: float,
    device_ids: List[int],
) -> list:
    if len(device_ids) > 1:
        pool = get_model_pool()
        primary = pool.get_instances(device_ids)[0]
        src = primary.get_faces(np.array(source_image))
    else:
        src = processor.get_faces(np.array(source_image))
    src = filter_faces_by_confidence(src, face_confidence)
    return sort_faces_by_position(src)


def _effective_chunk_size(
    device_ids: List[int],
    requested: int,
    enhance: bool,
    outscale: int,
    profile: Optional[WorkloadProfile] = None,
) -> int:
    if profile is not None and profile.chunk_size is not None:
        requested = profile.chunk_size
    if not torch.cuda.is_available():
        size = requested
    else:
        mm = MemoryManager(device_ids[0])
        size = max(config.min_batch_size, min(requested, mm.get_optimal_batch_size(requested)))

    if enhance:
        divisor = max(1, min(outscale, 4))
        size = max(config.min_batch_size, size // divisor)
    return size


def _gpu_chain_active(
    enhance: bool,
    restoration_session,
    profile: Optional[WorkloadProfile],
) -> bool:
    return bool(
        enhance
        and profile_flag(
            profile, "enhancement_chain", config.gpu_enhancement_chain_enabled
        )
        and profile_flag(
            profile, "frame_retention", config.gpu_frame_retention_enabled
        )
        and profile_flag(profile, "paste_on_gpu", config.gpu_paste_on_gpu)
        and restoration_session is None
    )


def _process_chunk(
    chunk: List[np.ndarray],
    *,
    device_ids: List[int],
    gpu_instances,
    processor: MediaProcessor,
    src_faces: list,
    face_confidence: float,
    face_mappings,
    adaptive_processor,
    batch_size: int,
    face_tracker: FaceTracker,
    defer_download: bool = False,
    profile: Optional[WorkloadProfile] = None,
) -> Tuple[List[np.ndarray], Optional[ChunkFrameBuffer]]:
    frame_buffer = None
    if profile_flag(profile, "frame_retention", config.gpu_frame_retention_enabled) and device_ids:
        frame_buffer = ChunkFrameBuffer(chunk, device_ids[0])

    if len(device_ids) > 1 and gpu_instances:
        processed = process_chunk_multi_gpu(
            chunk,
            src_faces,
            device_ids,
            gpu_instances,
            face_confidence,
            face_mappings,
            face_tracker=face_tracker,
            adaptive_processor=adaptive_processor,
            batch_size=batch_size,
            frame_buffer=frame_buffer,
            defer_download=defer_download,
            profile=profile,
        )
    else:
        processed = process_frames_batch(
            chunk,
            src_faces,
            face_confidence,
            face_mappings,
            face_tracker,
            adaptive_processor,
            processor=processor,
            gpu_instance=gpu_instances[0] if gpu_instances else None,
            frame_buffer=frame_buffer,
            defer_download=defer_download,
            profile=profile,
        )

    if defer_download and frame_buffer is not None and frame_buffer.has_gpu_batch():
        return [], frame_buffer
    return processed, frame_buffer


def _process_chunk_with_oom_fallback(
    chunk: List[np.ndarray],
    memory_manager: MemoryManager,
    **kwargs,
) -> Tuple[List[np.ndarray], Optional[ChunkFrameBuffer]]:
    try:
        return _process_chunk(chunk, **kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        logger.warning("OOM on chunk — clearing cache and falling back to per-frame")
        memory_manager.clear_cache(force=True)
        results = []
        tracker: Optional[FaceTracker] = kwargs.get("face_tracker")
        fallback_kwargs = {k: v for k, v in kwargs.items() if k != "face_tracker"}
        fallback_kwargs["defer_download"] = False
        for frame in chunk:
            single_tracker = FaceTracker(iou_threshold=config.iou_threshold) if tracker else None
            batch, _ = _process_chunk(
                [frame],
                face_tracker=single_tracker or tracker,
                **fallback_kwargs,
            )
            results.extend(batch)
        return results, None


def _postprocess_chunk(
    processed: List[np.ndarray],
    *,
    restoration_session: Optional[RestorationSession],
    enhancer: Optional[InMemoryEnhancer],
    maintain_dimensions: bool,
    original_size: Optional[Tuple[int, int]],
    frame_buffer: Optional[ChunkFrameBuffer] = None,
) -> List[np.ndarray]:
    if frame_buffer is not None and frame_buffer.has_gpu_batch() and enhancer:
        enhancer.enhance_chunk_buffer(
            frame_buffer,
            maintain_dimensions=maintain_dimensions,
            original_size=original_size,
        )
        return frame_buffer.download_all()

    if restoration_session:
        processed = restoration_session.restore_rgb_frames(processed)
    if enhancer:
        processed = enhancer.enhance_rgb_frames(
            processed,
            maintain_dimensions=maintain_dimensions,
            original_size=original_size,
        )
    return processed


def process_streaming(
    processor: MediaProcessor,
    source_image: np.ndarray,
    ctx: StreamingContext,
    *,
    face_confidence: float = 0.5,
    device_ids: Optional[List[int]] = None,
    face_mappings: Optional[List[Tuple[int, int]]] = None,
    enhance: bool = False,
    tile_size: int = 256,
    outscale: int = 4,
    model_name: str = "RealESRGAN_x4plus",
    denoise_strength: float = 0.5,
    use_fp32: bool = False,
    pre_pad: int = 0,
    restore_faces: bool = False,
    restoration_weight: float = 0.5,
    adaptive_detection: Optional[bool] = None,
    detection_scale: Optional[float] = None,
    enhancement_model: str = "RealESRGAN",
    restoration_model: str = "GFPGAN",
    workload_profile: Optional[WorkloadProfile] = None,
) -> Tuple[None, Optional[str]]:
    if not device_ids:
        device_ids = [processor.device_id]

    profile = workload_profile or resolve_workload_profile(
        media_type=ctx.media_type,
        enhance=enhance,
        enhancement_model=enhancement_model,
        restore_faces=restore_faces,
        face_mappings=face_mappings,
        width=ctx.width,
        height=ctx.height,
        outscale=outscale,
        face_enhance_in_esrgan=config.streaming_video_face_enhance,
    )
    log_workload_profile(profile)

    if adaptive_detection is None:
        adaptive_detection = config.adaptive_detection_enabled
    if detection_scale is None:
        detection_scale = config.detection_scale

    adaptive_processor = (
        ResolutionAdaptiveProcessor(detection_scale=detection_scale)
        if adaptive_detection
        else None
    )

    progress = get_progress_tracker()
    progress.set_stage("🔍 Face Detection")
    progress.log("📸 Detecting faces in source image...")
    src_faces = _detect_source_faces(processor, source_image, face_confidence, device_ids)
    progress.log(f"✅ Found {len(src_faces)} source face(s)")

    if face_mappings:
        from utils.validation import validate_face_mappings_or_raise

        with video_io.open_streaming_reader(
            ctx.dest_path,
            fps=ctx.fps if ctx.media_type == "gif" else None,
            hwaccel=config.streaming_hwaccel_decode,
        ) as preview_reader:
            preview_frames = preview_reader.read_chunk(1)
        if preview_frames:
            preview_dst = processor.get_faces(preview_frames[0])
            preview_dst = filter_faces_by_confidence(preview_dst, face_confidence)
            preview_dst = sort_faces_by_position(preview_dst)
            validate_face_mappings_or_raise(
                face_mappings, len(src_faces), len(preview_dst)
            )

    enhancer: Optional[InMemoryEnhancer] = None
    restoration_session: Optional[RestorationSession] = None
    if enhance:
        processor.release_gpu_models()
        prepare_for_enhancement(device_ids[0], device_ids=device_ids)
        enhancer = InMemoryEnhancer(
            enhancement_model,
            model_name,
            device_ids,
            tile_size,
            outscale,
            denoise_strength,
            use_fp32,
            pre_pad,
            face_enhance=config.streaming_video_face_enhance,
        )
    if restore_faces:
        restoration_session = RestorationSession(
            restoration_model, device_ids[0], restoration_weight
        )

    pool = get_model_pool() if len(device_ids) > 1 else None
    gpu_instances = pool.get_instances(device_ids) if pool else None

    chunk_size = _effective_chunk_size(
        device_ids, config.streaming_chunk_size, enhance, outscale, profile
    )
    if enhance:
        logger.info(
            "Enhancement enabled: chunk_size=%d, outscale=%dx, face_in_esrgan=%s, model=%s",
            chunk_size,
            outscale,
            config.streaming_video_face_enhance,
            enhancement_model,
        )
    batch_size = config.batch_size
    face_tracker = FaceTracker(iou_threshold=config.iou_threshold)
    memory_manager = MemoryManager(device_ids[0])

    timestamp = int(time.time() * 1000)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.media_type == "video":
        output_path = ctx.output_dir.resolve() / f"swapped_{timestamp}.mp4"
    else:
        output_path = ctx.output_dir.resolve() / f"swapped_{timestamp}.gif"

    progress.set_stage("Face Swapping (Streaming)")
    total_frames = 0
    decode_fps = ctx.fps if ctx.media_type == "gif" else None

    use_gpu_chain = _gpu_chain_active(enhance, restoration_session, profile)
    defer_download = use_gpu_chain and profile_flag(
        profile, "defer_download", config.gpu_enhancement_chain_enabled
    )
    # FFmpeg CUDA zero-copy (hwaccel_output_format=cuda) breaks GIF filter graphs.
    use_zero_copy = (
        ctx.media_type == "video"
        and profile_flag(profile, "zero_copy", config.streaming_zero_copy_enabled)
        and config.streaming_hwaccel_decode
    )
    use_nvcodec = (
        profile_flag(profile, "use_nvcodec_decode", config.streaming_nvcodec_decode)
        and config.streaming_hwaccel_decode
        and ctx.media_type == "video"
    )
    # Pinned NVENC expects one contiguous host batch. Multi-GPU uses CPU paste and
    # per-GPU frame lists — keep pinned encode on single-GPU jobs only.
    use_pinned_encode = bool(
        (profile.pinned_encode if profile else use_gpu_chain)
        and len(device_ids) == 1
    )
    gpu_paste_active = bool(
        len(device_ids) == 1
        and profile_flag(profile, "frame_retention", config.gpu_frame_retention_enabled)
        and profile_flag(profile, "paste_on_gpu", config.gpu_paste_on_gpu)
        and not face_mappings
    )

    chunk_kwargs = dict(
        device_ids=device_ids,
        gpu_instances=gpu_instances,
        processor=processor,
        src_faces=src_faces,
        face_confidence=face_confidence,
        face_mappings=face_mappings,
        adaptive_processor=adaptive_processor,
        batch_size=batch_size,
        face_tracker=face_tracker,
        defer_download=defer_download,
        profile=profile,
    )

    try:
        with video_io.open_streaming_reader(
            ctx.dest_path,
            fps=decode_fps,
            hwaccel=config.streaming_hwaccel_decode,
            zero_copy=use_zero_copy,
            pinned_pool_size=chunk_size if use_zero_copy else 0,
            use_nvcodec=use_nvcodec,
        ) as reader:
            if ctx.media_type == "video":
                with video_io.StreamingVideoWriter(
                    output_path,
                    reader.width,
                    reader.height,
                    ctx.fps,
                    audio_path=ctx.audio_path,
                    codec=config.streaming_video_codec,
                    preset=config.streaming_video_preset,
                    crf=config.streaming_video_crf,
                    use_nvenc=config.streaming_nvenc_encode,
                ) as writer:
                    with progress.track(0, "Processing frames", "frame") as pbar:
                        while True:
                            chunk = reader.read_chunk(chunk_size)
                            if not chunk:
                                break
                            processed, frame_buffer = _process_chunk_with_oom_fallback(
                                chunk, memory_manager, **chunk_kwargs
                            )
                            processed = _postprocess_chunk(
                                processed,
                                restoration_session=restoration_session,
                                enhancer=enhancer,
                                maintain_dimensions=enhance,
                                original_size=(reader.width, reader.height) if enhance else None,
                                frame_buffer=frame_buffer,
                            )
                            _write_chunk_frames(
                                writer,
                                processed,
                                use_pinned_encode=use_pinned_encode,
                                frame_buffer=frame_buffer,
                                gpu_paste_active=gpu_paste_active,
                            )
                            total_frames += len(processed)
                            pbar.total = total_frames
                            pbar.n = total_frames
                            pbar.refresh()
                    if writer.frames_written == 0 or not writer.finalize():
                        raise RuntimeError("Streaming video encode failed")
            else:
                temp_manager = get_temp_manager()
                gif_temp = temp_manager.get_temp_dir("gif") / f"stream_{timestamp}"
                with video_io.StreamingGifWriter(
                    output_path, gif_temp, durations=ctx.frame_durations
                ) as gif_writer:
                    with progress.track(0, "Processing GIF frames", "frame") as pbar:
                        while True:
                            chunk = reader.read_chunk(chunk_size)
                            if not chunk:
                                break
                            processed, frame_buffer = _process_chunk_with_oom_fallback(
                                chunk, memory_manager, **chunk_kwargs
                            )
                            processed = _postprocess_chunk(
                                processed,
                                restoration_session=restoration_session,
                                enhancer=enhancer,
                                maintain_dimensions=True,
                                original_size=(reader.width, reader.height),
                                frame_buffer=frame_buffer,
                            )
                            gif_writer.write_frames(processed)
                            total_frames += len(processed)
                            pbar.total = total_frames
                            pbar.n = total_frames
                            pbar.refresh()
                    if gif_writer.frames_written == 0 or not gif_writer.finalize():
                        raise RuntimeError("Streaming GIF encode failed")
    finally:
        if restoration_session:
            restoration_session.close()

    if enhance:
        progress.log("✅ Enhancement complete (single-pass, multi-GPU)")

    logger.info(
        "Streaming %s complete: %d frames → %s",
        ctx.media_type,
        total_frames,
        output_path,
    )
    return None, str(output_path)


def build_video_context(dest_path: str, output_dir: Path) -> StreamingContext:
    meta = video_io.probe_video(dest_path)
    fps = float(meta.get("fps") or 30.0)
    audio_path = None
    if meta.get("has_audio"):
        audio_path = video_io.extract_audio(dest_path, output_dir)
    return StreamingContext(
        media_type="video",
        dest_path=dest_path,
        output_dir=output_dir,
        fps=fps,
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
        audio_path=str(audio_path) if audio_path else None,
    )


def build_gif_context(dest_path: str, output_dir: Path) -> StreamingContext:
    meta = video_io.probe_video(dest_path)
    fps = float(config.streaming_gif_decode_fps)
    durations = video_io.extract_gif_frame_durations(dest_path)
    return StreamingContext(
        media_type="gif",
        dest_path=dest_path,
        output_dir=output_dir,
        fps=fps,
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
        frame_durations=durations,
    )