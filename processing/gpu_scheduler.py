"""
VRAM-aware frame assignment for multi-GPU video/GIF processing.
"""
import logging
from typing import Dict, List

import torch

from utils.memory_manager import MemoryManager

logger = logging.getLogger("FaceOff")


def _gpu_free_vram_mb(device_id: int) -> float:
    if not torch.cuda.is_available():
        return 100.0
    try:
        free_bytes, _total = torch.cuda.mem_get_info(device_id)
        return max(free_bytes / (1024 * 1024), 100.0)
    except Exception:
        return max(MemoryManager(device_id).get_memory_stats()["free_mb"], 100.0)


def assign_frames_to_gpus(
    num_frames: int,
    device_ids: List[int],
    frame_weights: List[float] | None = None,
) -> Dict[int, List[int]]:
    """
    Assign frame indices to GPUs proportional to free VRAM and optional per-frame cost.

    Args:
        num_frames: Total frames in the chunk.
        device_ids: CUDA device IDs.
        frame_weights: Optional per-frame cost (e.g. pixel count); same length as chunk.

    Returns:
        Dict mapping device_id → list of frame indices (original order within chunk).
    """
    if num_frames <= 0:
        return {d: [] for d in device_ids}
    if len(device_ids) == 1:
        return {device_ids[0]: list(range(num_frames))}

    vram_weights = [_gpu_free_vram_mb(d) for d in device_ids]
    total_vram = sum(vram_weights)

    if frame_weights and len(frame_weights) == num_frames:
        total_cost = sum(frame_weights) or float(num_frames)
        target_counts = [
            max(1, int(round(num_frames * (vw / total_vram))))
            for vw in vram_weights
        ]
        # Weighted round-robin using cumulative cost
        assignments: Dict[int, List[int]] = {d: [] for d in device_ids}
        gpu_loads = [0.0] * len(device_ids)
        for idx, cost in enumerate(frame_weights):
            gpu_idx = min(range(len(device_ids)), key=lambda i: gpu_loads[i] / vram_weights[i])
            assignments[device_ids[gpu_idx]].append(idx)
            gpu_loads[gpu_idx] += cost
        return assignments

    # Proportional count split with remainder distribution
    base_counts = [int(num_frames * (vw / total_vram)) for vw in vram_weights]
    remainder = num_frames - sum(base_counts)
    for i in range(remainder):
        base_counts[i % len(device_ids)] += 1

    assignments = {d: [] for d in device_ids}
    frame_idx = 0
    for dev_id, count in zip(device_ids, base_counts):
        assignments[dev_id] = list(range(frame_idx, frame_idx + count))
        frame_idx += count

    logger.debug(
        "VRAM frame split across %d GPUs: %s",
        len(device_ids),
        {d: len(v) for d, v in assignments.items()},
    )
    return assignments


def frame_pixel_weights(frames: list) -> List[float]:
    """Per-frame cost estimate based on resolution (H×W)."""
    weights = []
    for frame in frames:
        if frame is None:
            weights.append(1.0)
        else:
            h, w = frame.shape[:2]
            weights.append(float(h * w))
    return weights