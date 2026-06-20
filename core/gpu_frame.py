"""GPU-resident frame buffers for chunked video/GIF processing (Wave 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class ChunkFrameBuffer:
    """
    Host frames plus optional GPU upload for a decode chunk.

    Detection still uses host numpy (InsightFace ORT). Swap can use GPU tensors
    and ORT IoBinding to avoid per-inference H2D copies of blob batches.
    """

    frames: List[np.ndarray]
    device_id: int
    _gpu_batch: Optional[torch.Tensor] = None

    def upload(self) -> torch.Tensor:
        """Upload uint8 HWC frames to GPU as NHWC batch (single H2D per chunk)."""
        if self._gpu_batch is not None:
            return self._gpu_batch
        if not self.frames:
            raise ValueError("Cannot upload empty chunk")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for GPU frame retention")

        torch.cuda.set_device(self.device_id)
        stacked = np.stack(self.frames, axis=0)
        pinned = torch.from_numpy(stacked).pin_memory()
        self._gpu_batch = pinned.to(
            device=f"cuda:{self.device_id}",
            dtype=torch.uint8,
            non_blocking=True,
        )
        torch.cuda.synchronize(self.device_id)
        return self._gpu_batch

    def frame_numpy(self, index: int) -> np.ndarray:
        return self.frames[index]

    def frame_gpu(self, index: int) -> torch.Tensor:
        batch = self.upload()
        return batch[index]

    def download_all(self) -> List[np.ndarray]:
        """Materialize processed host frames (single D2H per chunk when used)."""
        if self._gpu_batch is None:
            return [f.copy() for f in self.frames]
        torch.cuda.synchronize(self.device_id)
        host = self._gpu_batch.detach().cpu().numpy()
        return [host[i] for i in range(host.shape[0])]

    def download_contiguous(self) -> np.ndarray:
        """Single D2H copy as contiguous NHWC uint8 (for pinned NVENC pipe)."""
        if self._gpu_batch is None:
            return np.stack([f.copy() for f in self.frames], axis=0)
        torch.cuda.synchronize(self.device_id)
        return np.ascontiguousarray(self._gpu_batch.detach().cpu().numpy())

    def replace_from_numpy(self, frames: List[np.ndarray]) -> None:
        """Update host frames and drop stale GPU batch."""
        self.frames = frames
        self._gpu_batch = None

    def update_frame_gpu(self, index: int, frame: torch.Tensor) -> None:
        """Write a processed HWC uint8 CUDA frame back into the chunk batch."""
        batch = self.upload()
        if frame.device.type != "cuda":
            raise ValueError("update_frame_gpu requires a CUDA tensor")
        if frame.shape != batch[index].shape:
            raise ValueError(f"Frame shape mismatch: {frame.shape} vs {batch[index].shape}")
        batch[index].copy_(frame)

    def has_gpu_batch(self) -> bool:
        return self._gpu_batch is not None