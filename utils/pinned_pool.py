"""Pinned host frame pools for overlap-friendly decode → H2D (Wave 3 phase 3)."""

from __future__ import annotations

from typing import List

import numpy as np
import torch


class PinnedFramePool:
    """Reuse pinned HWC uint8 buffers for FFmpeg rawvideo decode."""

    def __init__(self, height: int, width: int, capacity: int):
        self.height = height
        self.width = width
        self.capacity = max(1, capacity)
        self._slots: List[np.ndarray] = []
        for _ in range(self.capacity):
            host = np.empty((height, width, 3), dtype=np.uint8)
            torch.from_numpy(host).pin_memory()
            self._slots.append(host)

    def borrow(self, index: int) -> np.ndarray:
        return self._slots[index % self.capacity]