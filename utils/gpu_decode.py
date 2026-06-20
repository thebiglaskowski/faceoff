"""Optional PyNvVideoCodec NVDEC decode (Wave 5)."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("FaceOff")

_NVcodec_AVAILABLE: bool | None = None
_LEGACY_NVcodec_AVAILABLE: bool | None = None


def _probe_pynvvideocodec() -> bool:
    global _NVcodec_AVAILABLE
    if _NVcodec_AVAILABLE is not None:
        return _NVcodec_AVAILABLE
    try:
        import PyNvVideoCodec  # noqa: F401

        _NVcodec_AVAILABLE = True
        logger.debug("PyNvVideoCodec available for NVDEC decode")
    except ImportError:
        _NVcodec_AVAILABLE = False
    return _NVcodec_AVAILABLE


def _probe_legacy_pynvcodec() -> bool:
    global _LEGACY_NVcodec_AVAILABLE
    if _LEGACY_NVcodec_AVAILABLE is not None:
        return _LEGACY_NVcodec_AVAILABLE
    try:
        import PyNvCodec  # noqa: F401

        _LEGACY_NVcodec_AVAILABLE = True
        logger.debug("Legacy PyNvCodec available for GPU decode")
    except ImportError:
        _LEGACY_NVcodec_AVAILABLE = False
    return _LEGACY_NVcodec_AVAILABLE


def nvcodec_decode_available() -> bool:
    """True when PyNvVideoCodec or legacy PyNvCodec is importable."""
    return _probe_pynvvideocodec() or _probe_legacy_pynvcodec()


def _decoded_frame_to_numpy(frame) -> np.ndarray:
    """Copy a PyNvVideoCodec RGB DecodedFrame into a contiguous HWC uint8 array."""
    shape = tuple(int(x) for x in frame.shape)
    view = frame.cuda()[0]
    size = int(np.prod(shape))
    buf = (ctypes.c_uint8 * size).from_address(int(view.dataptr))
    return np.ctypeslib.as_array(buf).reshape(shape).copy()


class NvCodecFrameReader:
    """Decode video frames via PyNvVideoCodec SimpleDecoder (NVDEC)."""

    def __init__(
        self,
        media_path: str,
        fps: Optional[float] = None,
        pinned_pool_size: int = 0,
        gpu_id: int = 0,
    ):
        import PyNvVideoCodec as nvc

        self.media_path = str(media_path)
        self._decoder = nvc.CreateSimpleDecoder(
            self.media_path,
            gpuid=gpu_id,
            useDeviceMemory=False,
            decoderCacheSize=1,
            outputColorType=nvc.OutputColorType.RGB,
        )
        meta = self._decoder.get_stream_metadata()
        self.width = int(meta.width)
        self.height = int(meta.height)
        native_fps = float(meta.average_fps or 30.0)
        self.fps = float(fps or native_fps)
        if self.fps <= 0:
            self.fps = native_fps if native_fps > 0 else 30.0
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid media dimensions for {self.media_path}")

        self._total_frames = int(getattr(meta, "num_frames", 0) or 0)
        self._pinned_pool = None
        if pinned_pool_size > 0:
            from utils.pinned_pool import PinnedFramePool

            self._pinned_pool = PinnedFramePool(
                self.height, self.width, pinned_pool_size
            )

        self.frames_read = 0
        self._exhausted = False
        logger.info(
            "NVCodec decode started: %s (%dx%d @ %.2f fps, pinned=%s)",
            Path(self.media_path).name,
            self.width,
            self.height,
            self.fps,
            self._pinned_pool is not None,
        )

    def read_chunk(self, count: int) -> List[np.ndarray]:
        if self._exhausted or count <= 0:
            return []

        batch = self._decoder.get_batch_frames(count)
        if not batch:
            self._exhausted = True
            return []

        frames: List[np.ndarray] = []
        for i, decoded in enumerate(batch):
            rgb = _decoded_frame_to_numpy(decoded)
            if self._pinned_pool is not None:
                frame = self._pinned_pool.borrow(i)
                np.copyto(frame, rgb)
            else:
                frame = rgb
            frames.append(frame)
            self.frames_read += 1

        if len(batch) < count:
            self._exhausted = True
        return frames

    def close(self) -> None:
        self._decoder = None

    def __enter__(self) -> "NvCodecFrameReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False