"""Cached face-restoration session for streaming jobs."""
import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger("FaceOff")


class RestorationSession:
    """Reuse one restorer for an entire streaming/image job."""

    def __init__(
        self,
        restoration_model: str,
        gpu_id: int,
        restoration_weight: float = 0.5,
    ):
        self.restoration_model = restoration_model
        self.gpu_id = gpu_id
        self.restoration_weight = restoration_weight
        self._restorer = None
        self._fn = None
        self._init_restorer()

    def _init_restorer(self) -> None:
        if self.restoration_model == "CodeFormer":
            from processing.codeformer_restoration import _get_codeformer_restorer

            self._restorer = _get_codeformer_restorer(device_id=self.gpu_id)
            self._fn = lambda bgr, w: self._restorer.restore_faces_in_frame(
                bgr, fidelity_weight=w
            )
        else:
            from processing.face_restoration import FaceRestorer

            self._restorer = FaceRestorer(device_id=self.gpu_id)
            self._fn = lambda bgr, w: self._restorer.restore_faces_in_frame(bgr, weight=w)

    def restore_rgb_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        restored = []
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out_bgr = self._fn(bgr, self.restoration_weight)
            restored.append(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        return restored

    def restore_rgb_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.restore_rgb_frames([frame])[0]

    def close(self) -> None:
        if self._restorer is None:
            return
        if self.restoration_model == "CodeFormer":
            return
        try:
            self._restorer.cleanup()
        except Exception as exc:
            logger.debug("Restoration session cleanup: %s", exc)
        self._restorer = None