"""GPU-resident face detection for Wave 3 phase 3.

Downscales on GPU and performs a minimal D2H of the detection-sized frame
before InsightFace ORT inference — avoids full-resolution host copies.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.model_pool import GPUModelInstance
    from processing.resolution_adaptive import ResolutionAdaptiveProcessor

logger = __import__("logging").getLogger("FaceOff")


def downscale_frame_gpu(
    frame: torch.Tensor,
    scale: float,
    min_resolution: int,
) -> tuple[torch.Tensor, float]:
    """Downscale HWC uint8 CUDA frame; returns (tensor, scale_factor)."""
    if frame.dim() != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3, got {tuple(frame.shape)}")

    h, w = frame.shape[0], frame.shape[1]
    scale = float(np.clip(scale, 0.25, 1.0))
    if min(h, w) * scale < min_resolution:
        scale = min_resolution / float(min(h, w))
    if scale >= 0.999:
        return frame, 1.0

    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    nchw = frame.permute(2, 0, 1).unsqueeze(0).float()
    scaled = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = scaled.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
    actual = new_w / float(w)
    return out, actual


def detect_faces_from_gpu(
    gpu_instance: "GPUModelInstance",
    frame_gpu: torch.Tensor,
    adaptive_processor: Optional["ResolutionAdaptiveProcessor"] = None,
) -> List:
    """
    Detect faces using a GPU-resident frame.

    Downscales on GPU when adaptive processing is enabled, then D2H only the
    smaller detection tensor for InsightFace.
    """
    det_frame = frame_gpu
    scale_factor = 1.0
    if adaptive_processor is not None:
        if adaptive_processor.should_downscale(frame_gpu.shape):
            det_frame, scale_factor = downscale_frame_gpu(
                frame_gpu,
                adaptive_processor.detection_scale,
                adaptive_processor.min_resolution,
            )

    host_bgr = det_frame.detach().cpu().numpy()[:, :, ::-1]
    faces = gpu_instance.get_faces(host_bgr)
    if adaptive_processor is not None and scale_factor != 1.0:
        faces = adaptive_processor.scale_face_coordinates(faces, scale_factor)
    return faces