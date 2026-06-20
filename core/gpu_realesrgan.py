"""GPU-resident Real-ESRGAN inference for Wave 4 enhancement chain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from realesrgan import RealESRGANer

logger = logging.getLogger("FaceOff")


def _rgb_uint8_to_model_input(frame_rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """HWC uint8 RGB → NCHW float RGB on device."""
    return (
        frame_rgb.permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)


def _apply_mod_pad(img: torch.Tensor, scale: int) -> tuple[torch.Tensor, int, int]:
    mod_scale = 2 if scale == 2 else (4 if scale == 1 else None)
    pad_h = pad_w = 0
    if mod_scale is not None:
        _, _, h, w = img.shape
        if h % mod_scale != 0:
            pad_h = mod_scale - (h % mod_scale)
        if w % mod_scale != 0:
            pad_w = mod_scale - (w % mod_scale)
        if pad_h or pad_w:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    return img, pad_h, pad_w


def enhance_rgb_frame_gpu(
    frame_rgb: torch.Tensor,
    upsampler: "RealESRGANer",
    *,
    outscale: int,
    maintain_dimensions: bool,
    target_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Run Real-ESRGAN on a CUDA RGB uint8 frame; returns RGB uint8 on same device.
    """
    if frame_rgb.dim() != 3:
        raise ValueError(f"frame must be HWC, got {tuple(frame_rgb.shape)}")

    device = upsampler.device
    h, w = frame_rgb.shape[0], frame_rgb.shape[1]
    tile_size = int(getattr(upsampler, "tile_size", 0) or 0)

    if tile_size > 0 and (h > tile_size or w > tile_size):
        bgr = frame_rgb.detach().cpu().numpy()[:, :, ::-1]
        out_bgr, _ = upsampler.enhance(bgr, outscale=outscale)
        out_rgb = torch.from_numpy(out_bgr[:, :, ::-1].copy()).to(frame_rgb.device)
        if maintain_dimensions and target_size:
            ow, oh = target_size
            if (out_rgb.shape[1], out_rgb.shape[0]) != (ow, oh):
                out_rgb = (
                    F.interpolate(
                        out_rgb.permute(2, 0, 1).unsqueeze(0).float(),
                        size=(oh, ow),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
        return out_rgb

    img = _rgb_uint8_to_model_input(frame_rgb, device)
    pre_pad = int(getattr(upsampler, "pre_pad", 0) or 0)
    if pre_pad:
        img = F.pad(img, (0, pre_pad, 0, pre_pad), mode="reflect")

    scale = int(getattr(upsampler, "scale", outscale) or outscale)
    img, mod_pad_h, mod_pad_w = _apply_mod_pad(img, scale)

    if getattr(upsampler, "half", False):
        img = img.half()

    with torch.no_grad():
        output = upsampler.model(img)

    if getattr(upsampler, "half", False):
        output = output.float()

    out = output.squeeze(0).clamp(0, 1)
    if pre_pad:
        out = out[:, : -(pre_pad * scale) or None, : -(pre_pad * scale) or None]
    if mod_pad_h:
        out = out[:, :-mod_pad_h * scale, :]
    if mod_pad_w:
        out = out[:, :, :-mod_pad_w * scale]

    out_rgb = (out.permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)

    if maintain_dimensions and target_size:
        ow, oh = target_size
        if (out_rgb.shape[1], out_rgb.shape[0]) != (ow, oh):
            out_rgb = (
                F.interpolate(
                    out_rgb.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(oh, ow),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .clamp(0, 255)
                .to(torch.uint8)
            )

    return out_rgb