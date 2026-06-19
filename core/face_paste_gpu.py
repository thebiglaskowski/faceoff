"""GPU paste-back for face swap (Wave 3 phase 2)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _cv2_inv_affine_to_theta(
    M_inv: np.ndarray,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert cv2 2x3 inverse affine (dst→src pixels) to grid_sample theta."""
    m = np.vstack([M_inv, [0.0, 0.0, 1.0]])

    dst_to_norm = np.array(
        [
            [2.0 / max(dst_w - 1, 1), 0.0, -1.0],
            [0.0, 2.0 / max(dst_h - 1, 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    src_norm_to_pix = np.array(
        [
            [(src_w - 1) / 2.0, 0.0, (src_w - 1) / 2.0],
            [0.0, (src_h - 1) / 2.0, (src_h - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    theta = dst_to_norm @ m @ src_norm_to_pix
    return torch.tensor(theta[:2, :], dtype=torch.float32, device=device).unsqueeze(0)


def _warp_affine_gpu(
    src: torch.Tensor,
    M_inv: np.ndarray,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """Warp NCHW float tensor with cv2-style inverse affine."""
    theta = _cv2_inv_affine_to_theta(
        M_inv, src.shape[2], src.shape[3], out_h, out_w, src.device
    )
    grid = F.affine_grid(theta, size=(src.shape[0], src.shape[1], out_h, out_w), align_corners=True)
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def paste_swapped_face_gpu(
    frame: torch.Tensor,
    bgr_fake: torch.Tensor,
    aimg: torch.Tensor,
    M: np.ndarray,
) -> torch.Tensor:
    """
    Paste swapped face onto a GPU uint8 HWC frame (in-place style return).

    Mirrors ``core.face_paste.paste_swapped_face`` using torch ops on CUDA.
    """
    if frame.dim() != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3, got {tuple(frame.shape)}")

    device = frame.device
    h, w = frame.shape[0], frame.shape[1]
    IM = cv2.invertAffineTransform(M.astype(np.float32))

    aimg_f = aimg.float() if aimg.is_floating_point() else aimg.float()
    fake_f = bgr_fake.float() if bgr_fake.is_floating_point() else bgr_fake.float()
    fake_diff = (fake_f - aimg_f).abs().mean(dim=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0

    ones = torch.ones((aimg.shape[0], aimg.shape[1]), device=device, dtype=torch.float32) * 255.0
    fake_nchw = fake_f.permute(2, 0, 1).unsqueeze(0)
    ones_nchw = ones.unsqueeze(0).unsqueeze(0)
    diff_nchw = fake_diff.unsqueeze(0).unsqueeze(0)

    warped_fake = _warp_affine_gpu(fake_nchw, IM, h, w).squeeze(0).permute(1, 2, 0)
    warped_white = _warp_affine_gpu(ones_nchw, IM, h, w).squeeze(0).squeeze(0)
    warped_diff = _warp_affine_gpu(diff_nchw, IM, h, w).squeeze(0).squeeze(0)

    warped_white = torch.where(warped_white > 20, torch.tensor(255.0, device=device), warped_white)
    fthresh = 10.0
    warped_diff = torch.where(warped_diff < fthresh, torch.zeros_like(warped_diff), torch.full_like(warped_diff, 255.0))

    mask = warped_white
    mask_inds = (mask >= 255.0).nonzero(as_tuple=False)
    if mask_inds.numel() > 0:
        mask_h = int(mask_inds[:, 0].max() - mask_inds[:, 0].min())
        mask_w = int(mask_inds[:, 1].max() - mask_inds[:, 1].min())
        mask_size = int((mask_h * mask_w) ** 0.5)
    else:
        mask_size = 10

    k = max(mask_size // 10, 10)
    if k % 2 == 0:
        k += 1
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    mask_4d = -F.max_pool2d(-mask_4d, kernel_size=k, stride=1, padding=k // 2)

    k2 = max(mask_size // 20, 5)
    if k2 % 2 == 0:
        k2 += 1
    blur = torch.ones((1, 1, k2, k2), device=device) / (k2 * k2)
    mask_4d = F.conv2d(mask_4d, blur, padding=k2 // 2)
    mask = (mask_4d.squeeze(0).squeeze(0) / 255.0).clamp(0.0, 1.0)

    base = frame.float()
    merged = mask.unsqueeze(2) * warped_fake + (1.0 - mask.unsqueeze(2)) * base
    return merged.clamp(0, 255).to(torch.uint8)