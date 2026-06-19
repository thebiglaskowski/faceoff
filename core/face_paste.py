"""Shared face-swap paste-back logic (warp, mask, blend)."""

import cv2
import numpy as np


def ensure_rgb_image(image: np.ndarray) -> np.ndarray:
    """Return a contiguous HxWx3 uint8 image."""
    out = image.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    elif out.ndim == 3 and out.shape[2] == 1:
        out = np.squeeze(out, axis=-1)
        out = np.stack([out] * 3, axis=-1)
    return np.ascontiguousarray(out)


def paste_swapped_face(
    frame: np.ndarray,
    bgr_fake: np.ndarray,
    aimg: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    Paste a swapped face crop back onto the frame using an affine warp and mask blend.

    Args:
        frame: Full frame (modified in place via returned array)
        bgr_fake: Swapped face crop in BGR, same size as aimg
        aimg: Original aligned crop used for differencing
        M: Affine transform from aligned crop to frame coordinates

    Returns:
        Frame with the swapped face blended in
    """
    swapped = ensure_rgb_image(frame)

    fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0

    IM = cv2.invertAffineTransform(M)
    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
    bgr_fake = cv2.warpAffine(
        bgr_fake, IM, (swapped.shape[1], swapped.shape[0]), borderValue=0.0
    )
    img_white = cv2.warpAffine(
        img_white, IM, (swapped.shape[1], swapped.shape[0]), borderValue=0.0
    )
    fake_diff = cv2.warpAffine(
        fake_diff, IM, (swapped.shape[1], swapped.shape[0]), borderValue=0.0
    )
    img_white[img_white > 20] = 255
    fthresh = 10
    fake_diff[fake_diff < fthresh] = 0
    fake_diff[fake_diff >= fthresh] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
    else:
        mask_size = 10
    k = max(mask_size // 10, 10)
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)

    k2 = max(mask_size // 20, 5)
    blur_size = tuple(2 * i + 1 for i in (k2, k2))
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    img_mask /= 255
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * swapped.astype(np.float32)
    return fake_merged.astype(np.uint8)