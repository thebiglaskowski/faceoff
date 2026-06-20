"""In-memory enhancement with multi-GPU sharding and batched model reuse."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from processing.gpu_scheduler import assign_frames_to_gpus
from utils.config_manager import config
from utils.memory_manager import select_enhancement_gpu

logger = logging.getLogger("FaceOff")

# HAT/SwinIR share PyTorch CUDA allocators; parallel multi-GPU inference races on WSL2.
_SINGLE_GPU_ENHANCEMENT_BACKENDS = frozenset({"HAT", "SwinIR"})


def resolve_enhancement_device_ids(
    device_ids: List[int],
    enhancement_model: str,
) -> List[int]:
    """Pick enhancement GPU(s); transformer backends stay on one GPU for stability."""
    if len(device_ids) <= 1:
        return list(device_ids)

    if enhancement_model in _SINGLE_GPU_ENHANCEMENT_BACKENDS:
        gpu = select_enhancement_gpu(device_ids[0], device_ids)
        logger.info(
            "%s enhancement uses GPU %d only (%d GPUs available; "
            "multi-GPU PyTorch enhancement disabled for CUDA stability)",
            enhancement_model,
            gpu,
            len(device_ids),
        )
        return [gpu]

    if not config.enhancement_multi_gpu_enabled:
        gpu = select_enhancement_gpu(device_ids[0], device_ids)
        logger.info(
            "Enhancement uses GPU %d only (enhancement.multi_gpu_enabled=false)",
            gpu,
        )
        return [gpu]

    return list(device_ids)


class InMemoryEnhancer:
    """Enhance RGB frame batches without per-frame cache churn or disk I/O."""

    def __init__(
        self,
        enhancement_model: str,
        model_name: str,
        device_ids: List[int],
        tile_size: int = 256,
        outscale: int = 4,
        denoise_strength: float = 0.5,
        use_fp32: bool = False,
        pre_pad: int = 0,
        face_enhance: bool = False,
    ):
        self.enhancement_model = enhancement_model
        self.model_name = model_name
        self.device_ids = resolve_enhancement_device_ids(device_ids, enhancement_model)
        self.tile_size = tile_size
        self.outscale = outscale
        self.denoise_strength = denoise_strength
        self.use_fp32 = use_fp32
        self.pre_pad = pre_pad
        # GFPGAN-in-ESRGAN is ~2x slower per frame; off by default for streaming.
        self.face_enhance = face_enhance

        if self.enhancement_model == "HAT" and self.device_ids:
            from processing.hat_enhancement import preload_hat_models

            preload_hat_models(self.device_ids, self._resolve_hat_model())

    def _resolve_hat_model(self) -> str:
        return (
            self.model_name
            if self.model_name.startswith("HAT_")
            else "HAT_Base_4x_ImageNet"
        )

    def _resolve_swinir_model(self) -> str:
        return (
            self.model_name
            if self.model_name.startswith("Swin")
            else "Swin2SR_RealWorld_x4"
        )

    def _enhance_bgr_batch(self, bgr_frames: List[np.ndarray], gpu_id: int) -> List[np.ndarray]:
        from processing.enhancement import enhance_image
        from processing.hat_enhancement import enhance_image_batch_hat
        from processing.swinir_enhancement import enhance_image_swinir

        if self.enhancement_model == "HAT":
            results = enhance_image_batch_hat(
                bgr_frames,
                model_name=self._resolve_hat_model(),
                gpu_id=gpu_id,
                tile_size=self.tile_size,
                batch_size=max(1, config.batch_size),
                clear_cache=False,
            )
            return [r if r is not None else bgr_frames[i] for i, r in enumerate(results)]

        if self.enhancement_model == "SwinIR":
            out = []
            swinir_model = self._resolve_swinir_model()
            for bgr in bgr_frames:
                enhanced = enhance_image_swinir(
                    bgr, model_name=swinir_model, gpu_id=gpu_id, clear_cache=False
                )
                out.append(enhanced if enhanced is not None else bgr)
            return out

        out = []
        for bgr in bgr_frames:
            enhanced = enhance_image(
                bgr,
                tile_size=self.tile_size,
                outscale=self.outscale,
                gpu_id=gpu_id,
                model_name=self.model_name,
                denoise_strength=self.denoise_strength,
                use_fp32=self.use_fp32,
                pre_pad=self.pre_pad,
                face_enhance=self.face_enhance,
                clear_cache=False,
            )
            out.append(enhanced if enhanced is not None else bgr)
        return out

    def enhance_rgb_frames(
        self,
        frames: List[np.ndarray],
        maintain_dimensions: bool = False,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        if not frames:
            return []

        if len(self.device_ids) <= 1:
            gpu_id = self.device_ids[0]
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
            bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
            enhanced_bgr = self._enhance_bgr_batch(bgr_frames, gpu_id)
            return self._bgr_list_to_rgb(
                enhanced_bgr, frames, maintain_dimensions, original_size
            )

        assignments = assign_frames_to_gpus(len(frames), self.device_ids)
        results: List[Optional[np.ndarray]] = [None] * len(frames)
        lock = threading.Lock()

        def _worker(gpu_id: int, indices: List[int]) -> None:
            if not indices:
                return
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                torch.cuda.synchronize(gpu_id)
            subset_bgr = [
                cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR) for i in indices
            ]
            enhanced = self._enhance_bgr_batch(subset_bgr, gpu_id)
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)
            rgb_out = self._bgr_list_to_rgb(
                enhanced,
                [frames[i] for i in indices],
                maintain_dimensions,
                original_size,
            )
            with lock:
                for idx, rgb in zip(indices, rgb_out):
                    results[idx] = rgb

        with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
            futures = [
                executor.submit(_worker, dev_id, assignments.get(dev_id, []))
                for dev_id in self.device_ids
            ]
            for fut in futures:
                fut.result()

        return [results[i] if results[i] is not None else frames[i] for i in range(len(frames))]

    def enhance_rgb_image(self, frame: np.ndarray) -> np.ndarray:
        return self.enhance_rgb_frames([frame])[0]

    def enhance_chunk_buffer(
        self,
        buffer: "ChunkFrameBuffer",
        maintain_dimensions: bool = False,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> "ChunkFrameBuffer":
        """Enhance frames resident on GPU; avoids pre-enhance host round-trip."""
        from core.gpu_frame import ChunkFrameBuffer

        if not isinstance(buffer, ChunkFrameBuffer) or not buffer.has_gpu_batch():
            return buffer

        if self.enhancement_model == "HAT":
            self._enhance_hat_gpu_buffer(buffer, maintain_dimensions, original_size)
            return buffer

        frames = buffer.download_all()
        enhanced = self.enhance_rgb_frames(
            frames, maintain_dimensions=maintain_dimensions, original_size=original_size
        )
        buffer.replace_from_numpy(enhanced)
        try:
            buffer.upload()
        except Exception as exc:
            logger.debug("Chunk re-upload after numpy enhancement failed: %s", exc)
        return buffer

    def _enhance_hat_gpu_buffer(
        self,
        buffer: "ChunkFrameBuffer",
        maintain_dimensions: bool,
        original_size: Optional[Tuple[int, int]],
    ) -> None:
        import torch.nn.functional as F
        from processing.hat_enhancement import (
            HAT_UPSCALE,
            _apply_hat_model,
            _get_hat_model,
            _hat_gpu_lock,
            _should_use_tiled_inference,
        )

        gpu_id = self.device_ids[0]
        batch = buffer.upload()
        model, device, mean, img_range = _get_hat_model(
            self._resolve_hat_model(), gpu_id, None
        )

        with _hat_gpu_lock(gpu_id):
            for i in range(batch.shape[0]):
                frame_gpu = batch[i]
                h, w = frame_gpu.shape[0], frame_gpu.shape[1]
                try:
                    if _should_use_tiled_inference(
                        h, w, self.tile_size, gpu_id, force_tiled=False
                    ):
                        rgb_np = frame_gpu.detach().cpu().numpy()
                        enhanced_rgb = _apply_hat_model(
                            rgb_np,
                            model,
                            device,
                            mean,
                            img_range,
                            gpu_id,
                            self.tile_size,
                        )
                        out = torch.from_numpy(enhanced_rgb).to(frame_gpu.device)
                    else:
                        img = (
                            frame_gpu.permute(2, 0, 1).float().unsqueeze(0) / 255.0
                        ).to(device)
                        with torch.no_grad():
                            output = model(img.float())
                        out = (
                            output.squeeze(0)
                            .float()
                            .permute(1, 2, 0)
                            .clamp(0, 1)
                            * 255.0
                        )
                        out = out[: h * HAT_UPSCALE, : w * HAT_UPSCALE]
                        out = out.to(torch.uint8)

                    if maintain_dimensions and original_size:
                        ow, oh = original_size
                        if (out.shape[1], out.shape[0]) != (ow, oh):
                            resized = (
                                F.interpolate(
                                    out.permute(2, 0, 1).unsqueeze(0).float(),
                                    size=(oh, ow),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                .squeeze(0)
                                .permute(1, 2, 0)
                                .clamp(0, 255)
                                .to(torch.uint8)
                            )
                            out = resized
                    buffer.update_frame_gpu(i, out)
                except Exception as exc:
                    logger.warning("HAT GPU-chain failed for frame %d: %s", i, exc)

    @staticmethod
    def _bgr_list_to_rgb(
        bgr_list: List[np.ndarray],
        originals: List[np.ndarray],
        maintain_dimensions: bool,
        original_size: Optional[Tuple[int, int]],
    ) -> List[np.ndarray]:
        rgb_out = []
        for bgr, original in zip(bgr_list, originals):
            if bgr is None:
                rgb_out.append(original)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if maintain_dimensions and original_size and rgb.shape[:2] != (
                original_size[1],
                original_size[0],
            ):
                ow, oh = original_size
                rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
            rgb_out.append(rgb)
        return rgb_out