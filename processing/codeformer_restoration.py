"""
Face restoration using CodeFormer.

This module provides CodeFormer integration as an alternative to GFPGAN
for high-quality face restoration with controllable fidelity.
"""
import cv2
import gc
import logging
import numpy as np
import os
import torch
from pathlib import Path
from typing import List, Optional, Union

from utils.lru_cache import LRUModelCache

logger = logging.getLogger("FaceOff")


def _cleanup_codeformer(restorer):
    """Cleanup function for evicted CodeFormer instances."""
    try:
        if restorer is not None:
            restorer.cleanup()
    except Exception:
        pass


# LRU cache for CodeFormer instances (bounded to prevent memory growth)
_codeformer_cache = LRUModelCache("CodeFormer", cleanup_fn=_cleanup_codeformer)

# Model download URLs
CODEFORMER_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
DETECTION_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth"
PARSING_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth"


def _download_model(url: str, model_dir: Path, filename: str) -> Path:
    """Download model file if not exists."""
    model_path = model_dir / filename
    if model_path.exists():
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CodeFormer model: %s", filename)

    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(model_path))
        logger.info("Model downloaded: %s", model_path)
    except Exception as e:
        logger.error("Failed to download model %s: %s", filename, e)
        raise

    return model_path


class CodeFormerRestorer:
    """
    CodeFormer-based face restoration.

    Provides high-quality face restoration with adjustable fidelity parameter
    that controls the balance between quality and identity preservation.
    """

    def __init__(
        self,
        device_id: int = 0,
        upscale: int = 2,
        model_dir: Optional[str] = None
    ):
        """
        Initialize CodeFormer restorer.

        Args:
            device_id: GPU device ID
            upscale: Upscaling factor for output
            model_dir: Directory to store model weights
        """
        self.device_id = device_id
        self.upscale = upscale
        self.model_dir = Path(model_dir) if model_dir else Path("models/CodeFormer")
        self._restorer = None
        self._face_helper = None
        self._initialized = False
        self._device = None

    def _lazy_init(self) -> None:
        """Lazy initialization of CodeFormer model."""
        if self._initialized:
            return

        try:
            # Import CodeFormer dependencies
            from basicsr.utils import img2tensor, tensor2img
            from torchvision.transforms.functional import normalize

            # Import face restoration helper from facexlib (pip install facexlib)
            # Note: The package is 'facexlib' but imports as 'facexlib.utils...'
            try:
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            except ImportError:
                # Try alternative import paths
                try:
                    from facelib.utils.face_restoration_helper import FaceRestoreHelper
                except ImportError:
                    from basicsr.utils.face_restoration_helper import FaceRestoreHelper

            # Import CodeFormer architecture - try multiple approaches
            CodeFormerArch = None

            # Approach 1: Try direct import from codeformer package
            try:
                from codeformer.basicsr.archs.codeformer_arch import CodeFormer as CodeFormerArch
                logger.debug("Loaded CodeFormer from codeformer package")
            except ImportError:
                pass

            # Approach 2: Try from basicsr archs (if CodeFormer installed there)
            if CodeFormerArch is None:
                try:
                    from basicsr.archs.codeformer_arch import CodeFormer as CodeFormerArch
                    logger.debug("Loaded CodeFormer from basicsr.archs")
                except ImportError:
                    pass

            # Approach 3: Try from registry with alternative names
            if CodeFormerArch is None:
                try:
                    from basicsr.utils.registry import ARCH_REGISTRY
                    # Try different registry names
                    for name in ['CodeFormer', 'CodeFormer_basicsr', 'codeformer']:
                        try:
                            CodeFormerArch = ARCH_REGISTRY.get(name)
                            logger.debug("Loaded CodeFormer from registry as '%s'", name)
                            break
                        except KeyError:
                            continue
                except ImportError:
                    pass

            if CodeFormerArch is None:
                raise ImportError(
                    "CodeFormer architecture not found. Please install CodeFormer: "
                    "pip install codeformer-pytorch or clone from https://github.com/sczhou/CodeFormer"
                )

            self._device = torch.device(
                f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu'
            )

            # Download model if needed
            model_path = _download_model(
                CODEFORMER_MODEL_URL,
                self.model_dir,
                "codeformer.pth"
            )

            # Load CodeFormer model
            self._restorer = CodeFormerArch(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self._device)

            checkpoint = torch.load(str(model_path), map_location=self._device)
            self._restorer.load_state_dict(
                checkpoint['params_ema'] if 'params_ema' in checkpoint else checkpoint,
                strict=True
            )
            self._restorer.eval()

            # Initialize face helper
            self._face_helper = FaceRestoreHelper(
                upscale_factor=self.upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self._device
            )

            self._initialized = True
            logger.info("CodeFormer initialized on device %d", self.device_id)

        except ImportError as e:
            logger.error(
                "CodeFormer dependencies not found. Install with: "
                "pip install basicsr facexlib. Error: %s", e
            )
            raise
        except Exception as e:
            logger.error("Failed to initialize CodeFormer: %s", e, exc_info=True)
            raise

    def restore_face(
        self,
        image: np.ndarray,
        fidelity_weight: float = 0.5,
        only_center_face: bool = False,
        paste_back: bool = True
    ) -> np.ndarray:
        """
        Restore faces in an image using CodeFormer.

        Args:
            image: Input image (BGR format, numpy array)
            fidelity_weight: Balance between quality (0) and identity (1).
                            Lower = better quality, higher = better fidelity.
                            Recommended: 0.5 for face swapping.
            only_center_face: Only restore the center/largest face
            paste_back: Paste restored faces back to original image

        Returns:
            Restored image (BGR format)
        """
        self._lazy_init()

        if image is None or image.size == 0:
            logger.warning("Empty image provided to CodeFormer")
            return image

        try:
            from basicsr.utils import img2tensor, tensor2img
            from torchvision.transforms.functional import normalize

            # Prepare face helper
            self._face_helper.clean_all()
            self._face_helper.read_image(image)

            # Detect faces
            num_det_faces = self._face_helper.get_face_landmarks_5(
                only_center_face=only_center_face,
                resize=640,
                eye_dist_threshold=5
            )

            if num_det_faces == 0:
                logger.debug("No faces detected by CodeFormer, returning original")
                return image

            logger.debug("CodeFormer detected %d face(s)", num_det_faces)

            # Align and warp faces
            self._face_helper.align_warp_face()

            # Restore each face
            for idx, cropped_face in enumerate(self._face_helper.cropped_faces):
                # Prepare tensor
                cropped_face_t = img2tensor(
                    cropped_face / 255.,
                    bgr2rgb=True,
                    float32=True
                )
                normalize(
                    cropped_face_t,
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                    inplace=True
                )
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self._device)

                # CodeFormer inference
                with torch.no_grad():
                    output = self._restorer(
                        cropped_face_t,
                        w=fidelity_weight,
                        adain=True
                    )[0]

                    # Convert back to image
                    restored_face = tensor2img(
                        output,
                        rgb2bgr=True,
                        min_max=(-1, 1)
                    )

                restored_face = restored_face.astype('uint8')
                self._face_helper.add_restored_face(restored_face)

            # Paste faces back
            if paste_back:
                self._face_helper.get_inverse_affine(None)

                # Check if background upsampling is needed
                if self.upscale > 1:
                    # Use simple resize for background
                    bg_img = cv2.resize(
                        image,
                        (image.shape[1] * self.upscale, image.shape[0] * self.upscale),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                else:
                    bg_img = None

                restored_img = self._face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img
                )
            else:
                restored_img = self._face_helper.cropped_faces[0]

            return restored_img

        except Exception as e:
            logger.error("CodeFormer restoration failed: %s", e, exc_info=True)
            return image

    def restore_faces_in_frame(
        self,
        frame: np.ndarray,
        fidelity_weight: float = 0.5
    ) -> np.ndarray:
        """
        Restore all faces in a video/GIF frame.

        Args:
            frame: Input frame (BGR format)
            fidelity_weight: Fidelity weight (0-1)

        Returns:
            Frame with restored faces
        """
        return self.restore_face(
            frame,
            fidelity_weight=fidelity_weight,
            only_center_face=False,
            paste_back=True
        )

    def cleanup(self) -> None:
        """Release GPU resources."""
        self._restorer = None
        self._face_helper = None
        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("CodeFormer resources released")


def clear_codeformer_cache() -> None:
    """Clear the CodeFormer model cache."""
    _codeformer_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("CodeFormer cache cleared")


def _get_codeformer_restorer(device_id: int = 0, upscale: int = 1) -> CodeFormerRestorer:
    """
    Get or create a cached CodeFormer restorer instance.

    Args:
        device_id: GPU device ID
        upscale: Output upscaling factor

    Returns:
        CodeFormerRestorer instance
    """
    cache_key = (device_id, upscale)

    cached = _codeformer_cache.get(cache_key)
    if cached is not None:
        return cached

    logger.info("Creating CodeFormer restorer on GPU %d with upscale=%d", device_id, upscale)
    restorer = CodeFormerRestorer(device_id=device_id, upscale=upscale)
    _codeformer_cache.put(cache_key, restorer)

    return restorer


def restore_faces_codeformer(
    image: np.ndarray,
    device_id: int = 0,
    fidelity_weight: float = 0.5,
    upscale: int = 1
) -> np.ndarray:
    """
    Convenience function for CodeFormer face restoration.

    Uses cached restorer instance for efficiency.

    Args:
        image: Input image (BGR format)
        device_id: GPU device ID
        fidelity_weight: Fidelity weight (0=quality, 1=identity)
        upscale: Output upscaling factor

    Returns:
        Restored image (BGR format)
    """
    restorer = _get_codeformer_restorer(device_id=device_id, upscale=upscale)
    return restorer.restore_face(image, fidelity_weight=fidelity_weight)


def restore_frames_codeformer(
    frames: List[np.ndarray],
    device_id: int = 0,
    fidelity_weight: float = 0.5
) -> List[np.ndarray]:
    """
    Restore faces in multiple frames using CodeFormer.

    Uses cached restorer instance for efficiency.

    Args:
        frames: List of frames (BGR format)
        device_id: GPU device ID
        fidelity_weight: Fidelity weight (0-1)

    Returns:
        List of restored frames
    """
    from tqdm import tqdm

    restorer = _get_codeformer_restorer(device_id=device_id, upscale=1)

    restored_frames = []
    for frame in tqdm(frames, desc="CodeFormer Restoration", unit="frame"):
        restored = restorer.restore_faces_in_frame(frame, fidelity_weight)
        restored_frames.append(restored)
    return restored_frames
