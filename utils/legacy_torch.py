"""Scoped ``torch.load`` compatibility for legacy GPU-vision packages.

PyTorch 2.6+ defaults ``weights_only=True``, but ``basicsr`` 1.4.2 (pulled
in by gfpgan/realesrgan) calls ``torch.load`` without the parameter and
fails to deserialize its weights. Instead of relaxing deserialization
process-wide, this shim applies ``weights_only=False`` only when the call
originates from a known legacy package; every other caller keeps the
secure default.
"""

from __future__ import annotations

import logging
from types import FrameType
from typing import Any, Optional

logger = logging.getLogger("FaceOff")

# Packages that predate weights_only and load trusted upstream weights.
LEGACY_PACKAGES = ("basicsr", "gfpgan", "facexlib", "realesrgan", "codeformer")

_original_load: Optional[Any] = None


def module_needs_legacy_load(module_name: str) -> bool:
    """True when *module_name* belongs to a legacy package needing full pickle."""
    return module_name.split(".")[0] in LEGACY_PACKAGES


def _legacy_caller_on_stack(frame: Optional[FrameType]) -> bool:
    """Walk outward from the ``torch.load`` caller for a legacy package."""
    seen = 0
    while frame is not None and seen < 32:
        name = frame.f_globals.get("__name__", "")
        if isinstance(name, str) and module_needs_legacy_load(name):
            return True
        frame = frame.f_back
        seen += 1
    return False


def install_legacy_torch_load_shim() -> None:
    """Patch ``torch.load`` to relax weights_only for legacy callers only."""
    global _original_load
    import torch

    if _original_load is not None:
        return
    _original_load = torch.load

    def _scoped_load(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs:
            import inspect

            caller = inspect.currentframe()
            try:
                # currentframe -> _scoped_load -> real caller
                real_caller = caller.f_back if caller is not None else None
                if _legacy_caller_on_stack(real_caller):
                    kwargs["weights_only"] = False
            finally:
                del caller
        return _original_load(*args, **kwargs)

    torch.load = _scoped_load
    logger.debug("Scoped torch.load shim installed for %s", LEGACY_PACKAGES)
