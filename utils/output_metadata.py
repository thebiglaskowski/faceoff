"""
Persist and display processing settings alongside gallery outputs.

Each output file may have a sibling ``<filename>.meta.json`` sidecar that
records the options used to produce it for quality comparison in the gallery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("FaceOff")

METADATA_VERSION = 1
METADATA_SUFFIX = ".meta.json"


def metadata_path_for(output_path: str | Path) -> Path:
    """Return the sidecar metadata path for an output media file."""
    path = Path(output_path)
    return path.with_name(f"{path.name}{METADATA_SUFFIX}")


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    return str(value)


def build_settings_from_options(opts: Any) -> Dict[str, Any]:
    """Extract gallery-relevant fields from ProcessOptions (or similar)."""
    if is_dataclass(opts):
        raw = asdict(opts)
        raw.pop("source_image", None)
        return {k: _serialize_value(v) for k, v in raw.items()}
    if isinstance(opts, dict):
        data = dict(opts)
        data.pop("source_image", None)
        return {k: _serialize_value(v) for k, v in data.items()}
    raise TypeError(f"Unsupported options type: {type(opts)!r}")


def save_output_metadata(output_path: str | Path, opts: Any) -> Optional[Path]:
    """
    Write processing settings next to an output file.

    Returns:
        Path to the metadata file, or None if writing failed.
    """
    out = Path(output_path)
    if not out.exists():
        logger.warning("Cannot save metadata — output file missing: %s", out)
        return None

    payload = {
        "version": METADATA_VERSION,
        "output_file": out.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": build_settings_from_options(opts),
    }

    meta_path = metadata_path_for(out)
    try:
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("Saved output metadata: %s", meta_path)
        return meta_path
    except OSError as exc:
        logger.warning("Failed to save output metadata for %s: %s", out, exc)
        return None


def load_output_metadata(output_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load sidecar metadata for an output file, if present."""
    meta_path = metadata_path_for(output_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read metadata %s: %s", meta_path, exc)
        return None


def format_settings_summary(settings: Dict[str, Any]) -> str:
    """One-line summary for gallery captions."""
    if not settings:
        return "Settings: unknown (processed before metadata tracking)"

    parts: list[str] = []

    if settings.get("enhance"):
        model = settings.get("model_display") or settings.get("model_name", "on")
        framework = settings.get("enhancement_model", "RealESRGAN")
        scale = settings.get("outscale", 4)
        parts.append(f"{framework} {model} {scale}x")
    else:
        parts.append("Enhance: off")

    if settings.get("restore_faces"):
        rest = settings.get("restoration_model", "GFPGAN")
        weight = settings.get("restoration_weight", 0.5)
        parts.append(f"{rest} {weight:.0%}")
    else:
        parts.append("Restore: off")

    gpu = settings.get("gpu_selection") or "GPU 0"
    if isinstance(gpu, str) and len(gpu) > 28:
        gpu = gpu.split(":")[0].strip()
    parts.append(str(gpu))

    mappings = settings.get("face_mappings")
    if mappings:
        parts.append(f"{len(mappings)} mapping(s)")

    return " · ".join(parts)


def format_settings_detail(
    metadata: Optional[Dict[str, Any]],
    file_path: Optional[str] = None,
) -> str:
    """Multi-line markdown block for the gallery detail panel."""
    if not metadata:
        name = Path(file_path).name if file_path else "this file"
        return (
            f"**Processing settings**\n\n"
            f"No metadata found for `{name}`.\n\n"
            "_Files created before this feature was added won't have saved settings._"
        )

    settings = metadata.get("settings", {})
    created = metadata.get("created_at", "")
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            created = created_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            pass

    lines = ["**Processing settings**", ""]
    if created:
        lines.append(f"- **Created:** {created}")
    lines.append(f"- **Media type:** {settings.get('media_type', '—')}")

    lines.append(f"- **Enhancement:** {'on' if settings.get('enhance') else 'off'}")
    if settings.get("enhance"):
        lines.append(
            f"  - Framework: {settings.get('enhancement_model', '—')}"
        )
        display = settings.get("model_display") or settings.get("model_name", "—")
        lines.append(f"  - Model: {display}")
        lines.append(f"  - Upscale: {settings.get('outscale', '—')}x")
        lines.append(f"  - Tile size: {settings.get('tile_size', '—')}")
        lines.append(f"  - Denoise: {settings.get('denoise_strength', '—')}")
        lines.append(f"  - FP32: {settings.get('use_fp32', False)}")
        lines.append(f"  - Pre-pad: {settings.get('pre_pad', 0)}")

    lines.append(
        f"- **Face restoration:** {'on' if settings.get('restore_faces') else 'off'}"
    )
    if settings.get("restore_faces"):
        lines.append(f"  - Model: {settings.get('restoration_model', '—')}")
        lines.append(f"  - Weight: {settings.get('restoration_weight', '—')}")

    lines.append(f"- **Face confidence:** {settings.get('face_confidence', '—')}")
    lines.append(f"- **GPU:** {settings.get('gpu_selection') or 'default'}")
    lines.append(f"- **TensorRT FP16:** {settings.get('tensorrt_fp16', '—')}")

    mappings = settings.get("face_mappings")
    if mappings:
        lines.append(f"- **Face mappings:** {mappings}")
    else:
        lines.append("- **Face mappings:** auto (first source → all targets)")

    return "\n".join(lines)


def build_gallery_caption(file_path: Path, mod_time: datetime) -> str:
    """Build a three-line gallery caption: name, timestamp, settings summary."""
    time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")
    metadata = load_output_metadata(file_path)
    if metadata:
        summary = format_settings_summary(metadata.get("settings", {}))
    else:
        summary = "Settings: not recorded"
    return f"{file_path.name}\n{time_str}\n{summary}"