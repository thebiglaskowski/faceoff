"""
Configuration management for FaceOff application.

This module handles loading and accessing configuration from config.yaml,
with fallbacks to default values if config is missing or invalid.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager with fallback to defaults."""

    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: Path = (
        Path(__file__).parent.parent / "config.yaml"
    )  # Root directory, not utils/

    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if self._config_path.exists():
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Configuration loaded from {self._config_path}")
            else:
                logger.warning(
                    f"Config file not found: {self._config_path}. Using defaults."
                )
                self._config = {}
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            self._config = {}

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            *keys: Keys to traverse the config hierarchy
            default: Default value if key not found

        Example:
            config.get('gpu', 'batch_size', default=4)
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    def update(self, *keys: str, value: Any) -> None:
        """
        Update a configuration value at runtime.

        Args:
            *keys: Keys to traverse the config hierarchy
            value: New value to set

        Example:
            config.update('gpu', 'tensorrt_fp16', value=True)

        Note: This updates the in-memory config only, not the file.
        """
        if not keys:
            return

        # Navigate to the parent dict
        current = self._config
        if not isinstance(current, dict):
            raise ValueError("Config root is not a mapping; cannot update")
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            if not isinstance(current, dict):
                raise ValueError(
                    f"Cannot update '{'.'.join(keys)}': '{key}' is a value, "
                    "not a section"
                )

        # Set the value
        current[keys[-1]] = value
        logger.debug(f"Config updated: {'.'.join(keys)} = {value}")

    def write_values(
        self,
        scalars: Optional[Dict[tuple, Any]] = None,
        pairs: Optional[Dict[tuple, tuple]] = None,
        lists: Optional[Dict[tuple, list]] = None,
    ) -> None:
        """Persist leaf values to config.yaml, preserving comments.

        Scalar lines are replaced in place (indentation and trailing
        comments survive); the two non-scalar shapes the editor supports
        — a ``[w, h]`` flow pair and 1-D string lists — get targeted
        rewrites. The result is re-parsed and every change is verified
        before the file is atomically replaced; any failure leaves the
        original file untouched.

        Raises:
            ValueError: On unknown paths, type mismatches, or a failed
                verification.
        """
        scalars = scalars or {}
        pairs = pairs or {}
        lists = lists or {}
        if not (scalars or pairs or lists):
            return
        original = self._config_path.read_text(encoding="utf-8")
        current_doc = yaml.safe_load(original) or {}
        _validate_changes(current_doc, scalars, pairs, lists)
        updated = _apply_to_text(original, scalars, pairs, lists)
        _verify_changes(updated, scalars, pairs, lists)
        tmp_path = self._config_path.with_suffix(".yaml.tmp")
        tmp_path.write_text(updated, encoding="utf-8")
        os.replace(tmp_path, self._config_path)
        self._load_config()
        logger.info(
            "Config saved: %d value(s) written to %s",
            len(scalars) + len(pairs) + len(lists),
            self._config_path,
        )

    # =============================================================================
    # Convenience properties for commonly accessed values
    # =============================================================================

    # File limits
    @property
    def max_file_size_mb(self) -> int:
        return self.get("limits", "max_file_size_mb", default=500)

    @property
    def max_video_duration_sec(self) -> int:
        return self.get("limits", "max_video_duration_sec", default=300)

    @property
    def max_image_pixels(self) -> int:
        return self.get("limits", "max_image_pixels", default=16777216)

    @property
    def max_gif_frames(self) -> int:
        return self.get("limits", "max_gif_frames", default=500)

    # GPU settings
    @property
    def batch_size(self) -> int:
        return self.get("gpu", "batch_size", default=4)

    @property
    def max_batch_size(self) -> int:
        return self.get("gpu", "max_batch_size", default=16)

    @property
    def workers_per_gpu(self) -> int:
        return self.get("gpu", "workers_per_gpu", default=4)

    @property
    def multi_gpu_video_enabled(self) -> bool:
        return self.get("gpu", "multi_gpu_video_enabled", default=True)

    @property
    def tensorrt_enabled(self) -> bool:
        return self.get("gpu", "tensorrt_enabled", default=True)

    @property
    def gpu_frame_retention_enabled(self) -> bool:
        return self.get("gpu", "frame_retention_enabled", default=False)

    @property
    def gpu_paste_on_gpu(self) -> bool:
        return self.get("gpu", "paste_on_gpu", default=False)

    @property
    def gpu_detection_on_gpu(self) -> bool:
        return self.get("gpu", "detection_on_gpu", default=False)

    @property
    def gpu_enhancement_chain_enabled(self) -> bool:
        return self.get("gpu", "enhancement_chain_enabled", default=False)

    @property
    def gpu_auto_workload_tune(self) -> bool:
        return self.get("gpu", "auto_workload_tune", default=True)

    @property
    def tensorrt_fp16(self) -> bool:
        return self.get("gpu", "tensorrt_fp16", default=True)

    @property
    def tensorrt_workspace_mb(self) -> int:
        return self.get("gpu", "tensorrt_workspace_mb", default=2048)

    # Face detection
    @property
    def inswapper_model_path(self) -> str:
        return self.get(
            "face_detection", "inswapper_model_path", default="inswapper_128.onnx"
        )

    @property
    def buffalo_model_path(self) -> str:
        return self.get(
            "face_detection", "buffalo_model_path", default="models/buffalo_l"
        )

    @property
    def face_analysis_name(self) -> str:
        return self.get("face_detection", "face_analysis_name", default="buffalo_l")

    @property
    def face_analysis_det_size(self) -> list:
        return self.get("face_detection", "face_analysis_det_size", default=[640, 640])

    @property
    def face_confidence_threshold(self) -> float:
        return self.get("face_detection", "confidence_threshold", default=0.5)

    @property
    def iou_threshold(self) -> float:
        return self.get("face_detection", "iou_threshold", default=0.3)

    @property
    def embedding_tracking_enabled(self) -> bool:
        return self.get("face_detection", "embedding_tracking_enabled", default=True)

    @property
    def embedding_similarity_threshold(self) -> float:
        return self.get(
            "face_detection", "embedding_similarity_threshold", default=0.35
        )

    @property
    def embedding_ema_alpha(self) -> float:
        return self.get("face_detection", "embedding_ema_alpha", default=0.25)

    @property
    def ui_padding(self) -> int:
        return self.get("face_detection", "ui_padding", default=20)

    @property
    def adaptive_detection_enabled(self) -> bool:
        return self.get("face_detection", "adaptive_enabled", default=True)

    @property
    def detection_scale(self) -> float:
        return self.get("face_detection", "detection_scale", default=0.5)

    @property
    def min_detection_resolution(self) -> int:
        return self.get("face_detection", "min_resolution", default=640)

    # Enhancement
    @property
    def default_enhancement_model(self) -> str:
        return self.get("enhancement", "default_model", default="RealESRGAN_x4plus")

    @property
    def enhancement_models(self) -> Dict[str, Dict[str, Any]]:
        return self.get("enhancement", "models", default={})

    @property
    def default_tile_size(self) -> int:
        return self.get("enhancement", "defaults", "tile_size", default=256)

    @property
    def default_outscale(self) -> int:
        return self.get("enhancement", "defaults", "outscale", default=4)

    @property
    def default_pre_pad(self) -> int:
        return self.get("enhancement", "defaults", "pre_pad", default=0)

    @property
    def default_use_fp32(self) -> bool:
        return self.get("enhancement", "defaults", "use_fp32", default=False)

    @property
    def default_denoise_strength(self) -> float:
        return self.get("enhancement", "defaults", "denoise_strength", default=0.5)

    @property
    def enhancement_multi_gpu_enabled(self) -> bool:
        return self.get("enhancement", "multi_gpu_enabled", default=False)

    # Face restoration
    @property
    def gfpgan_enabled_by_default(self) -> bool:
        return self.get("face_restoration", "enabled_by_default", default=False)

    @property
    def gfpgan_model_version(self) -> str:
        return self.get("face_restoration", "model_version", default="1.3")

    @property
    def gfpgan_default_weight(self) -> float:
        return self.get("face_restoration", "default_weight", default=0.5)

    # Streaming pipeline
    @property
    def streaming_enabled(self) -> bool:
        return self.get("streaming", "enabled", default=True)

    @property
    def streaming_chunk_size(self) -> int:
        return self.get("streaming", "chunk_size", default=32)

    @property
    def streaming_video_face_enhance(self) -> bool:
        return self.get("streaming", "video_face_enhance", default=False)

    @property
    def streaming_gif_decode_fps(self) -> float:
        return float(self.get("streaming", "gif_decode_fps", default=10.0))

    @property
    def streaming_hwaccel_decode(self) -> bool:
        return self.get("streaming", "hwaccel_decode", default=True)

    @property
    def streaming_zero_copy_enabled(self) -> bool:
        return self.get("streaming", "zero_copy_enabled", default=False)

    @property
    def streaming_nvcodec_decode(self) -> bool:
        return self.get("streaming", "nvcodec_decode", default=True)

    @property
    def streaming_nvenc_encode(self) -> bool:
        return self.get("streaming", "nvenc_encode", default=True)

    @property
    def streaming_video_codec(self) -> str:
        return self.get("streaming", "video_codec", default="libx264")

    @property
    def streaming_video_preset(self) -> str:
        return self.get("streaming", "video_preset", default="medium")

    @property
    def streaming_video_crf(self) -> int:
        return self.get("streaming", "video_crf", default=18)

    @property
    def compression_skip_optimized_video(self) -> bool:
        return self.get("compression", "skip_optimized_video", default=True)

    @property
    def preload_enhancement_on_startup(self) -> bool:
        return self.get("model_cache", "preload_enhancement_on_startup", default=False)

    # Logging
    @property
    def log_file(self) -> str:
        return self.get("logging", "log_file", default="app.log")

    @property
    def log_max_file_size_mb(self) -> int:
        return self.get("logging", "max_file_size_mb", default=10)

    @property
    def log_backup_count(self) -> int:
        return self.get("logging", "backup_count", default=5)

    @property
    def log_console_level(self) -> str:
        return self.get("logging", "console_level", default="INFO")

    @property
    def log_file_level(self) -> str:
        return self.get("logging", "file_level", default="DEBUG")

    @property
    def log_format(self) -> str:
        return self.get(
            "logging",
            "format",
            default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @property
    def log_date_format(self) -> str:
        return self.get("logging", "date_format", default="%Y-%m-%d %H:%M:%S")

    @property
    def log_json_format(self) -> bool:
        return self.get("logging", "json_format", default=False)

    @property
    def log_json_file(self) -> str:
        return self.get("logging", "json_log_file", default="app.json.log")

    @property
    def log_terminal_buffer_lines(self) -> int:
        return self.get("logging", "terminal_buffer_lines", default=3000)

    @property
    def log_terminal_auto_refresh_sec(self) -> float:
        return float(self.get("logging", "terminal_auto_refresh_sec", default=2))

    # Model cache
    @property
    def tensorrt_cache_dir(self) -> str:
        return self.get("model_cache", "tensorrt_cache_dir", default="cache/tensorrt")

    @property
    def tensorrt_cache_enabled(self) -> bool:
        return self.get("model_cache", "tensorrt_cache_enabled", default=True)

    @property
    def tensorrt_timing_cache_enabled(self) -> bool:
        return self.get("model_cache", "tensorrt_timing_cache_enabled", default=True)

    @property
    def preload_on_startup(self) -> bool:
        return self.get("model_cache", "preload_on_startup", default=False)

    @property
    def preload_models(self) -> list:
        return self.get(
            "model_cache", "preload_models", default=["buffalo_l", "inswapper_128"]
        )

    # Memory management
    @property
    def auto_clear_cache(self) -> bool:
        return self.get("memory", "auto_clear_cache", default=True)

    @property
    def clear_cache_threshold_mb(self) -> int:
        return self.get("memory", "clear_cache_threshold_mb", default=1024)

    @property
    def reduce_batch_on_oom(self) -> bool:
        return self.get("memory", "reduce_batch_on_oom", default=True)

    @property
    def mb_per_batch_estimate(self) -> int:
        return self.get("memory", "mb_per_batch_estimate", default=500)

    @property
    def min_batch_size(self) -> int:
        return self.get("memory", "min_batch_size", default=1)

    @property
    def release_swap_models_before_enhance(self) -> bool:
        return self.get("memory", "release_swap_models_before_enhance", default=True)

    @property
    def hat_force_tiled_below_free_mb(self) -> int:
        return self.get("memory", "hat_force_tiled_below_free_mb", default=2048)

    @property
    def hat_oom_min_tile_size(self) -> int:
        return self.get("memory", "hat_oom_min_tile_size", default=64)

    # File formats
    @property
    def supported_image_formats(self) -> list:
        return self.get(
            "file_formats", "images", default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        )

    @property
    def supported_video_formats(self) -> list:
        return self.get(
            "file_formats", "videos", default=[".mp4", ".webp", ".avi", ".mov"]
        )

    @property
    def supported_gif_formats(self) -> list:
        return self.get("file_formats", "gifs", default=[".gif"])

    # Directories
    @property
    def temp_gif_frames_dir(self) -> str:
        return self.get("directories", "temp_gif_frames", default="temp_gif_frames")

    @property
    def temp_gif_enhanced_dir(self) -> str:
        return self.get("directories", "temp_gif_enhanced", default="temp_gif_enhanced")

    @property
    def output_dir(self) -> str:
        return self.get("directories", "output", default="outputs")

    @property
    def models_dir(self) -> str:
        return self.get("directories", "models", default="models")

    @property
    def cache_dir(self) -> str:
        return self.get("directories", "cache", default="cache")

    # UI settings
    @property
    def ui_server_name(self) -> str:
        return self.get("ui", "server_name", default="127.0.0.1")

    @property
    def server_name(self) -> str:
        """Alias for ui_server_name."""
        return self.ui_server_name

    @property
    def ui_server_port(self) -> int:
        return self.get("ui", "server_port", default=7860)

    @property
    def server_port(self) -> int:
        """Alias for ui_server_port."""
        return self.ui_server_port

    @property
    def ui_share(self) -> bool:
        return self.get("ui", "share", default=False)

    @property
    def share(self) -> bool:
        """Alias for ui_share."""
        return self.ui_share

    @property
    def ui_theme(self) -> str:
        """UI theme name — "soft" or "neon". See ui/faceoff_theme.py."""
        return self.get("ui", "theme", default="soft")

    @property
    def theme(self) -> str:
        """Alias for ui_theme."""
        return self.ui_theme


# =============================================================================
# Comment-preserving YAML surgery for the Settings editor
# =============================================================================

_MAPPING_LINE = re.compile(r"^(\s*)([^#\s][^:]*):(.*)$")


def _split_comment(text: str) -> Tuple[str, str]:
    """Split a YAML line into (code, comment), honoring quotes."""
    in_single = False
    in_double = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return text[:idx], text[idx:]
    return text, ""


def _section_stack(lines: List[str], upto: int) -> List[Tuple[int, str]]:
    """Mapping-key stack (indent, key) describing the context of a line."""
    stack: List[Tuple[int, str]] = []
    for line in lines[:upto]:
        match = _MAPPING_LINE.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        key = match.group(2).strip()
        code, _comment = _split_comment(match.group(3))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        # A line with an empty value opens a nested block; a scalar or
        # flow value is a leaf (pushed so its own line resolves, then
        # popped by the next same-indent key). Block-style scalars
        # (``key:`` with the value on the next line) resolve at query
        # time, where the seeking line dedents out of them.
        stack.append((indent, key))
        if code.strip():
            stack.pop()
    return stack


def _find_key_line(lines: List[str], path: Tuple[str, ...]) -> int:
    """Return the index of the mapping line starting ``path``."""
    for idx, line in enumerate(lines):
        match = _MAPPING_LINE.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        stack = _section_stack(lines, idx)
        # The line itself dedents out of any deeper block above it.
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full = [key for _, key in stack] + [match.group(2).strip()]
        if full == list(path):
            # Own indent must nest strictly deeper than its parent.
            if len(stack) == len(path) - 1 and (not stack or stack[-1][0] < indent):
                return idx
    raise ValueError(f"Unknown config path: {'.'.join(path)}")


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"{}"'.format(value.replace("\\", "\\\\").replace('"', '\\"'))
    raise ValueError(f"Unsupported config value type: {type(value).__name__}")


def _eol(line: str) -> str:
    return "\n" if line.endswith("\n") else ""


def _set_scalar_line(lines: List[str], idx: int, value: Any) -> None:
    """Replace the scalar on a ``key: value`` line, keeping its comment."""
    match = _MAPPING_LINE.match(lines[idx])
    assert match is not None
    eol = _eol(lines[idx])
    code, comment = _split_comment(match.group(3))
    if code.strip():
        gap = code[len(code.rstrip()) :]
        lines[idx] = (
            f"{match.group(1)}{match.group(2)}: "
            f"{_format_scalar(value)}{gap}{comment}{eol}"
        )
        return
    # Block style: value lives on the next deeper-indented scalar line.
    key_indent = len(match.group(1))
    for nxt in range(idx + 1, len(lines)):
        nline = lines[nxt]
        if not nline.strip() or nline.strip().startswith("#"):
            continue
        nmatch = _MAPPING_LINE.match(nline)
        nindent = len(nmatch.group(1)) if nmatch else key_indent + 1
        if nindent <= key_indent:
            break
        ncode, ncomment = _split_comment(nline)
        if ncode.strip().startswith("- ") or ncode.strip().endswith(":"):
            break
        pad = nline[: len(nline) - len(nline.lstrip())]
        gap = ncode[len(ncode.rstrip()) :]
        # nline carries its own newline inside ncomment; strip it so the
        # explicit EOL below does not double it into a blank line.
        ncomment = ncomment.rstrip("\n")
        lines[nxt] = f"{pad}{_format_scalar(value)}{gap}{ncomment}{_eol(nline)}"
        return
    raise ValueError(f"No scalar value found for: {lines[idx].strip()}")


def _set_flow_pair(lines: List[str], idx: int, pair: Tuple[Any, Any]) -> None:
    match = _MAPPING_LINE.match(lines[idx])
    assert match is not None
    code, comment = _split_comment(match.group(3))
    if not re.match(r"^\s*\[.*\]\s*$", code):
        raise ValueError(f"Not a [w, h] pair line: {lines[idx].strip()}")
    w, h = pair
    lines[idx] = (
        f"{match.group(1)}{match.group(2)}: [{w}, {h}]{comment}{_eol(lines[idx])}"
    )


def _set_str_list(lines: List[str], idx: int, items: List[str]) -> List[str]:
    match = _MAPPING_LINE.match(lines[idx])
    assert match is not None
    key_indent = len(match.group(1))
    end = idx + 1
    item_indent: Optional[str] = None
    while end < len(lines):
        stripped = lines[end].strip()
        if not stripped or stripped.startswith("#"):
            break
        indent = len(lines[end]) - len(lines[end].lstrip())
        if indent <= key_indent or not stripped.startswith("- "):
            break
        if item_indent is None:
            item_indent = lines[end][:indent]
        end += 1
    if item_indent is None:
        raise ValueError(f"No list items found for: {lines[idx].strip()}")
    new_block = [f"{item_indent}- {_format_scalar(v)}\n" for v in items]
    return lines[: idx + 1] + new_block + lines[end:]


def _lookup(doc: Any, path: Tuple[str, ...]) -> Tuple[bool, Any]:
    value = doc
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _validate_changes(
    doc: Dict[str, Any],
    scalars: Dict[tuple, Any],
    pairs: Dict[tuple, tuple],
    lists: Dict[tuple, list],
) -> None:
    for path, value in scalars.items():
        found, current = _lookup(doc, path)
        if not found:
            raise ValueError(f"Unknown config path: {'.'.join(path)}")
        if isinstance(current, bool) != isinstance(value, bool):
            raise ValueError(f"Type mismatch for {'.'.join(path)}: bool required")
        if isinstance(current, (int, float)) and not isinstance(value, (int, float)):
            raise ValueError(f"Type mismatch for {'.'.join(path)}: number required")
        if isinstance(current, str) and not isinstance(value, str):
            raise ValueError(f"Type mismatch for {'.'.join(path)}: text required")
        if isinstance(current, (dict, list)):
            raise ValueError(f"Not a scalar option: {'.'.join(path)}")
    for path, pair in pairs.items():
        found, current = _lookup(doc, path)
        if (
            not found
            or not isinstance(current, list)
            or len(current) != 2
            or not all(isinstance(v, (int, float)) for v in pair)
        ):
            raise ValueError(f"Not a [w, h] pair option: {'.'.join(path)}")
    for path, items in lists.items():
        found, current = _lookup(doc, path)
        if (
            not found
            or not isinstance(current, list)
            or not all(isinstance(v, str) for v in current)
            or not items
            or not all(isinstance(v, str) and v.strip() for v in items)
        ):
            raise ValueError(f"Not a text-list option: {'.'.join(path)}")


def _apply_to_text(
    text: str,
    scalars: Dict[tuple, Any],
    pairs: Dict[tuple, tuple],
    lists: Dict[tuple, list],
) -> str:
    lines = text.splitlines(keepends=True)
    for path, value in scalars.items():
        _set_scalar_line(lines, _find_key_line(lines, path), value)
    for path, pair in pairs.items():
        _set_flow_pair(lines, _find_key_line(lines, path), pair)
    # Rewrite list blocks last (they shift line indexes); longest path
    # first so nested blocks never invalidate a pending sibling index.
    for path in sorted(lists, key=len, reverse=True):
        lines = _set_str_list(lines, _find_key_line(lines, path), lists[path])
    return "".join(lines)


def _verify_changes(
    text: str,
    scalars: Dict[tuple, Any],
    pairs: Dict[tuple, tuple],
    lists: Dict[tuple, list],
) -> None:
    try:
        doc = yaml.safe_load(text) or {}
    except Exception as exc:
        raise ValueError(f"Saved config would not parse: {exc}") from exc

    def _number(value: Any) -> Any:
        return float(value) if isinstance(value, float) else value

    for path, value in scalars.items():
        found, current = _lookup(doc, path)
        if not found or _number(current) != _number(value):
            raise ValueError(f"Write verification failed: {'.'.join(path)}")
    for path, pair in pairs.items():
        found, current = _lookup(doc, path)
        if not found or list(current) != [_number(v) for v in pair]:
            raise ValueError(f"Write verification failed: {'.'.join(path)}")
    for path, items in lists.items():
        found, current = _lookup(doc, path)
        if not found or list(current) != list(items):
            raise ValueError(f"Write verification failed: {'.'.join(path)}")


# Global config instance
config = Config()


# Helper function for getting model options in UI format
def get_model_options() -> Dict[str, Dict[str, Any]]:
    """
    Get enhancement models in UI-compatible format.

    Returns:
        Dict mapping display names to model configurations
        Example: {"RealESRGAN_x4plus (General)": {"model_name": "RealESRGAN_x4plus", ...}}
    """
    models = config.enhancement_models
    if not models:
        # Fallback to defaults if config fails to load
        return {
            "RealESRGAN_x4plus (General - Best for Photos)": {
                "model_name": "RealESRGAN_x4plus",
                "supports_denoise": False,
                "description": "General purpose, best for photorealistic images",
            },
            "RealESRGAN_x4plus_anime_6B (Anime/Illustrations)": {
                "model_name": "RealESRGAN_x4plus_anime_6B",
                "supports_denoise": False,
                "description": "Optimized for anime and illustrated content",
            },
            "RealESRNet_x4plus (Conservative)": {
                "model_name": "RealESRNet_x4plus",
                "supports_denoise": False,
                "description": "More conservative, fewer artifacts",
            },
            "realesr-general-x4v3 (With Denoise)": {
                "model_name": "realesr-general-x4v3",
                "supports_denoise": True,
                "description": "General model with adjustable denoising",
            },
            "realesr-animevideov3 (Anime Video)": {
                "model_name": "realesr-animevideov3",
                "supports_denoise": False,
                "description": "Designed specifically for anime videos",
            },
            "RealESRGAN_x2plus (Fast 2x)": {
                "model_name": "RealESRGAN_x2plus",
                "supports_denoise": False,
                "description": "Faster 2x upscaling",
            },
        }

    # Build options dict from config
    result = {}
    for model_name, model_config in models.items():
        display_name = model_config.get("display_name", model_name)
        result[display_name] = {
            "model_name": model_name,
            "supports_denoise": model_config.get("supports_denoise", False),
            "description": model_config.get("description", ""),
        }
    return result


def get_swinir_model_options() -> Dict[str, Dict[str, Any]]:
    """
    Get SwinIR/Swin2SR models in UI-compatible format.

    Returns:
        Dict mapping display names to model configurations
    """
    return {
        "Swin2SR_RealWorld_x4 (Recommended)": {
            "model_name": "Swin2SR_RealWorld_x4",
            "scale": 4,
            "description": "Real-world SR 4x - Best for degraded images",
        },
        "Swin2SR_x4 (Standard)": {
            "model_name": "Swin2SR_x4",
            "scale": 4,
            "description": "Classical SR 4x - Standard upscaling",
        },
        "Swin2SR_Compressed_x4 (JPEG Artifacts)": {
            "model_name": "Swin2SR_Compressed_x4",
            "scale": 4,
            "description": "Compressed SR 4x - For JPEG artifacts",
        },
        "Swin2SR_x2 (Fast 2x)": {
            "model_name": "Swin2SR_x2",
            "scale": 2,
            "description": "Classical SR 2x - Fast, good for clean images",
        },
    }


def get_hat_model_options() -> Dict[str, Dict[str, Any]]:
    """
    Get HAT (Hybrid Attention Transformer) models in UI-compatible format.

    Returns:
        Dict mapping display names to model configurations
    """
    return {
        "HAT_Base_4x_ImageNet (General)": {
            "model_name": "HAT_Base_4x_ImageNet",
            "scale": 4,
            "description": "HAT Base 4x pre-trained on ImageNet — strong general SR",
        },
        "HAT_Base_4x_GAN_sharper (Real-World Sharpness)": {
            "model_name": "HAT_Base_4x_GAN_sharper",
            "scale": 4,
            "description": "HAT GAN 4x — sharper perceptual quality for real-world SR",
        },
    }
