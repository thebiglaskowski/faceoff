"""Settings tab: full-options config.yaml editor.

Every editable leaf in config.yaml gets a widget, generated from
``SECTIONS`` below so widget order, save order, and tests share one
source of truth. Saves go through ``Config.write_values``, which
replaces lines in place — file comments and layout survive. The three
model-catalog tables are system configuration and render read-only.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gradio as gr

from utils.config_manager import config

logger = logging.getLogger("FaceOff")

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
CUDNN_ALGOS = ["HEURISTIC", "EXHAUSTIVE"]
THEMES = ["soft", "neon"]
VIDEO_CODECS = ["libx264", "libx265", "mpeg4"]
VIDEO_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
]
OUTSCALES = [2, 4]
EVICTION_POLICIES = ["lru"]

#: Paths whose change only takes effect after a restart.
RESTART_KEYS = {
    ("ui", "server_name"),
    ("ui", "server_port"),
    ("ui", "share"),
    ("logging", "log_file"),
    ("logging", "json_log_file"),
    ("model_cache", "tensorrt_cache_dir"),
    ("model_cache", "preload_on_startup"),
    ("model_cache", "preload_models"),
    ("model_cache", "preload_enhancement_on_startup"),
    ("directories", "temp_gif_frames"),
    ("directories", "temp_gif_enhanced"),
    ("directories", "output"),
    ("directories", "models"),
    ("directories", "cache"),
}


@dataclass(frozen=True)
class _Bool:
    path: Tuple[str, ...]
    label: str
    default: bool = False
    info: str = ""


@dataclass(frozen=True)
class _Int:
    path: Tuple[str, ...]
    label: str
    default: int = 0
    minimum: int = 0
    maximum: int = 100
    step: int = 1
    info: str = ""


@dataclass(frozen=True)
class _Float:
    path: Tuple[str, ...]
    label: str
    default: float = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    info: str = ""


@dataclass(frozen=True)
class _Text:
    path: Tuple[str, ...]
    label: str
    default: str = ""
    info: str = ""


@dataclass(frozen=True)
class _Choice:
    path: Tuple[str, ...]
    label: str
    choices: Tuple[str, ...] = ()
    default: str = ""
    info: str = ""


@dataclass(frozen=True)
class _IntChoice:
    path: Tuple[str, ...]
    label: str
    choices: Tuple[int, ...] = ()
    default: int = 0
    info: str = ""


@dataclass(frozen=True)
class _Pair:
    path: Tuple[str, ...]
    label: str
    default: Tuple[int, int] = (640, 640)
    minimum: int = 160
    maximum: int = 1280
    info: str = ""


@dataclass(frozen=True)
class _Csv:
    path: Tuple[str, ...]
    label: str
    default: Tuple[str, ...] = ()
    info: str = ""


def _enhancement_model_choices() -> Tuple[str, ...]:
    models = config.get("enhancement", "models", default={}) or {}
    if models:
        return tuple(models.keys())
    return ("RealESRGAN_x4plus",)


SECTIONS: List[Tuple[str, List[Any]]] = [
    (
        "Limits & Files",
        [
            _Int(
                ("limits", "max_file_size_mb"),
                "Max upload size (MB)",
                500,
                10,
                4000,
                10,
            ),
            _Int(
                ("limits", "max_video_duration_sec"),
                "Max video duration (sec)",
                300,
                10,
                3600,
                10,
            ),
            _Int(
                ("limits", "max_image_pixels"),
                "Max image pixels",
                16777216,
                65536,
                67108864,
                65536,
            ),
            _Int(
                ("limits", "max_gif_frames"),
                "Max GIF frames",
                500,
                10,
                5000,
                10,
            ),
        ],
    ),
    (
        "GPU & Performance",
        [
            _Int(("gpu", "batch_size"), "Frame batch size", 8, 1, 64),
            _Int(("gpu", "max_batch_size"), "Max batch size (OOM guard)", 16, 1, 64),
            _Int(("gpu", "workers_per_gpu"), "Worker threads per GPU", 4, 1, 16),
            _Bool(
                ("gpu", "multi_gpu_video_enabled"),
                "Multi-GPU video/GIF swap",
                True,
            ),
            _Bool(("gpu", "tensorrt_enabled"), "TensorRT optimization", True),
            _Bool(("gpu", "tensorrt_fp16"), "TensorRT FP16 precision", True),
            _Int(
                ("gpu", "tensorrt_workspace_mb"),
                "TensorRT workspace (MB)",
                2048,
                256,
                8192,
                256,
            ),
            _Int(
                ("gpu", "onnx_mem_limit_mb"),
                "ONNX memory headroom (MB)",
                3072,
                512,
                16384,
                256,
                "Lower this on BFC arena allocation errors",
            ),
            _Choice(
                ("gpu", "cudnn_conv_algo_search"),
                "cuDNN conv algo search",
                tuple(CUDNN_ALGOS),
                "HEURISTIC",
                "HEURISTIC is safer on WSL2",
            ),
            _Bool(
                ("gpu", "frame_retention_enabled"),
                "GPU chunk upload + swap IoBinding",
                True,
            ),
            _Bool(("gpu", "paste_on_gpu"), "GPU paste-back", True),
            _Bool(("gpu", "detection_on_gpu"), "GPU detection", True),
            _Bool(
                ("gpu", "enhancement_chain_enabled"),
                "GPU enhancement chain",
                True,
            ),
            _Bool(("gpu", "auto_workload_tune"), "Auto workload tuning", True),
        ],
    ),
    (
        "Face Detection",
        [
            _Text(
                ("face_detection", "inswapper_model_path"),
                "Inswapper model path",
                "inswapper_128.onnx",
            ),
            _Text(
                ("face_detection", "buffalo_model_path"),
                "Buffalo model path",
                "models/buffalo_l",
            ),
            _Text(
                ("face_detection", "face_analysis_name"),
                "Face analysis pack",
                "buffalo_l",
            ),
            _Pair(
                ("face_detection", "face_analysis_det_size"),
                "Detection size (w × h)",
                (640, 640),
                160,
                1280,
            ),
            _Float(
                ("face_detection", "confidence_threshold"),
                "Confidence threshold",
                0.5,
                0.0,
                1.0,
                0.05,
            ),
            _Float(
                ("face_detection", "iou_threshold"),
                "IoU threshold",
                0.3,
                0.1,
                0.5,
                0.05,
            ),
            _Bool(
                ("face_detection", "embedding_tracking_enabled"),
                "Embedding re-ID across cuts",
                True,
            ),
            _Float(
                ("face_detection", "embedding_similarity_threshold"),
                "Same-person similarity",
                0.35,
                0.2,
                0.6,
                0.05,
            ),
            _Float(
                ("face_detection", "embedding_ema_alpha"),
                "Identity anchor blend",
                0.25,
                0.1,
                0.5,
                0.05,
            ),
            _Int(
                ("face_detection", "ui_padding"), "Face thumbnail padding", 20, 0, 100
            ),
            _Bool(
                ("face_detection", "adaptive_enabled"),
                "Adaptive resolution",
                True,
            ),
            _Float(
                ("face_detection", "detection_scale"),
                "Detection scale",
                0.5,
                0.25,
                1.0,
                0.05,
            ),
            _Int(
                ("face_detection", "min_resolution"),
                "Min detection resolution",
                640,
                160,
                1920,
                32,
            ),
        ],
    ),
    (
        "Enhancement (RealESRGAN)",
        [
            _Choice(
                ("enhancement", "default_model"),
                "Default model",
                _enhancement_model_choices(),
                "realesr-general-x4v3",
            ),
            _Bool(
                ("enhancement", "multi_gpu_enabled"),
                "Shard frames across GPUs",
                False,
                "RealESRGAN only; HAT/SwinIR stay single-GPU",
            ),
            _Int(
                ("enhancement", "defaults", "tile_size"),
                "Tile size",
                256,
                64,
                512,
                64,
                "Lower = less VRAM",
            ),
            _IntChoice(
                ("enhancement", "defaults", "outscale"),
                "Upscale factor",
                tuple(OUTSCALES),
                4,
            ),
            _Int(
                ("enhancement", "defaults", "pre_pad"),
                "Edge padding",
                0,
                0,
                20,
            ),
            _Bool(("enhancement", "defaults", "use_fp32"), "FP32 precision", False),
            _Float(
                ("enhancement", "defaults", "denoise_strength"),
                "Denoise strength",
                0.5,
                0.0,
                1.0,
                0.05,
            ),
        ],
    ),
    (
        "Face Restoration (GFPGAN)",
        [
            _Bool(
                ("face_restoration", "enabled_by_default"),
                "Restore faces by default",
                False,
            ),
            _Text(("face_restoration", "model_version"), "Model version", "1.3"),
            _Float(
                ("face_restoration", "default_weight"),
                "Restoration weight",
                0.5,
                0.0,
                1.0,
                0.05,
            ),
        ],
    ),
    (
        "Streaming Pipeline",
        [
            _Bool(("streaming", "enabled"), "Chunked streaming", True),
            _Int(("streaming", "chunk_size"), "Frames per chunk", 32, 4, 128, 4),
            _Bool(
                ("streaming", "video_face_enhance"),
                "Per-frame GFPGAN in ESRGAN",
                False,
                "~2x slower; prefer Restore Faces",
            ),
            _Float(
                ("streaming", "gif_decode_fps"),
                "GIF decode rate (fps)",
                10.0,
                1.0,
                30.0,
                1.0,
            ),
            _Bool(("streaming", "hwaccel_decode"), "FFmpeg CUDA hwaccel", True),
            _Bool(("streaming", "zero_copy_enabled"), "NVDEC zero-copy", True),
            _Bool(("streaming", "nvcodec_decode"), "PyNvVideoCodec decode", True),
            _Bool(("streaming", "nvenc_encode"), "NVENC encode", True),
            _Choice(
                ("streaming", "video_codec"),
                "Software video codec",
                tuple(VIDEO_CODECS),
                "libx264",
            ),
            _Choice(
                ("streaming", "video_preset"),
                "Encoder preset",
                tuple(VIDEO_PRESETS),
                "medium",
            ),
            _Int(("streaming", "video_crf"), "Video CRF", 18, 0, 51),
        ],
    ),
    (
        "Compression & GIF Output",
        [
            _Bool(("compression", "enabled"), "Post-process compression", True),
            _Bool(
                ("compression", "skip_optimized_video"),
                "Skip re-encode if already NVENC/h264",
                True,
            ),
            _Float(
                ("compression", "min_image_size_mb"),
                "Compress images above (MB)",
                5.0,
                0.0,
                50.0,
                1.0,
            ),
            _Int(("gif", "palette_colors"), "GIF palette size", 256, 2, 256),
            _Bool(("gif", "dither"), "Floyd-Steinberg dither", False),
        ],
    ),
    (
        "Logging",
        [
            _Text(("logging", "log_file"), "Log file", "app.log"),
            _Int(("logging", "max_file_size_mb"), "Rotation size (MB)", 25, 1, 500),
            _Int(("logging", "backup_count"), "Rotated backups", 5, 0, 20),
            _Choice(
                ("logging", "console_level"),
                "Terminal level",
                tuple(LOG_LEVELS),
                "INFO",
            ),
            _Choice(
                ("logging", "file_level"), "File level", tuple(LOG_LEVELS), "DEBUG"
            ),
            _Text(
                ("logging", "format"),
                "Log format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ),
            _Text(("logging", "date_format"), "Date format", "%Y-%m-%d %H:%M:%S"),
            _Int(
                ("logging", "terminal_buffer_lines"),
                "Logs-tab buffer lines",
                5000,
                1000,
                20000,
                500,
            ),
            _Float(
                ("logging", "terminal_auto_refresh_sec"),
                "Logs-tab refresh (sec, 0 = manual)",
                2.0,
                0.0,
                30.0,
                0.5,
            ),
            _Bool(("logging", "json_format"), "JSON log sidecar", True),
            _Text(("logging", "json_log_file"), "JSON log file", "app.json.log"),
        ],
    ),
    (
        "Model Cache",
        [
            _Text(
                ("model_cache", "tensorrt_cache_dir"),
                "TensorRT cache dir",
                "cache/tensorrt",
            ),
            _Bool(
                ("model_cache", "tensorrt_cache_enabled"),
                "Persist TRT engines",
                True,
            ),
            _Bool(
                ("model_cache", "tensorrt_timing_cache_enabled"),
                "Kernel-timing cache",
                True,
            ),
            _Bool(
                ("model_cache", "preload_on_startup"),
                "Preload models at startup",
                True,
            ),
            _Csv(
                ("model_cache", "preload_models"),
                "Preload list (comma-separated)",
                ("buffalo_l", "inswapper_128"),
            ),
            _Bool(
                ("model_cache", "preload_enhancement_on_startup"),
                "Warm enhancement model",
                False,
            ),
        ],
    ),
    (
        "Memory",
        [
            _Bool(("memory", "auto_clear_cache"), "Auto-clear CUDA cache", True),
            _Int(
                ("memory", "clear_cache_threshold_mb"),
                "Pre-enhance clear threshold (MB)",
                1536,
                256,
                8192,
                128,
            ),
            _Int(
                ("memory", "mb_per_batch_estimate"),
                "Batch VRAM estimate (MB)",
                400,
                50,
                2000,
                50,
            ),
            _Bool(
                ("memory", "release_swap_models_before_enhance"),
                "Unload swap models before enhance",
                True,
            ),
            _Int(
                ("memory", "hat_force_tiled_below_free_mb"),
                "Force tiled HAT below free (MB)",
                3072,
                512,
                16384,
                256,
            ),
            _Int(
                ("memory", "hat_oom_min_tile_size"),
                "Smallest OOM-retry tile",
                64,
                32,
                256,
                32,
            ),
            _Bool(("memory", "reduce_batch_on_oom"), "Shrink batch on OOM", True),
            _Int(("memory", "min_batch_size"), "Minimum batch size", 1, 1, 8),
        ],
    ),
    (
        "File Formats",
        [
            _Csv(
                ("file_formats", "images"),
                "Image extensions",
                (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
            ),
            _Csv(
                ("file_formats", "videos"),
                "Video extensions",
                (".mp4", ".webp", ".avi", ".mov"),
            ),
            _Csv(("file_formats", "gifs"), "GIF extensions", (".gif",)),
        ],
    ),
    (
        "Directories",
        [
            _Text(
                ("directories", "temp_gif_frames"),
                "GIF frame temp dir",
                "temp/gif/frames",
            ),
            _Text(
                ("directories", "temp_gif_enhanced"),
                "GIF enhanced temp dir",
                "temp/gif/enhanced",
            ),
            _Text(("directories", "output"), "Output dir", "outputs"),
            _Text(("directories", "models"), "Models dir", "models"),
            _Text(("directories", "cache"), "Cache dir", "cache"),
        ],
    ),
    (
        "Cleanup",
        [
            _Bool(("cleanup", "enabled"), "Auto-clean old files", True),
            _Float(
                ("cleanup", "max_age_hours"),
                "Delete files older than (hours)",
                24.0,
                1.0,
                720.0,
                1.0,
            ),
        ],
    ),
    (
        "Server & Queue",
        [
            _Text(("ui", "server_name"), "Bind address", "127.0.0.1"),
            _Int(("ui", "server_port"), "Port", 7860, 1024, 65535),
            _Bool(("ui", "share"), "Public Gradio link", False),
            _Choice(("ui", "theme"), "Theme", tuple(THEMES), "soft"),
            _Int(("ui", "queue_max_size"), "Max queued requests", 20, 1, 100),
            _Int(
                ("ui", "queue_concurrency_limit"),
                "Max concurrent tasks",
                4,
                1,
                16,
            ),
        ],
    ),
    (
        "In-Memory Model Limits",
        [
            _Int(
                ("model_cache_limits", "max_models_per_cache"),
                "Max models per cache",
                5,
                1,
                20,
            ),
            _Choice(
                ("model_cache_limits", "eviction_policy"),
                "Eviction policy",
                tuple(EVICTION_POLICIES),
                "lru",
            ),
        ],
    ),
]

#: Read-only system tables shown as JSON in their sections.
READONLY_TABLES = {
    "Enhancement (RealESRGAN)": ("enhancement", "models"),
    "HAT Models": ("hat_enhancement", "models"),
    "SwinIR Models": ("swinir", "models"),
}


def iter_editable_fields():
    """Yield every editable spec in widget order (single source of truth)."""
    for _section, fields in SECTIONS:
        for spec in fields:
            yield spec


def field_arity(spec) -> int:
    """Number of Gradio inputs one spec consumes (pairs take two)."""
    return 2 if isinstance(spec, _Pair) else 1


def total_inputs() -> int:
    """Total widget inputs the save handler receives."""
    return sum(field_arity(spec) for spec in iter_editable_fields())


def value_offset(path: Tuple[str, ...]) -> int:
    """Cursor into the flat value list where ``path`` starts."""
    cursor = 0
    for spec in iter_editable_fields():
        if tuple(spec.path) == tuple(path):
            return cursor
        cursor += field_arity(spec)
    raise KeyError(f"Unknown settings path: {path}")


def collect_current_values() -> List[Any]:
    """Read current config.yaml values in widget order (pairs expand)."""
    values: List[Any] = []
    for spec in iter_editable_fields():
        current = config.get(*spec.path, default=spec.default)
        if isinstance(spec, _Bool):
            values.append(bool(current))
        elif isinstance(spec, _Int):
            values.append(int(current))
        elif isinstance(spec, _Float):
            values.append(float(current))
        elif isinstance(spec, _Text):
            values.append("" if current is None else str(current))
        elif isinstance(spec, _Choice):
            values.append(current if current in spec.choices else spec.default)
        elif isinstance(spec, _IntChoice):
            try:
                values.append(
                    int(current) if int(current) in spec.choices else spec.default
                )
            except (TypeError, ValueError):
                values.append(spec.default)
        elif isinstance(spec, _Pair):
            try:
                w, h = list(current)[:2]
                values.extend([int(w), int(h)])
            except (TypeError, ValueError):
                values.extend(list(spec.default))
        elif isinstance(spec, _Csv):
            values.append(", ".join(current or spec.default))
    return values


def _coerce(spec, raw: Any, extra: Any = None):
    """Cast one widget value; raises ValueError with a UI-ready message."""
    label = spec.label
    if isinstance(spec, _Bool):
        return bool(raw)
    if isinstance(spec, _Int):
        try:
            value = int(round(float(raw)))
        except (TypeError, ValueError):
            raise ValueError(f"{label}: expected a whole number")
        if not spec.minimum <= value <= spec.maximum:
            raise ValueError(
                f"{label}: {value} is outside {spec.minimum}–{spec.maximum}"
            )
        return value
    if isinstance(spec, _Float):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: expected a number")
        if not spec.minimum <= value <= spec.maximum:
            raise ValueError(
                f"{label}: {value} is outside {spec.minimum}–{spec.maximum}"
            )
        return value
    if isinstance(spec, _Text):
        return "" if raw is None else str(raw).strip()
    if isinstance(spec, _Choice):
        if raw not in spec.choices:
            raise ValueError(f"{label}: {raw!r} is not a valid choice")
        return raw
    if isinstance(spec, _IntChoice):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: expected one of {list(spec.choices)}")
        if value not in spec.choices:
            raise ValueError(f"{label}: {value} is not a valid choice")
        return value
    if isinstance(spec, _Pair):
        try:
            w, h = int(round(float(raw))), int(round(float(extra)))
        except (TypeError, ValueError):
            raise ValueError(f"{label}: expected two whole numbers")
        for dim in (w, h):
            if not spec.minimum <= dim <= spec.maximum:
                raise ValueError(
                    f"{label}: {dim} is outside {spec.minimum}–{spec.maximum}"
                )
        return (w, h)
    if isinstance(spec, _Csv):
        items = [part.strip() for part in str(raw or "").split(",")]
        items = [item for item in items if item]
        if not items:
            raise ValueError(f"{label}: enter at least one value")
        return items
    raise ValueError(f"{label}: unsupported field type")


def apply_settings_values(*values: Any) -> str:
    """Validate widget values and persist them to config.yaml."""
    specs = list(iter_editable_fields())
    if len(values) != total_inputs():
        return (
            f"❌ Internal error: got {len(values)} values for "
            f"{total_inputs()} settings — nothing saved."
        )
    scalars: Dict[tuple, Any] = {}
    pairs: Dict[tuple, tuple] = {}
    str_lists: Dict[tuple, list] = {}
    # Only persist what actually changed: keeps the file diff minimal
    # (e.g. unquoted strings are not re-quoted) and makes the restart
    # note fire only when a restart-gated key changed.
    current_flat = collect_current_values()
    cursor = 0
    try:
        for spec in specs:
            if isinstance(spec, _Pair):
                parsed = _coerce(spec, values[cursor], values[cursor + 1])
                current_pair = (current_flat[cursor], current_flat[cursor + 1])
                if (int(parsed[0]), int(parsed[1])) != (
                    int(current_pair[0]),
                    int(current_pair[1]),
                ):
                    pairs[spec.path] = parsed
                cursor += 2
            elif isinstance(spec, _Csv):
                parsed = _coerce(spec, values[cursor])
                current_items = [
                    part.strip() for part in str(current_flat[cursor] or "").split(",")
                ]
                current_items = [item for item in current_items if item]
                if list(parsed) != current_items:
                    str_lists[spec.path] = parsed
                cursor += 1
            else:
                parsed = _coerce(spec, values[cursor])
                current_value = current_flat[cursor]
                if isinstance(parsed, float) and isinstance(
                    current_value, (int, float)
                ):
                    changed = float(parsed) != float(current_value)
                else:
                    changed = parsed != current_value
                if changed:
                    scalars[spec.path] = parsed
                cursor += 1
    except ValueError as exc:
        return f"❌ {exc} — nothing saved."
    if not (scalars or pairs or str_lists):
        return "✅ No changes — config.yaml already matches these values."
    try:
        config.write_values(scalars=scalars, pairs=pairs, lists=str_lists)
    except ValueError as exc:
        logger.error("Settings save failed: %s", exc)
        return f"❌ Save failed: {exc} — config.yaml unchanged."
    count = len(scalars) + len(pairs) + len(str_lists)
    message = f"✅ Saved {count} settings to config.yaml."
    touched = set(scalars) | set(pairs) | set(str_lists)
    if touched & RESTART_KEYS:
        message += (
            " Server, logging-file, preload, directory, or cache settings "
            "changed — restart the app for those to take effect. Everything "
            "else applies to the next swap."
        )
    else:
        message += " Values apply to the next swap."
    logger.info("Settings tab saved %d value(s)", count)
    return message


def discard_settings_values() -> List[Any]:
    """Re-read config.yaml (used by the Discard button)."""
    return collect_current_values() + [
        "↩️ Discarded UI edits — showing config.yaml values."
    ]


def _build_widget(spec):
    current = config.get(*spec.path, default=spec.default)
    if isinstance(spec, _Bool):
        return gr.Checkbox(label=spec.label, value=bool(current), info=spec.info)
    if isinstance(spec, _Int):
        return gr.Slider(
            minimum=spec.minimum,
            maximum=spec.maximum,
            step=spec.step,
            value=int(current),
            label=spec.label,
            info=spec.info,
        )
    if isinstance(spec, _Float):
        return gr.Slider(
            minimum=spec.minimum,
            maximum=spec.maximum,
            step=spec.step,
            value=float(current),
            label=spec.label,
            info=spec.info,
        )
    if isinstance(spec, _Text):
        return gr.Textbox(
            value="" if current is None else str(current),
            label=spec.label,
            info=spec.info,
        )
    if isinstance(spec, _Choice):
        return gr.Dropdown(
            choices=list(spec.choices),
            value=current if current in spec.choices else spec.default,
            label=spec.label,
            info=spec.info,
        )
    if isinstance(spec, _IntChoice):
        try:
            value = int(current) if int(current) in spec.choices else spec.default
        except (TypeError, ValueError):
            value = spec.default
        return gr.Dropdown(
            choices=list(spec.choices),
            value=value,
            label=spec.label,
            info=spec.info,
        )
    if isinstance(spec, _Csv):
        return gr.Textbox(
            value=", ".join(current or spec.default),
            label=spec.label,
            info=spec.info,
        )
    raise ValueError(f"Cannot build widget for {spec.label}")


def create_settings_tab() -> dict:
    """Create the Settings tab with every config.yaml option."""
    with gr.Tab("⚙️ Settings"):
        gr.Markdown(
            "All `config.yaml` options. **Save** writes them back preserving "
            "comments; **Discard** re-reads the file. Server address, ports, "
            "log files, preload, directory, and cache settings need a restart — "
            "everything else applies to the next swap."
        )
        with gr.Row():
            save_btn = gr.Button("💾 Save settings", variant="primary", size="sm")
            discard_btn = gr.Button(
                "↩️ Discard changes", size="sm", variant="secondary"
            )
        status = gr.Markdown(
            value="Edit values, then **Save settings**. "
            "Model catalogs below are system configuration (read-only)."
        )

        field_widgets: List[Any] = []
        for section, fields in SECTIONS:
            with gr.Accordion(section, open=False):
                for spec in fields:
                    if isinstance(spec, _Pair):
                        current = config.get(*spec.path, default=spec.default)
                        try:
                            w0, h0 = list(current)[:2]
                        except (TypeError, ValueError):
                            w0, h0 = spec.default
                        with gr.Row():
                            field_widgets.append(
                                gr.Number(
                                    value=int(w0),
                                    minimum=spec.minimum,
                                    maximum=spec.maximum,
                                    precision=0,
                                    label=f"{spec.label} — width",
                                    info=spec.info,
                                )
                            )
                            field_widgets.append(
                                gr.Number(
                                    value=int(h0),
                                    minimum=spec.minimum,
                                    maximum=spec.maximum,
                                    precision=0,
                                    label=f"{spec.label} — height",
                                )
                            )
                    else:
                        field_widgets.append(_build_widget(spec))
                table_path = READONLY_TABLES.get(section)
                if table_path is not None:
                    gr.JSON(
                        value=config.get(*table_path, default={}),
                        label="Model catalog (read-only)",
                    )
        for title, table_path in READONLY_TABLES.items():
            if title in {section for section, _fields in SECTIONS}:
                continue
            with gr.Accordion(title, open=False):
                gr.JSON(
                    value=config.get(*table_path, default={}),
                    label="Model catalog (read-only)",
                )

        with gr.Row():
            save_btn_bottom = gr.Button(
                "💾 Save settings", variant="primary", size="sm"
            )

    return {
        "field_widgets": field_widgets,
        "save_btn": save_btn,
        "save_btn_bottom": save_btn_bottom,
        "discard_btn": discard_btn,
        "status": status,
    }


def wire_settings_tab(comps) -> None:
    """Bind Save/Discard for the Settings tab."""
    field_inputs = list(comps["field_widgets"])
    for button in (comps["save_btn"], comps["save_btn_bottom"]):
        button.click(
            fn=apply_settings_values,
            inputs=field_inputs,
            outputs=[comps["status"]],
        )
    comps["discard_btn"].click(
        fn=discard_settings_values,
        inputs=[],
        outputs=[*field_inputs, comps["status"]],
    )
