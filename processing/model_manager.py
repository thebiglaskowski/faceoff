"""Backend for the Models UI tab.

UI-safe helpers (imported by ``ui/`` through this ``processing/`` module,
never ``core/`` directly): installed-model status, upstream downloads via
``scripts/download_models.py`` as a subprocess with a captured progress
tail, TensorRT cache inspection/clearing, and integrity verification.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator, List

logger = logging.getLogger("FaceOff")

# Set by the UI Cancel button; checked by stream_model_download.
_cancel_download = threading.Event()

# Hard cap for one download run (matches the old blocking timeout).
DOWNLOAD_TIMEOUT_SEC = 3600

# UI dropdown label -> --models value for scripts/download_models.py
AVAILABLE_DOWNLOADS = {
    "All models (recommended)": "all",
    "inswapper_128.onnx (face swap)": "inswapper",
    "buffalo_l (face detection)": "buffalo_l",
    "osediff (diffusion quality, ~5 GB)": "osediff",
}

# (display name, relative path, kind) for the status table.
MANAGED_MODELS = [
    ("inswapper_128.onnx", "inswapper_128.onnx", "file"),
    ("buffalo_l", "models/buffalo_l", "dir"),
]

# Keep the download log textbox bounded.
LOG_TAIL_LINES = 60


def _root() -> Path:
    return Path.cwd()


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024


def get_model_status() -> List[List[str]]:
    """Rows of [Model, Size, Status] for the UI dataframe."""
    try:
        from utils.config_manager import config

        inswapper = Path(config.inswapper_model_path)
        buffalo = Path(config.buffalo_model_path)
    except Exception:
        inswapper = _root() / "inswapper_128.onnx"
        buffalo = _root() / "models" / "buffalo_l"

    rows = []
    if inswapper.is_file():
        size = inswapper.stat().st_size / 1024 / 1024
        rows.append(["inswapper_128.onnx", f"{size:.0f} MB", "✅ Installed"])
    else:
        rows.append(["inswapper_128.onnx", "—", "❌ Missing"])

    if buffalo.is_dir() and (buffalo / "det_10g.onnx").exists():
        count = len(list(buffalo.glob("*.onnx")))
        rows.append(
            [
                "buffalo_l",
                f"{_dir_size_mb(buffalo):.0f} MB",
                f"✅ Installed ({count} files)",
            ]
        )
    else:
        rows.append(["buffalo_l", "—", "❌ Missing"])
    return rows


def run_model_download(selection: str, force: bool = False) -> str:
    """Run scripts/download_models.py, return the tailed log for the UI."""
    models_arg = AVAILABLE_DOWNLOADS.get(selection, "all")
    script = Path(__file__).resolve().parent.parent / "scripts" / "download_models.py"
    cmd = [sys.executable, str(script), "--models", models_arg]
    if force:
        cmd.append("--force")
    logger.info("Model download started: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return "❌ Download timed out after 1 hour. Check your connection and retry."
    except Exception as exc:
        return f"❌ Could not launch download script: {exc}"

    output = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    tail = "\n".join(lines[-LOG_TAIL_LINES:])
    status = (
        "✅ Download finished."
        if proc.returncode == 0
        else (f"❌ Download script exited with code {proc.returncode}.")
    )
    logger.info("Model download finished with code %d", proc.returncode)
    return f"{status}\n\n{tail}" if tail else status


def request_download_cancel() -> None:
    """Signal a running stream_model_download to stop (UI Cancel button)."""
    _cancel_download.set()


def _download_cmd(selection: str, force: bool) -> List[str]:
    models_arg = AVAILABLE_DOWNLOADS.get(selection, "all")
    script = Path(__file__).resolve().parent.parent / "scripts" / "download_models.py"
    # -u: unbuffered child stdout so progress lines stream live.
    cmd = [sys.executable, "-u", str(script), "--models", models_arg]
    if force:
        cmd.append("--force")
    return cmd


def stream_model_download(selection: str, force: bool = False) -> Iterator[str]:
    """Yield the download log live; last yield carries the final status.

    Generator wired to the Gradio download button. Honors
    request_download_cancel() and DOWNLOAD_TIMEOUT_SEC.
    """
    _cancel_download.clear()
    cmd = _download_cmd(selection, force)
    logger.info("Model download started: %s", " ".join(cmd))
    tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        yield f"❌ Could not launch download script: {exc}"
        return

    start = time.monotonic()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            tail.append(line.rstrip())
            if _cancel_download.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                logger.info("Model download cancelled by user")
                yield "⛔ Download cancelled by user.\n\n" + "\n".join(tail)
                return
            if time.monotonic() - start > DOWNLOAD_TIMEOUT_SEC:
                proc.kill()
                yield "❌ Download timed out after 1 hour.\n\n" + "\n".join(tail)
                return
            yield "\n".join(tail)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    code = proc.returncode
    logger.info("Model download finished with code %d", code)
    status = (
        "✅ Download finished."
        if code == 0
        else f"❌ Download script exited with code {code}."
    )
    body = "\n".join(tail)
    yield f"{status}\n\n{body}" if body else status


def get_trt_cache_text() -> str:
    """Human-readable TensorRT cache size for the UI."""
    try:
        from utils.model_cache import get_cache_info

        info = get_cache_info()
        return (
            f"**Engines:** {info['num_files']} cached "
            f"({info['total_size_mb']:.1f} MB) in `{info['cache_dir']}`"
        )
    except Exception as exc:
        return f"Cache info unavailable: {exc}"


def clear_trt_cache() -> str:
    """Delete cached TensorRT engines; engines rebuild on next run."""
    try:
        from utils.model_cache import get_cache_info

        info = get_cache_info()
        cache_dir = Path(info["cache_dir"])
        removed = 0
        if cache_dir.is_dir():
            for f in list(cache_dir.glob("*.cache")) + list(cache_dir.glob("gpu_*")):
                if f.is_file():
                    f.unlink()
                    removed += 1
                elif f.is_dir():
                    import shutil

                    shutil.rmtree(f, ignore_errors=True)
                    removed += 1
        logger.info("TensorRT cache cleared (%d entries)", removed)
        return (
            f"🗑️ Cleared {removed} cache entr{'y' if removed == 1 else 'ies'}. "
            + get_trt_cache_text()
        )
    except Exception as exc:
        return f"❌ Cache clear failed: {exc}"


def verify_models() -> str:
    """Check managed weights exist and meet minimum sizes; report markdown."""
    try:
        import scripts.download_models as dm
    except Exception:
        dm = None

    lines = []
    swapper = _root() / "inswapper_128.onnx"
    buffalo_marker = _root() / "models" / "buffalo_l" / "det_10g.onnx"
    min_swap = getattr(dm, "INSWAPPER_MIN_BYTES", 0) or 0
    min_buff = getattr(dm, "BUFFALO_L_MIN_BYTES", 0) or 0

    if swapper.is_file() and swapper.stat().st_size >= min_swap:
        lines.append(
            f"✅ `inswapper_128.onnx` OK ({swapper.stat().st_size / 1024 / 1024:.0f} MB)"
        )
    else:
        lines.append("❌ `inswapper_128.onnx` missing or suspiciously small")

    if buffalo_marker.is_file():
        total = _dir_size_mb(buffalo_marker.parent)
        ok = total * 1024 * 1024 >= min_buff
        lines.append(
            f"{'✅' if ok else '❌'} `models/buffalo_l/` "
            f"{'OK' if ok else 'incomplete'} ({total:.0f} MB)"
        )
    else:
        lines.append("❌ `models/buffalo_l/` missing (no `det_10g.onnx`)")
    return "\n\n".join(lines)
