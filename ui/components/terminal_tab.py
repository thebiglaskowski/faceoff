"""
In-app live log viewer backed by a thread-safe ring buffer.

Captures root logger output for the Gradio Logs tab while file/console
handlers continue to write to app.log and the launch terminal.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from typing import Deque, Dict, List, Tuple

logger = logging.getLogger("FaceOff")


def _buffer_capacity() -> int:
    try:
        from utils.config_manager import config

        return int(config.log_terminal_buffer_lines)
    except Exception:
        return 3000


class TerminalLogHandler(logging.Handler):
    """Thread-safe ring buffer handler for live UI log display."""

    def __init__(self, capacity: int = 3000) -> None:
        super().__init__()
        self._capacity = max(1, capacity)
        self._lock = threading.Lock()
        self._lines: Deque[str] = deque(maxlen=self._capacity)
        self._level_counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self._lock:
                self._lines.append(line)
                self._level_counts[record.levelname] += 1
        except Exception:
            self.handleError(record)

    def get_logs(self) -> List[str]:
        with self._lock:
            return list(self._lines)

    def get_level_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._level_counts)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._level_counts.clear()


terminal_handler = TerminalLogHandler(capacity=_buffer_capacity())


def get_terminal_output() -> str:
    """Return buffered logs as a single string for Gradio display."""
    logs = terminal_handler.get_logs()
    if not logs:
        return "(no log entries yet — processing activity will appear here)"
    return "\n".join(logs)


def get_terminal_stats() -> str:
    """Human-readable summary of captured log levels."""
    counts = terminal_handler.get_level_counts()
    if not counts:
        return "**Logs:** 0 entries captured"
    total = sum(counts.values())
    parts = ", ".join(f"{level}: {count}" for level, count in sorted(counts.items()))
    return f"**Logs:** {total} entries ({parts})"


def clear_terminal_logs() -> Tuple[str, str]:
    """Clear the in-memory buffer and return fresh view + stats."""
    terminal_handler.clear()
    logger.info("In-app log buffer cleared")
    return get_terminal_output(), get_terminal_stats()


def refresh_terminal_view() -> Tuple[str, str]:
    """Refresh log text and stats."""
    return get_terminal_output(), get_terminal_stats()


def create_terminal_tab() -> dict:
    """Create the Gradio Logs tab with auto-refresh."""
    import gradio as gr

    refresh_seconds = _auto_refresh_seconds()

    with gr.Tab("📟 Logs"):
        gr.Markdown(
            "Live application logs (respects `logging.console_level` / `logging.file_level`). "
            "Full history with rotation is also written to **`app.log`**."
        )
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")
            clear_btn = gr.Button("🗑️ Clear buffer", size="sm", variant="secondary")
            stats_display = gr.Markdown(value=get_terminal_stats())

        log_output = gr.Textbox(
            label="Application log",
            value=get_terminal_output(),
            lines=28,
            max_lines=40,
            interactive=False,
            buttons=["copy"],
            elem_id="faceoff_terminal_logs",
        )

        refresh_btn.click(
            refresh_terminal_view,
            outputs=[log_output, stats_display],
        )
        clear_btn.click(
            clear_terminal_logs,
            outputs=[log_output, stats_display],
        )

        if refresh_seconds > 0:
            timer = gr.Timer(refresh_seconds, active=True)
            timer.tick(get_terminal_output, outputs=log_output)

    return {
        "log_output": log_output,
        "stats_display": stats_display,
        "refresh_btn": refresh_btn,
        "clear_btn": clear_btn,
    }


def _auto_refresh_seconds() -> float:
    try:
        from utils.config_manager import config

        return float(config.log_terminal_auto_refresh_sec)
    except Exception:
        return 2.0