"""Unit tests for in-app terminal log handler."""

import logging
import threading

import pytest


@pytest.fixture(autouse=True)
def attached_terminal_handler():
    from ui.components.terminal_tab import terminal_handler

    terminal_handler.clear()
    terminal_handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    logging.getLogger("FaceOff").setLevel(logging.DEBUG)
    root.addHandler(terminal_handler)
    yield
    root.removeHandler(terminal_handler)
    terminal_handler.clear()


def test_terminal_handler_captures_log_levels():
    from ui.components.terminal_tab import (
        get_terminal_output,
        get_terminal_stats,
        terminal_handler,
    )

    test_logger = logging.getLogger("FaceOff.terminal_test")
    test_logger.info("info message")
    test_logger.warning("warn message")
    test_logger.debug("debug message")

    logs = terminal_handler.get_logs()
    assert len(logs) == 3
    output = get_terminal_output()
    assert "info message" in output
    assert "warn message" in output
    assert "debug message" in output
    assert "INFO" in get_terminal_stats()


def test_terminal_handler_ring_buffer_truncates():
    from ui.components.terminal_tab import TerminalLogHandler

    handler = TerminalLogHandler(capacity=3)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("FaceOff.ring_test")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    for idx in range(5):
        logger.info("line-%d", idx)

    assert handler.get_logs() == ["line-2", "line-3", "line-4"]
    logger.removeHandler(handler)
    logger.propagate = True


def test_terminal_handler_thread_safe():
    from ui.components.terminal_tab import terminal_handler

    logger = logging.getLogger("FaceOff.thread_test")

    def worker(offset: int) -> None:
        for idx in range(20):
            logger.debug("t%d-%d", offset, idx)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(terminal_handler.get_logs()) == 80


def test_clear_terminal_logs():
    from ui.components.terminal_tab import clear_terminal_logs, terminal_handler

    logging.getLogger("FaceOff").info("before clear")
    assert terminal_handler.get_logs()

    output, stats = clear_terminal_logs()
    # clear_terminal_logs() itself emits one INFO line after wiping the buffer
    assert len(terminal_handler.get_logs()) == 1
    assert "cleared" in output.lower()
    assert "INFO: 1" in stats


def test_level_filter_hides_lower_levels():
    from ui.components.terminal_tab import get_terminal_output

    logging.getLogger("FaceOff").debug("filter-debug-line")
    logging.getLogger("FaceOff").warning("filter-warn-line")

    assert "filter-debug-line" in get_terminal_output("DEBUG")
    warnings_only = get_terminal_output("WARNING")
    assert "filter-warn-line" in warnings_only
    assert "filter-debug-line" not in warnings_only


def test_unknown_level_filter_shows_everything():
    from ui.components.terminal_tab import get_terminal_output, terminal_handler

    logging.getLogger("FaceOff").debug("fallback-debug-line")
    assert terminal_handler.get_logs()
    assert "fallback-debug-line" in get_terminal_output("NOPE")


def test_refresh_view_honors_level_filter():
    from ui.components.terminal_tab import refresh_terminal_view

    logging.getLogger("FaceOff").debug("refresh-debug-line")
    logging.getLogger("FaceOff").error("refresh-error-line")

    text, stats = refresh_terminal_view("ERROR")
    assert "refresh-error-line" in text
    assert "refresh-debug-line" not in text
    assert "entries" in stats


def test_download_snapshot_contains_buffer(tmp_path, monkeypatch):
    import os

    from ui.components.terminal_tab import download_terminal_logs

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    logging.getLogger("FaceOff").info("download-marker-line")

    path = download_terminal_logs()
    try:
        assert path.endswith(".log")
        with open(path, encoding="utf-8") as fh:
            assert "download-marker-line" in fh.read()
    finally:
        os.unlink(path)


def test_terminal_tab_exposes_filter_and_download():
    import gradio as gr

    from ui.components.terminal_tab import LOG_LEVEL_CHOICES, create_terminal_tab

    with gr.Blocks():
        comps = create_terminal_tab()
    assert comps["level_filter"].value == "DEBUG"
    assert [c[0] for c in comps["level_filter"].choices] == LOG_LEVEL_CHOICES
    assert "Download" in comps["download_btn"].label
