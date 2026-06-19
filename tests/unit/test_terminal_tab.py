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