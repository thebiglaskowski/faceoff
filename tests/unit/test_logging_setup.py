"""Unit tests for logging setup."""

import logging
from unittest.mock import MagicMock, patch

import pytest


def _mock_log_config(tmp_path, log_name: str = "test.log"):
    cfg = MagicMock()
    cfg.log_file = str(tmp_path / log_name)
    cfg.log_max_file_size_mb = 1
    cfg.log_backup_count = 1
    cfg.log_console_level = "DEBUG"
    cfg.log_file_level = "DEBUG"
    cfg.log_format = "%(levelname)s - %(message)s"
    cfg.log_date_format = "%H:%M:%S"
    cfg.log_json_format = False
    cfg.log_json_file = str(tmp_path / "test.json.log")
    cfg.log_terminal_buffer_lines = 100
    cfg.log_terminal_auto_refresh_sec = 0
    return cfg


@pytest.fixture(autouse=True)
def reset_logging_state():
    from utils import logging_setup

    logging_setup.reset_logging_for_tests()
    yield
    logging_setup.reset_logging_for_tests()


def test_setup_logging_is_idempotent(tmp_path):
    from utils import logging_setup

    with patch.object(logging_setup, "config", _mock_log_config(tmp_path)):
        logging_setup.setup_logging()
        handler_count = len(logging.getLogger().handlers)
        logging_setup.setup_logging()
        assert len(logging.getLogger().handlers) == handler_count


def test_setup_logging_writes_to_file(tmp_path):
    from utils import logging_setup

    cfg = _mock_log_config(tmp_path, "faceoff_test.log")
    with patch.object(logging_setup, "config", cfg):
        logging_setup.setup_logging()
        logging.getLogger("FaceOff").info("setup logging test marker")

    content = (tmp_path / "faceoff_test.log").read_text(encoding="utf-8")
    assert "setup logging test marker" in content


def test_setup_logging_attaches_terminal_handler(tmp_path):
    from utils import logging_setup
    from ui.components.terminal_tab import terminal_handler

    terminal_handler.clear()
    with patch.object(logging_setup, "config", _mock_log_config(tmp_path)):
        logging_setup.setup_logging()

    root_handlers = logging.getLogger().handlers
    assert terminal_handler in root_handlers


def test_uncaught_exception_hook_logs_critical(caplog):
    from utils import logging_setup

    with caplog.at_level(logging.CRITICAL, logger="FaceOff"):
        logging.getLogger("FaceOff").critical(
            "Uncaught exception",
            exc_info=ValueError("hook test failure"),
        )

    assert any("Uncaught exception" in record.message for record in caplog.records)
    assert any(record.exc_info is not None for record in caplog.records)