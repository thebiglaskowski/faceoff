"""Central logging configuration for FaceOff."""

from __future__ import annotations

import logging
import sys
import threading
import warnings
from logging.handlers import RotatingFileHandler
from typing import Optional

from utils.config_manager import config

_CONFIGURED = False


def suppress_third_party_warnings() -> None:
    """Suppress noisy warnings from dependencies we cannot patch."""
    warnings.filterwarnings(
        "ignore",
        message=r".*`rcond` parameter will change.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*parameter 'pretrained' is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Arguments other than a weight enum.*",
        category=UserWarning,
    )


def _level_from_name(name: str, default: int) -> int:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(name.upper(), default)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt=config.log_format,
        datefmt=config.log_date_format,
    )


def _install_exception_hooks(app_logger: logging.Logger) -> None:
    """Log uncaught main-thread and worker-thread exceptions."""

    def excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        app_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = excepthook

    if hasattr(threading, "excepthook"):
        def thread_excepthook(args: threading.ExceptHookArgs) -> None:
            app_logger.critical(
                "Uncaught exception in thread %s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = thread_excepthook


def setup_logging(log_file: Optional[str] = None, log_level: Optional[int] = None) -> None:
    """
    Configure console, rotating file, optional JSON, and in-app log handlers.

    Idempotent — safe to call from main.py and ui/app.py.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    suppress_third_party_warnings()

    try:
        from ui.components.terminal_tab import terminal_handler

        terminal_handler_available = True
    except ImportError:
        terminal_handler = None
        terminal_handler_available = False

    log_file = log_file or config.log_file
    max_bytes = config.log_max_file_size_mb * 1024 * 1024
    backup_count = config.log_backup_count
    formatter = _build_formatter()

    console_level = log_level or _level_from_name(config.log_console_level, logging.INFO)
    file_level = _level_from_name(config.log_file_level, logging.DEBUG)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    if config.log_json_format:
        try:
            from utils.json_formatter import create_json_handler

            json_handler = create_json_handler(
                config.log_json_file,
                level=file_level,
                include_timestamp=True,
                include_source=True,
            )
            root_logger.addHandler(json_handler)
        except Exception as exc:
            logging.getLogger("FaceOff").warning(
                "Failed to enable JSON logging: %s", exc
            )

    if terminal_handler_available and terminal_handler is not None:
        terminal_handler.setLevel(logging.DEBUG)
        terminal_handler.setFormatter(formatter)
        root_logger.addHandler(terminal_handler)

    # Keep ONNX Runtime chatter at WARNING unless explicitly overridden.
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    # Matplotlib backend probing logs full tracebacks at DEBUG — noise in Logs tab.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    app_logger = logging.getLogger("FaceOff")
    _install_exception_hooks(app_logger)

    app_logger.info(
        "Logging initialized: file=%s (level=%s, max_size=%dMB, backups=%d), "
        "console_level=%s, terminal_buffer=%d lines",
        log_file,
        config.log_file_level.upper(),
        config.log_max_file_size_mb,
        backup_count,
        config.log_console_level.upper(),
        config.log_terminal_buffer_lines,
    )

    if config.log_json_format:
        app_logger.info("JSON logging enabled: %s", config.log_json_file)

    if terminal_handler_available:
        app_logger.info("In-app Logs tab enabled (auto-refresh every %ss)",
                        config.log_terminal_auto_refresh_sec)
    else:
        app_logger.debug("In-app terminal handler not available during setup")

    _CONFIGURED = True


def reset_logging_for_tests() -> None:
    """Reset logging state between unit tests."""
    global _CONFIGURED
    _CONFIGURED = False
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.WARNING)