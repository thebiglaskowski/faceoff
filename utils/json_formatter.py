"""
JSON log formatter for structured logging output.

This module provides a JSON formatter that outputs log records
as structured JSON objects for easier parsing by log aggregation tools.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON objects.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:45.123456",
        "level": "INFO",
        "logger": "FaceOff",
        "message": "Processing started",
        "module": "orchestrator",
        "function": "process_media",
        "line": 150,
        "extra": { ... }  # Any extra fields passed to logger
    }
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_source: bool = True,
        timestamp_format: Optional[str] = None
    ):
        """
        Initialize the JSON formatter.

        Args:
            include_timestamp: Include timestamp field
            include_source: Include module/function/line fields
            timestamp_format: Custom timestamp format (ISO8601 if None)
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_source = include_source
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_dict: Dict[str, Any] = {}

        # Timestamp
        if self.include_timestamp:
            if self.timestamp_format:
                log_dict['timestamp'] = datetime.fromtimestamp(record.created).strftime(
                    self.timestamp_format
                )
            else:
                log_dict['timestamp'] = datetime.fromtimestamp(record.created).isoformat()

        # Core fields
        log_dict['level'] = record.levelname
        log_dict['logger'] = record.name
        log_dict['message'] = record.getMessage()

        # Source location
        if self.include_source:
            log_dict['module'] = record.module
            log_dict['function'] = record.funcName
            log_dict['line'] = record.lineno

        # Exception info
        if record.exc_info:
            log_dict['exception'] = self.formatException(record.exc_info)

        # Extra fields (anything added via logger.info("msg", extra={...}))
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'message', 'thread',
                'threadName', 'taskName'
            }
        }
        if extra_fields:
            log_dict['extra'] = extra_fields

        return json.dumps(log_dict, default=str)


class JSONLogHandler(logging.FileHandler):
    """
    File handler that writes JSON-formatted logs.

    Convenience class that combines FileHandler with JSONFormatter.
    """

    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        encoding: str = 'utf-8',
        **formatter_kwargs
    ):
        """
        Initialize the JSON log handler.

        Args:
            filename: Path to log file
            mode: File mode ('a' for append, 'w' for overwrite)
            encoding: File encoding
            **formatter_kwargs: Arguments passed to JSONFormatter
        """
        super().__init__(filename, mode=mode, encoding=encoding)
        self.setFormatter(JSONFormatter(**formatter_kwargs))


def create_json_handler(
    filename: str,
    level: int = logging.DEBUG,
    include_timestamp: bool = True,
    include_source: bool = True
) -> logging.Handler:
    """
    Create a JSON logging handler.

    Args:
        filename: Path to log file
        level: Minimum log level
        include_timestamp: Include timestamp in output
        include_source: Include source location in output

    Returns:
        Configured logging handler
    """
    handler = JSONLogHandler(
        filename,
        include_timestamp=include_timestamp,
        include_source=include_source
    )
    handler.setLevel(level)
    return handler
