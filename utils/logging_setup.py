import logging
import sys
import warnings
from logging.handlers import RotatingFileHandler
from utils.config_manager import config


def suppress_third_party_warnings() -> None:
    """
    Suppress noisy warnings from third-party libraries.

    These warnings are from external packages we can't modify:
    - InsightFace: numpy.linalg.lstsq rcond FutureWarning
    - Torchvision: pretrained parameter deprecation warnings
    """
    # Suppress numpy lstsq FutureWarning (triggered by InsightFace)
    # "rcond parameter will change to the default of machine precision"
    warnings.filterwarnings(
        "ignore",
        message=r".*`rcond` parameter will change.*",
        category=FutureWarning
    )

    # Suppress torchvision pretrained parameter deprecation
    # "The parameter 'pretrained' is deprecated since 0.13"
    warnings.filterwarnings(
        "ignore",
        message=r".*parameter 'pretrained' is deprecated.*",
        category=UserWarning
    )

    # Suppress torchvision weights argument deprecation
    # "Arguments other than a weight enum or `None` for 'weights' are deprecated"
    warnings.filterwarnings(
        "ignore",
        message=r".*Arguments other than a weight enum.*",
        category=UserWarning
    )


def setup_logging(log_file: str = None, log_level: int = None) -> None:
    """
    Set up logging for the FaceOff project.
    Logs to both the console and a rotating log file.
    
    Uses configuration from config.yaml:
    - logging.log_file: Log file path
    - logging.max_file_size_mb: Max size before rotation (MB)
    - logging.backup_count: Number of backup files to keep
    - logging.console_level: Console log level (INFO, DEBUG, WARNING, ERROR)
    - logging.file_level: File log level (INFO, DEBUG, WARNING, ERROR)

    Args:
        log_file: Optional override for log file path (uses config if None)
        log_level: Optional override for console log level (uses config if None)
    """
    # Suppress noisy third-party library warnings first
    suppress_third_party_warnings()

    # Import terminal handler here to avoid circular imports
    try:
        from ui.components.terminal_tab import terminal_handler
        terminal_handler_available = True
    except ImportError:
        terminal_handler_available = False
    
    # Get settings from config with optional overrides
    log_file = log_file or config.log_file
    max_bytes = config.log_max_file_size_mb * 1024 * 1024  # Convert MB to bytes
    backup_count = config.log_backup_count
    
    # Parse log levels from config
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    console_level = log_level or level_map.get(config.log_console_level.upper(), logging.INFO)
    file_level = level_map.get(config.log_file_level.upper(), logging.DEBUG)
    
    # Configure rotating file handler with config settings
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Configure console handler with config settings
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Set up root logger with the most permissive level (handlers will filter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Let handlers control filtering
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Add JSON handler if enabled in config
    if config.log_json_format:
        try:
            from utils.json_formatter import create_json_handler
            json_handler = create_json_handler(
                config.log_json_file,
                level=file_level,
                include_timestamp=True,
                include_source=True
            )
            root_logger.addHandler(json_handler)
        except Exception as e:
            logging.getLogger("FaceOff").warning("Failed to enable JSON logging: %s", e)

    # Add terminal handler if available (for UI terminal tab)
    if terminal_handler_available:
        terminal_handler.setLevel(logging.DEBUG)  # Capture all logs for terminal
        root_logger.addHandler(terminal_handler)

    # Log the configuration
    logger = logging.getLogger("FaceOff")
    logger.info("Logging initialized: file=%s (level=%s, max_size=%dMB, backups=%d), console_level=%s",
                log_file, config.log_file_level.upper(), config.log_max_file_size_mb,
                backup_count, config.log_console_level.upper())

    if config.log_json_format:
        logger.info("JSON logging enabled: %s", config.log_json_file)

    if terminal_handler_available:
        logger.info("Terminal tab logging enabled - logs will appear in UI Terminal tab")
    else:
        logger.debug("Terminal tab handler not available during setup")
