import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_file="app.log", log_level=logging.INFO) -> None:
    """
    Set up logging for the FaceOff project.
    Logs to both the console and a rotating log file.
    """
    # Configure rotating file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Set up logging
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
