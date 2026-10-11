import logging
import os
import sys

def setup_logger(name: str = "evidentia", level: str = "INFO") -> logging.Logger:
    """Configures and returns an application logger with standard formatting."""
    log_level = getattr(logging, os.getenv("LOG_LEVEL", level).upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if already initialized
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()
