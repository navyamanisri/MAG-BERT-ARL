"""
MAG-BERT-ARL Logger Configuration

This module provides a beginner-friendly, clean, and centralized setup for
python's built-in logging. It outputs structured, readable logs to the console
and optionally to a log file if the log directory is available.
"""

import logging
import sys
from pathlib import Path
from backend.core.constants import LOGS_DIR, DEBUG

# Define log format
# Dynamic, highly legible layout highlighting time, severity level, module name, and message.
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str = "mag_bert_arl") -> logging.Logger:
    """
    Initializes and returns a configured logger instance.
    
    Args:
        name (str): The name of the logger, typically __name__ of the calling module.
        
    Returns:
        logging.Logger: A configured Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Set the logging level based on debug status
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logger.setLevel(log_level)
    
    # If the logger already has handlers, avoid adding duplicate ones
    if logger.hasHandlers():
        return logger
        
    # Create console handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Add handler to the logger
    logger.addHandler(console_handler)
    
    # Optional: If the LOGS_DIR exists, also log to a file for persistent records
    try:
        if LOGS_DIR:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOGS_DIR / "backend.log", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    except Exception as e:
        # Fallback gracefully if file logging cannot be initialized (e.g. permission issues)
        logger.warning(f"Could not initialize file logging: {e}. Defaulting to console-only logging.")
        
    return logger

# Create a default system-wide logger for immediate imports
logger = setup_logger("sys_main")
