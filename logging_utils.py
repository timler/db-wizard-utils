import os
import sys
import logging
from datetime import datetime
from typing import Optional

def setup_logging(logger_name: str, log_level: str = 'DEBUG', log_file_prefix: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration with both file and console handlers.
    
    Args:
        logger_name: Name of the logger to create
        log_level: Logging level (default: 'DEBUG')
        log_file_prefix: Prefix for the log file name (default: None, will use logger_name)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level))
    logger.handlers = []  # Remove any existing handlers

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file_prefix = log_file_prefix or logger_name
    file_handler = logging.FileHandler(f'logs/{log_file_prefix}_{timestamp}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler (errors only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter('ERROR: %(message)s'))
    logger.addHandler(console_handler)

    return logger 