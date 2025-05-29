#!/usr/bin/env python3
"""
Logging configuration for the accent detector application
"""

import logging
import sys
from pathlib import Path


def setup_logger(name=None, log_level=logging.INFO, log_file=None):
    """
    Setup and configure logger with consistent formatting
    
    Args:
        name (str): Logger name (defaults to calling module name)
        log_level (int): Logging level (default: INFO)
        log_file (str): Optional log file path
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name or __name__)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance with default configuration
    
    Args:
        name (str): Logger name (defaults to calling module name)
        
    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger(name)


# Default logger for the application
default_logger = setup_logger('accent_detector') 