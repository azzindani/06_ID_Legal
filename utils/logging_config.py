# utils/logging_config.py
"""
Centralized logging configuration for the RAG system.
Provides consistent logging across all modules.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

# ANSI color codes for console output
class LogColors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': LogColors.GRAY,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.MAGENTA,
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for file logging
        record.levelname = levelname
        return result

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    use_colors: bool = True
) -> logging.Logger:
    """
    Create a configured logger with file + console output.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (auto-generated if None)
        level: Base logging level
        console_level: Console handler level
        file_level: File handler level
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files
        use_colors: Enable colored console output
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("System started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)-30s | %(levelname)-8s | %(funcName)-25s:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    if use_colors and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%H:%M:%S'
        ))
    else:
        console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        # Create separate logs per module
        module_name = name.split('.')[-1]
        log_file = log_dir / f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(module_name: str) -> logging.Logger:
    """
    Get or create a logger for a module.
    
    Args:
        module_name: Usually __name__
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = get_logger(__name__)
    """
    return setup_logger(module_name)

# Performance logging decorator
def log_performance(logger):
    """
    Decorator to log function execution time.
    
    Example:
        >>> @log_performance(logger)
        >>> def slow_function():
        >>>     time.sleep(1)
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"⏱️  {func.__name__} started")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"✅ {func.__name__} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ {func.__name__} failed after {elapsed:.2f}s: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator

# Context manager for logging blocks
class LogBlock:
    """
    Context manager for logging code blocks.
    
    Example:
        >>> with LogBlock(logger, "Processing documents", level=logging.INFO):
        >>>     process_documents()
    """
    def __init__(self, logger, description: str, level: int = logging.INFO):
        self.logger = logger
        self.description = description
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"🔹 {self.description} - Starting...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"✅ {self.description} - Completed in {elapsed:.2f}s")
        else:
            self.logger.error(f"❌ {self.description} - Failed after {elapsed:.2f}s: {exc_val}")
        return False  # Don't suppress exceptions

import time