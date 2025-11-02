"""
Centralized Logging Configuration for Indonesian Legal RAG System
==================================================================

Provides a consistent logging infrastructure across all modules with:
- Separate console and file handlers
- Rotating file logs to prevent disk space issues
- UTF-8 encoding for Indonesian text support
- Per-module logger instances
- Daily log files with automatic cleanup

Author: Legal RAG Team
Date: 2025-01-09
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading

# Global lock for thread-safe logger creation
_logger_lock = threading.Lock()

# Cache of created loggers to avoid duplicates
_loggers = {}

# Default log directory
LOG_DIR = Path("logs")

# Log format templates
FILE_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
CONSOLE_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_DATE_FORMAT = "%H:%M:%S"


def setup_logging_directory() -> Path:
    """
    Create logs directory if it doesn't exist.
    
    Returns:
        Path: Absolute path to logs directory
        
    Raises:
        OSError: If directory cannot be created
    """
    try:
        LOG_DIR.mkdir(exist_ok=True, parents=True)
        return LOG_DIR.absolute()
    except OSError as e:
        print(f"ERROR: Failed to create logs directory: {e}")
        raise


def get_daily_log_filename(module_name: str) -> str:
    """
    Generate a daily log filename for a given module.
    
    Args:
        module_name: Name of the module (e.g., 'core.search_engine')
        
    Returns:
        str: Filename in format '{module_name}_YYYYMMDD.log'
        
    Example:
        >>> get_daily_log_filename('core.search_engine')
        'core.search_engine_20250109.log'
    """
    date_str = datetime.now().strftime("%Y%m%d")
    # Replace dots with underscores for cleaner filenames
    safe_module_name = module_name.replace(".", "_")
    return f"{safe_module_name}_{date_str}.log"


def create_console_handler(level: int = logging.INFO) -> logging.Handler:
    """
    Create a console handler with simple formatting.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        logging.Handler: Configured console handler
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Simple format for console
    console_formatter = logging.Formatter(
        fmt=CONSOLE_FORMAT,
        datefmt=CONSOLE_DATE_FORMAT
    )
    console_handler.setFormatter(console_formatter)
    
    return console_handler


def create_file_handler(
    log_file: Path, 
    level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Handler:
    """
    Create a rotating file handler with detailed formatting.
    
    Args:
        log_file: Path to log file
        level: Logging level (default: DEBUG)
        max_bytes: Maximum file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        
    Returns:
        logging.Handler: Configured rotating file handler
    """
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'  # Critical for Indonesian text
    )
    file_handler.setLevel(level)
    
    # Detailed format for file logs
    file_formatter = logging.Formatter(
        fmt=FILE_FORMAT,
        datefmt=DATE_FORMAT
    )
    file_handler.setFormatter(file_formatter)
    
    return file_handler


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Create and configure a logger with console and file handlers.
    
    Args:
        name: Logger name (typically module name)
        log_file: Optional custom log file path. If None, uses daily log file.
        level: Base logger level (default: INFO)
        console_level: Console handler level (default: INFO)
        file_level: File handler level (default: DEBUG)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logger('core.search_engine')
        >>> logger.info("Search started")
        >>> logger.debug("Processing 100 documents")
    """
    # Thread-safe logger creation
    with _logger_lock:
        # Check if logger already exists
        if name in _loggers:
            return _loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        # Add console handler
        console_handler = create_console_handler(console_level)
        logger.addHandler(console_handler)
        
        # Add file handler
        try:
            log_dir = setup_logging_directory()
            
            if log_file:
                # Use custom log file
                log_path = log_dir / log_file
            else:
                # Use daily log file
                daily_filename = get_daily_log_filename(name)
                log_path = log_dir / daily_filename
            
            file_handler = create_file_handler(log_path, file_level)
            logger.addHandler(file_handler)
            
            logger.debug(f"Logger '{name}' initialized with file: {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to create file handler: {e}")
            # Continue with console-only logging
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Cache logger
        _loggers[name] = logger
        
        return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get or create a logger for a specific module.
    
    This is the main entry point for getting loggers throughout the application.
    Use this in every module at the top level:
    
    ```python
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    ```
    
    Args:
        module_name: Name of the module (use __name__)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Function started")
        >>> logger.debug("Variable value: %s", value)
        >>> logger.error("Operation failed", exc_info=True)
    """
    if module_name in _loggers:
        return _loggers[module_name]
    
    return setup_logger(module_name)


def cleanup_old_logs(days_to_keep: int = 7) -> None:
    """
    Remove log files older than specified days.
    
    Args:
        days_to_keep: Number of days to keep logs (default: 7)
        
    Example:
        >>> cleanup_old_logs(days_to_keep=30)
    """
    try:
        log_dir = setup_logging_directory()
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)
        
        deleted_count = 0
        for log_file in log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date:
                log_file.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logger = get_logger(__name__)
            logger.info(f"Cleaned up {deleted_count} old log files (older than {days_to_keep} days)")
    
    except Exception as e:
        print(f"ERROR: Failed to cleanup logs: {e}")


# Initialize root logger on module import
_root_logger = setup_logger(
    'indonesian_legal_rag',
    log_file='main.log'
)
_root_logger.info("=" * 80)
_root_logger.info("Indonesian Legal RAG System - Logging initialized")
_root_logger.info("=" * 80)


if __name__ == "__main__":
    # Demo usage
    print("Testing Indonesian Legal RAG Logging System")
    print("=" * 80)
    
    # Test 1: Basic logging
    logger = get_logger(__name__)
    logger.info("Test 1: Basic logging")
    logger.debug("This is a debug message (file only)")
    logger.warning("This is a warning message")
    
    # Test 2: Indonesian text
    logger.info("Test 2: Indonesian text support")
    logger.info("Mencari peraturan hukum Indonesia")
    logger.debug("Regulasi: Undang-Undang No. 13 Tahun 2003")
    
    # Test 3: Exception logging
    logger.info("Test 3: Exception handling")
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.error("Caught exception", exc_info=True)
    
    # Test 4: Multiple loggers
    logger1 = get_logger('core.search_engine')
    logger2 = get_logger('generation.llm')
    
    logger1.info("Search engine initialized")
    logger2.info("LLM generator ready")
    
    # Test 5: Verify caching
    logger3 = get_logger('core.search_engine')
    assert logger1 is logger3, "Logger caching failed!"
    logger.info("Test 5: Logger caching ✓")
    
    print(f"\n✅ All tests passed!")
    print(f"📁 Check logs directory: {LOG_DIR.absolute()}")