"""
Centralized Logging System for KG-Enhanced Indonesian Legal RAG System
All modules log to a single centralized log file

File: logger_utils.py
"""

import threading
import time
import os
from datetime import datetime
from typing import List, Callable, Optional
from enum import Enum


class LogLevel(Enum):
    """Log levels for structured logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class CentralizedLogger:
    """
    Centralized logger that writes all logs to a single file
    Thread-safe singleton pattern ensures all modules use the same log file
    """
    
    _instance = None
    _lock = threading.Lock()
    _file_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return

        self.log_file = None
        self.enable_file_logging = False
        self.log_dir = "logs"
        self.append = True
        self.verbosity_mode = "minimal"  # minimal, normal, verbose
        self._initialized = True
    
    def initialize(self, enable_file_logging: bool = True,
                   log_dir: str = "logs",
                   append: bool = True,
                   log_filename: str = None,
                   verbosity_mode: str = "minimal"):
        """
        Initialize the centralized logger

        Args:
            enable_file_logging: Enable/disable file logging
            log_dir: Directory for log files
            append: If True, append to daily log. If False, create new file each run
            log_filename: Custom filename (if None, auto-generated)
            verbosity_mode: Logging verbosity ('minimal', 'normal', 'verbose')
        """
        self.enable_file_logging = enable_file_logging
        self.log_dir = log_dir
        self.append = append
        self.verbosity_mode = verbosity_mode
        
        if enable_file_logging:
            os.makedirs(log_dir, exist_ok=True)
            
            if log_filename:
                self.log_file = f"{log_dir}/{log_filename}"
            elif append:
                # Daily log file (all runs on same day append)
                date_str = datetime.now().strftime("%Y%m%d")
                self.log_file = f"{log_dir}/system_{date_str}.log"
            else:
                # Unique log file per run
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file = f"{log_dir}/system_{timestamp}.log"
            
            # Write session header
            self._write_session_header()
    
    def _write_session_header(self):
        """Write a session header to the log file"""
        try:
            with self._file_lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*100}\n")
                    f.write(f"NEW SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*100}\n\n")
        except Exception as e:
            print(f"Warning: Failed to write session header: {e}")
    
    def log(self, level: LogLevel, module: str, message: str, context: Optional[dict] = None):
        """
        Write a log entry

        Args:
            level: Log level
            module: Module name (e.g., "DataLoader", "Config", "Main")
            message: Log message
            context: Optional context dictionary
        """
        formatted_msg = self._format_message(level, module, message, context)

        # Determine if should print to console based on verbosity mode
        should_print = self._should_print_to_console(level)

        # Print to console if allowed
        if should_print:
            print(formatted_msg)

        # Always write to file if enabled (regardless of verbosity)
        if self.enable_file_logging and self.log_file:
            try:
                with self._file_lock:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(formatted_msg + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to log file: {e}")

    def _should_print_to_console(self, level: LogLevel) -> bool:
        """
        Determine if a log message should be printed to console based on verbosity mode

        Verbosity modes:
        - minimal: Only ERROR, WARNING, SUCCESS
        - normal: ERROR, WARNING, SUCCESS, INFO
        - verbose: All (ERROR, WARNING, SUCCESS, INFO, DEBUG)

        Args:
            level: Log level

        Returns:
            True if should print to console, False otherwise
        """
        # Always print critical messages
        if level in [LogLevel.ERROR, LogLevel.WARNING, LogLevel.SUCCESS]:
            return True

        # Check verbosity mode for INFO and DEBUG
        if level == LogLevel.INFO:
            return self.verbosity_mode in ['normal', 'verbose']

        if level == LogLevel.DEBUG:
            return self.verbosity_mode == 'verbose'

        # Default: print
        return True
    
    def _format_message(self, level: LogLevel, module: str, message: str, 
                       context: Optional[dict] = None) -> str:
        """Format a log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
        
        # Add emoji for console readability
        emoji_map = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️ ",
            LogLevel.WARNING: "⚠️ ",
            LogLevel.ERROR: "❌",
            LogLevel.SUCCESS: "✅"
        }
        emoji = emoji_map.get(level, "  ")
        
        formatted = f"[{timestamp}] {emoji} [{level.value:7}] [{module:20}] {message}"
        
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            formatted += f" | {context_str}"
        
        return formatted


# Global centralized logger instance
_central_logger = CentralizedLogger()


class Logger:
    """
    Module-specific logger that writes to the centralized log
    """
    
    def __init__(self, name: str):
        self.name = name
        self.central_logger = _central_logger
    
    def debug(self, message: str, context: Optional[dict] = None):
        """Log a debug message"""
        self.central_logger.log(LogLevel.DEBUG, self.name, message, context)
    
    def info(self, message: str, context: Optional[dict] = None):
        """Log an info message"""
        self.central_logger.log(LogLevel.INFO, self.name, message, context)
    
    def warning(self, message: str, context: Optional[dict] = None):
        """Log a warning message"""
        self.central_logger.log(LogLevel.WARNING, self.name, message, context)
    
    def error(self, message: str, context: Optional[dict] = None):
        """Log an error message"""
        self.central_logger.log(LogLevel.ERROR, self.name, message, context)
    
    def success(self, message: str, context: Optional[dict] = None):
        """Log a success message"""
        self.central_logger.log(LogLevel.SUCCESS, self.name, message, context)


def initialize_logging(enable_file_logging: bool = True,
                      log_dir: str = "logs",
                      append: bool = True,
                      log_filename: str = None,
                      verbosity_mode: str = "minimal"):
    """
    Initialize the centralized logging system
    CALL THIS ONCE at the start of your application

    Args:
        enable_file_logging: Enable/disable file logging
        log_dir: Directory for log files
        append: If True, append to daily log. If False, create new file each run
        log_filename: Custom filename (if None, auto-generated)
        verbosity_mode: Logging verbosity ('minimal', 'normal', 'verbose')

    Example:
        # At the start of main.py
        from logger_utils import initialize_logging
        initialize_logging(enable_file_logging=True, append=True, verbosity_mode='minimal')
    """
    _central_logger.initialize(enable_file_logging, log_dir, append, log_filename, verbosity_mode)


def get_logger(name: str) -> Logger:
    """
    Get a logger for a specific module
    All loggers write to the same centralized log file
    
    Args:
        name: Module name
        
    Returns:
        Logger instance for the module
        
    Example:
        from logger_utils import get_logger
        
        logger = get_logger("DataLoader")
        logger.info("Loading data")
    """
    return Logger(name)


class ProgressTracker:
    """Thread-safe progress tracking for research process"""
    
    def __init__(self, module_name: str = "Progress"):
        self.messages: List[str] = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.logger = get_logger(module_name)
    
    def add(self, message: str, level: LogLevel = LogLevel.INFO) -> str:
        """Add a progress message"""
        with self.lock:
            elapsed = time.time() - self.start_time
            timestamp = f"[{elapsed:.1f}s]"
            formatted_msg = f"{timestamp} {message}"
            
            self.messages.append(formatted_msg)
            
            # Log to centralized logger
            if level == LogLevel.DEBUG:
                self.logger.debug(message, {"elapsed": f"{elapsed:.1f}s"})
            elif level == LogLevel.WARNING:
                self.logger.warning(message, {"elapsed": f"{elapsed:.1f}s"})
            elif level == LogLevel.ERROR:
                self.logger.error(message, {"elapsed": f"{elapsed:.1f}s"})
            elif level == LogLevel.SUCCESS:
                self.logger.success(message, {"elapsed": f"{elapsed:.1f}s"})
            else:
                self.logger.info(message, {"elapsed": f"{elapsed:.1f}s"})
            
            return self.format()
    
    def format(self) -> str:
        """Format all messages for display"""
        with self.lock:
            return "\n".join(self.messages)
    
    def clear(self):
        """Clear all messages"""
        with self.lock:
            self.messages = []
            self.start_time = time.time()
    
    def get_duration(self) -> float:
        """Get elapsed time"""
        return time.time() - self.start_time


def create_progress_callback(messages_list: List[str], module_name: str = "Callback") -> Callable[[str], None]:
    """
    Create a progress callback function that appends to a messages list
    
    Args:
        messages_list: List to append progress messages to
        module_name: Name for the logger
        
    Returns:
        Callback function that accepts progress messages
    """
    logger = get_logger(module_name)
    
    def callback(message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        messages_list.append(formatted)
        
        # Also log to centralized logger
        logger.info(message)
    
    return callback


# Convenience functions for quick logging
def log_info(message: str, context: Optional[dict] = None, module: str = "Global"):
    """Quick info log"""
    logger = get_logger(module)
    logger.info(message, context)


def log_warning(message: str, context: Optional[dict] = None, module: str = "Global"):
    """Quick warning log"""
    logger = get_logger(module)
    logger.warning(message, context)


def log_error(message: str, context: Optional[dict] = None, module: str = "Global"):
    """Quick error log"""
    logger = get_logger(module)
    logger.error(message, context)


def log_debug(message: str, context: Optional[dict] = None, module: str = "Global"):
    """Quick debug log"""
    logger = get_logger(module)
    logger.debug(message, context)


def log_success(message: str, context: Optional[dict] = None, module: str = "Global"):
    """Quick success log"""
    logger = get_logger(module)
    logger.success(message, context)


def get_log_file_path() -> Optional[str]:
    """
    Get the path to the current log file
    
    Returns:
        Path to log file or None if file logging disabled
    """
    return _central_logger.log_file if _central_logger.enable_file_logging else None


def log_session_end():
    """Write a session end marker to the log"""
    if _central_logger.enable_file_logging and _central_logger.log_file:
        try:
            with _central_logger._file_lock:
                with open(_central_logger.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*100}\n")
                    f.write(f"SESSION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*100}\n\n")
        except Exception:
            pass