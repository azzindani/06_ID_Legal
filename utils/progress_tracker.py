# utils/progress_tracker.py
"""
Progress tracking utilities with callbacks and logging.
"""
import time
from typing import Optional, Callable
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ProgressTracker:
    """
    Track progress of long-running operations with callbacks.
    """
    
    def __init__(
        self,
        total: int,
        description: str = "Processing",
        callback: Optional[Callable[[str], None]] = None,
        log_interval: int = 10  # Log every N%
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Operation description
            callback: Optional callback for progress updates
            log_interval: Log every N percent
        """
        self.total = total
        self.description = description
        self.callback = callback
        self.log_interval = log_interval
        
        self.current = 0
        self.start_time = time.time()
        self.last_log_percent = 0
        
        logger.info(f"Progress tracker started: {description} ({total} items)")
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """
        Update progress.
        
        Args:
            increment: Number of items processed
            message: Optional custom message
        """
        self.current += increment
        percent = (self.current / self.total * 100) if self.total > 0 else 0
        
        # Check if we should log
        if percent - self.last_log_percent >= self.log_interval:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            status = (
                f"{self.description}: {self.current:,}/{self.total:,} "
                f"({percent:.1f}%) - "
                f"Rate: {rate:.1f} items/s - "
                f"ETA: {eta:.0f}s"
            )
            
            if message:
                status += f" - {message}"
            
            logger.info(status)
            
            if self.callback:
                self.callback(status)
            
            self.last_log_percent = percent
    
    def finish(self):
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        
        status = (
            f"{self.description} complete: {self.current:,} items "
            f"in {elapsed:.1f}s ({rate:.1f} items/s)"
        )
        
        logger.info(status)
        
        if self.callback:
            self.callback(status)