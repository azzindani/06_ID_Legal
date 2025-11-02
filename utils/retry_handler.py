# utils/retry_handler.py
"""
Retry logic with exponential backoff for API calls.
"""
import time
from typing import Callable, TypeVar, Optional
from utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class RetryHandler:
    """
    Handle retries with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        
        logger.debug(
            f"RetryHandler initialized: max_retries={max_retries}, "
            f"initial_delay={initial_delay}s"
        )
    
    def execute(
        self,
        func: Callable[[], T],
        exceptions: tuple = (Exception,),
        on_retry: Optional[Callable[[int, Exception], None]] = None
    ) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            exceptions: Tuple of exceptions to catch and retry
            on_retry: Optional callback on retry (retry_count, exception)
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func()
                
                if attempt > 0:
                    logger.info(f"✅ Retry succeeded on attempt {attempt + 1}")
                
                return result
                
            except exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.initial_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(
                        f"⚠️ Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"❌ All {self.max_retries + 1} attempts failed. Last error: {e}"
                    )
        
        # All retries exhausted
        raise last_exception