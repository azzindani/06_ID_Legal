"""
Rate Limiting Module

Provides flexible rate limiting for API endpoints and UI interactions.
Can be used as FastAPI middleware or standalone validator.

File: security/rate_limiting.py
"""

import time
from collections import defaultdict
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from threading import Lock
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm
    Supports per-IP, per-key, or per-user rate limiting
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: Optional[int] = None
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            requests_per_day: Max requests per day (optional)
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        
        # Store request timestamps per identifier
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
        self.day_requests: Dict[str, list] = defaultdict(list)
        
        # Thread safety
        self.lock = Lock()
        
        logger.info(f"Rate limiter initialized: {requests_per_minute}/min, {requests_per_hour}/hour")
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Check if request is allowed
        
        Args:
            identifier: Unique identifier (IP, API key, user ID)
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        with self.lock:
            now = time.time()
            
            # Clean old entries and check minute limit
            self.minute_requests[identifier] = [
                ts for ts in self.minute_requests[identifier]
                if now - ts < 60
            ]
            if len(self.minute_requests[identifier]) >= self.requests_per_minute:
                oldest = self.minute_requests[identifier][0]
                retry_after = int(60 - (now - oldest))
                return False, f"{retry_after}s"
            
            # Check hour limit
            self.hour_requests[identifier] = [
                ts for ts in self.hour_requests[identifier]
                if now - ts < 3600
            ]
            if len(self.hour_requests[identifier]) >= self.requests_per_hour:
                oldest = self.hour_requests[identifier][0]
                retry_after = int(3600 - (now - oldest))
                return False, f"{retry_after}s"
            
            # Check day limit (if configured)
            if self.requests_per_day:
                self.day_requests[identifier] = [
                    ts for ts in self.day_requests[identifier]
                    if now - ts < 86400
                ]
                if len(self.day_requests[identifier]) >= self.requests_per_day:
                    oldest = self.day_requests[identifier][0]
                    retry_after = int(86400 - (now - oldest))
                    return False, f"{retry_after}s"
            
            # Record this request
            self.minute_requests[identifier].append(now)
            self.hour_requests[identifier].append(now)
            if self.requests_per_day:
                self.day_requests[identifier].append(now)
            
            return True, None
    
    def reset(self, identifier: str):
        """Reset rate limits for an identifier"""
        with self.lock:
            self.minute_requests.pop(identifier, None)
            self.hour_requests.pop(identifier, None)
            self.day_requests.pop(identifier, None)
    
    def get_stats(self, identifier: str) -> Dict[str, int]:
        """Get current usage stats for an identifier"""
        with self.lock:
            now = time.time()
            
            return {
                'requests_last_minute': len([
                    ts for ts in self.minute_requests.get(identifier, [])
                    if now - ts < 60
                ]),
                'requests_last_hour': len([
                    ts for ts in self.hour_requests.get(identifier, [])
                    if now - ts < 3600
                ]),
                'requests_last_day': len([
                    ts for ts in self.day_requests.get(identifier, [])
                    if now - ts < 86400
                ]) if self.requests_per_day else 0,
            }


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts limits based on system load
    and user behavior patterns
    """
    
    def __init__(
        self,
        base_rpm: int = 60,
        base_rph: int = 1000,
        trusted_multiplier: float = 2.0,
        suspicious_multiplier: float = 0.5
    ):
        """
        Initialize adaptive limiter
        
        Args:
            base_rpm: Base requests per minute
            base_rph: Base requests per hour
            trusted_multiplier: Multiplier for trusted users
            suspicious_multiplier: Multiplier for suspicious users
        """
        self.base_rpm = base_rpm
        self.base_rph = base_rph
        self.trusted_multiplier = trusted_multiplier
        self.suspicious_multiplier = suspicious_multiplier
        
        # Track user trust scores (0.0 to 1.0)
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Create base limiter
        self.limiter = RateLimiter(base_rpm, base_rph)
        
        self.lock = Lock()
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """Check rate limit with adaptive adjustment"""
        with self.lock:
            trust = self.trust_scores[identifier]
            
            # Adjust limits based on trust
            if trust > 0.7:
                multiplier = self.trusted_multiplier
            elif trust < 0.3:
                multiplier = self.suspicious_multiplier
            else:
                multiplier = 1.0
            
            # Create adjusted limiter for this check
            adjusted_limiter = RateLimiter(
                requests_per_minute=int(self.base_rpm * multiplier),
                requests_per_hour=int(self.base_rph * multiplier)
            )
            
            return adjusted_limiter.check_rate_limit(identifier)
    
    def update_trust(self, identifier: str, delta: float):
        """
        Update trust score
        
        Args:
            identifier: User identifier
            delta: Change in trust (-1.0 to 1.0)
        """
        with self.lock:
            current = self.trust_scores[identifier]
            self.trust_scores[identifier] = max(0.0, min(1.0, current + delta))


# Global rate limiter instance
_global_limiter: Optional[RateLimiter] = None
_limiter_lock = Lock()


def get_limiter() -> RateLimiter:
    """Get or create the global rate limiter"""
    global _global_limiter
    with _limiter_lock:
        if _global_limiter is None:
            _global_limiter = RateLimiter()
        return _global_limiter


def check_rate_limit(identifier: str) -> Tuple[bool, Optional[str]]:
    """
    Check rate limit (convenience function)
    
    Args:
        identifier: Unique identifier
        
    Returns:
        Tuple of (is_allowed, retry_after)
    """
    limiter = get_limiter()
    return limiter.check_rate_limit(identifier)


def rate_limit_exceeded_message(retry_after: str) -> str:
    """Generate user-friendly rate limit message"""
    return (
        f"Rate limit exceeded. Please try again in {retry_after}. "
        "If you need higher limits, contact the administrator."
    )
