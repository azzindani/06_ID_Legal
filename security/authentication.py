"""
Authentication Module

Provides API Key validation and authentication utilities.
Can be used by FastAPI middleware, Gradio UI, or any other service.

File: security/authentication.py
"""

import os
import secrets
import hashlib
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class APIKeyValidator:
    """
    Validates API keys against configured secrets.
    Supports multiple keys for different client types.
    """
    
    def __init__(self, master_key: Optional[str] = None, additional_keys: Optional[List[str]] = None):
        """
        Initialize validator with API keys
        
        Args:
            master_key: Primary API key (from .env)
            additional_keys: Additional valid keys for multi-client scenarios
        """
        self.master_key = master_key or os.getenv('LEGAL_API_KEY', '')
        self.additional_keys = additional_keys or []
        
        # Parse additional keys from environment
        env_keys = os.getenv('LEGAL_API_KEYS_ADDITIONAL', '')
        if env_keys:
            self.additional_keys.extend(env_keys.split(','))
        
        # Build valid keys set
        self.valid_keys = set()
        if self.master_key:
            self.valid_keys.add(self.master_key)
        self.valid_keys.update(self.additional_keys)
        
        if not self.valid_keys:
            logger.warning("No API keys configured! All requests will be rejected.")
        else:
            logger.info(f"API Key validator initialized with {len(self.valid_keys)} valid key(s)")
    
    def validate(self, provided_key: str) -> bool:
        """
        Validate an API key
        
        Args:
            provided_key: The key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not provided_key:
            return False
        
        # Use constant-time comparison to prevent timing attacks
        is_valid = any(
            secrets.compare_digest(provided_key, valid_key)
            for valid_key in self.valid_keys
        )
        
        if not is_valid:
            logger.warning("Invalid API key provided")
        
        return is_valid
    
    def generate_key(self, prefix: str = "legal_") -> str:
        """
        Generate a new API key
        
        Args:
            prefix: Prefix for the key
            
        Returns:
            A new secure API key
        """
        random_bytes = secrets.token_bytes(32)
        key_hash = hashlib.sha256(random_bytes).hexdigest()
        return f"{prefix}{key_hash[:40]}"


# Global validator instance (singleton pattern)
_validator: Optional[APIKeyValidator] = None


def get_validator() -> APIKeyValidator:
    """Get or create the global API key validator"""
    global _validator
    if _validator is None:
        _validator = APIKeyValidator()
    return _validator


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key (convenience function)
    
    Args:
        api_key: The key to validate
        
    Returns:
        True if valid, False otherwise
    """
    validator = get_validator()
    return validator.validate(api_key)


def require_api_key(api_key: Optional[str]) -> None:
    """
    Validate API key or raise exception
    
    Args:
        api_key: The key to validate
        
    Raises:
        ValueError: If key is missing or invalid
    """
    if not api_key:
        raise ValueError("API key is required")
    
    if not validate_api_key(api_key):
        raise ValueError("Invalid API key")


class TokenBucket:
    """
    Token bucket for per-key rate limiting
    Allows burst traffic while maintaining average rate
    """
    
    def __init__(self, capacity: int = 100, refill_rate: int = 10):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum tokens (burst size)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.now()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens available, False otherwise
        """
        # Refill tokens based on time elapsed
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class APIKeyManager:
    """
    Advanced API key management with metadata tracking
    For production environments with multiple clients
    """
    
    def __init__(self):
        self.keys: Dict[str, Dict] = {}
        self.buckets: Dict[str, TokenBucket] = {}
    
    def add_key(self, key: str, metadata: Optional[Dict] = None):
        """Add a key with metadata"""
        self.keys[key] = {
            'created_at': datetime.now(),
            'last_used': None,
            'total_requests': 0,
            'metadata': metadata or {}
        }
        self.buckets[key] = TokenBucket()
    
    def validate_and_track(self, key: str) -> bool:
        """Validate key and track usage"""
        if key not in self.keys:
            return False
        
        # Update tracking
        self.keys[key]['last_used'] = datetime.now()
        self.keys[key]['total_requests'] += 1
        
        # Check rate limit
        if key in self.buckets:
            return self.buckets[key].consume(1)
        
        return True
