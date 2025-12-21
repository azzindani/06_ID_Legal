"""
Shared Validators for API Request Models

REFACTORED: Now uses the centralized security module
for consistent validation across all services.

File: api/validators.py
"""

import re
from typing import Optional
import sys
import os

# Add parent directory to path for security module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security import sanitize_query as security_sanitize_query
from security import is_safe_input


def validate_session_id(v: Optional[str]) -> Optional[str]:
    """
    Validate session ID format
    
    Args:
        v: Session ID to validate
        
    Returns:
        Validated session ID or None
        
    Raises:
        ValueError: If session ID contains invalid characters
    """
    if v is None:
        return v
    # Session ID should be alphanumeric with hyphens/underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', v):
        raise ValueError("Session ID must contain only alphanumeric characters, hyphens, and underscores")
    return v


def validate_query(v: str) -> str:
    """
    Enhanced validation for query input with XSS prevention
    
    Uses centralized security.input_safety module for consistency
    
    Args:
        v: Query string to validate
        
    Returns:
        Validated and stripped query string
        
    Raises:
        ValueError: If query is empty or contains dangerous patterns
    """
    # Delegate to centralized security module
    return security_sanitize_query(v)
