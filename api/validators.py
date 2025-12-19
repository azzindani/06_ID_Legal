"""
Shared Validators for API Request Models

Centralized validation logic to avoid duplication across route handlers.
"""

import re
from typing import Optional


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

    Args:
        v: Query string to validate

    Returns:
        Validated and stripped query string

    Raises:
        ValueError: If query is empty or contains dangerous patterns
    """
    v = v.strip()
    if len(v) == 0:
        raise ValueError("Query cannot be empty or only whitespace")

    # Check for suspicious patterns (XSS prevention)
    dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=']
    v_lower = v.lower()
    for pattern in dangerous_patterns:
        if pattern in v_lower:
            raise ValueError("Query contains potentially dangerous content")

    return v
