"""
Input Safety Module

Prevents XSS, SQL injection, prompt injection, and other malicious inputs.
Works for both API requests and UI inputs.

File: security/input_safety.py
"""

import re
from typing import List, Tuple, Optional
from utils.logger_utils import get_logger

logger = get_logger(__name__)


# Dangerous patterns that indicate potential attacks
DANGEROUS_PATTERNS = [
    # XSS patterns
    r'<script',
    r'javascript:',
    r'onerror\s*=',
    r'onclick\s*=',
    r'onload\s*=',
    r'<iframe',
    r'<object',
    r'<embed',
    
    # SQL Injection patterns
    r';\s*drop\s+table',
    r';\s*delete\s+from',
    r'union\s+select',
    r';\s*insert\s+into',
    r'--\s*$',
    
    # Command injection
    r'&&\s*rm\s+-rf',
    r';\s*rm\s+-rf',
    r'\|\s*rm\s+-rf',
    r'`.*`',
    
    # Path traversal
    r'\.\./\.\./',
    r'\.\.\\\.\.\\',
]


# Prompt injection patterns (specific to LLM systems)
PROMPT_INJECTION_PATTERNS = [
    r'ignore\s+previous\s+instructions',
    r'ignore\s+all\s+instructions',
    r'disregard\s+previous',
    r'forget\s+everything',
    r'you\s+are\s+now',
    r'new\s+role:',
    r'system\s+prompt:',
    r'override\s+instructions',
]


def check_for_injection(text: str, strict: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Check if text contains potential injection attacks
    
    Args:
        text: Text to check
        strict: If True, also check for prompt injection patterns
        
    Returns:
        Tuple of (is_safe, reason)
        - is_safe: True if safe, False if dangerous
        - reason: Description of why it's dangerous (if unsafe)
    """
    if not text:
        return True, None
    
    text_lower = text.lower()
    
    # Check dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning(f"Dangerous pattern detected: {pattern}")
            return False, f"Potentially dangerous pattern detected: {pattern}"
    
    # Check prompt injection (if strict mode)
    if strict:
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"Prompt injection detected: {pattern}")
                return False, f"Potential prompt injection detected: {pattern}"
    
    return True, None


def sanitize_query(query: str, max_length: int = 2000) -> str:
    """
    Sanitize user query for safe processing
    
    Args:
        query: Raw user query
        max_length: Maximum allowed length
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If query contains dangerous content
    """
    # Strip whitespace
    query = query.strip()
    
    # Check length
    if len(query) == 0:
        raise ValueError("Query cannot be empty")
    
    if len(query) > max_length:
        raise ValueError(f"Query exceeds maximum length of {max_length} characters")
    
    # Check for dangerous patterns
    is_safe, reason = check_for_injection(query, strict=True)
    if not is_safe:
        raise ValueError(f"Query contains potentially dangerous content: {reason}")
    
    # Remove any null bytes
    query = query.replace('\x00', '')
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query)
    
    return query


def is_safe_input(text: str, allow_html: bool = False) -> bool:
    """
    Check if input is safe (non-raising version)
    
    Args:
        text: Text to check
        allow_html: If True, allow some HTML tags
        
    Returns:
        True if safe, False otherwise
    """
    try:
        # Basic checks
        if not text or len(text) > 10000:
            return False
        
        # Check for dangerous patterns
        is_safe, _ = check_for_injection(text, strict=True)
        if not is_safe:
            return False
        
        # If HTML not allowed, check for any tags
        if not allow_html and re.search(r'<[^>]+>', text):
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking input safety: {e}")
        return False


def escape_html(text: str) -> str:
    """
    Escape HTML special characters
    
    Args:
        text: Text to escape
        
    Returns:
        HTML-safe text
    """
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&#x27;",
        "<": "&lt;",
        ">": "&gt;",
    }
    return "".join(html_escape_table.get(c, c) for c in text)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Safe filename
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Remove null bytes
    filename = filename.replace('\x00', '')
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*]', '', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > max_length:
        # Preserve extension
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        name = name[:max_length - len(ext) - 1]
        filename = f"{name}.{ext}" if ext else name
    
    # Ensure not empty
    if not filename:
        filename = 'unnamed'
    
    return filename


class InputValidator:
    """
    Configurable input validator for different contexts
    """
    
    def __init__(
        self,
        max_length: int = 2000,
        allow_html: bool = False,
        strict_injection_check: bool = True,
        custom_patterns: Optional[List[str]] = None
    ):
        """
        Initialize validator
        
        Args:
            max_length: Maximum input length
            allow_html: Allow HTML tags
            strict_injection_check: Enable strict prompt injection detection
            custom_patterns: Additional patterns to block
        """
        self.max_length = max_length
        self.allow_html = allow_html
        self.strict = strict_injection_check
        self.custom_patterns = custom_patterns or []
    
    def validate(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate input
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Length check
        if len(text) > self.max_length:
            return False, f"Input exceeds maximum length of {self.max_length}"
        
        # Injection check
        is_safe, reason = check_for_injection(text, strict=self.strict)
        if not is_safe:
            return False, reason
        
        # HTML check
        if not self.allow_html and re.search(r'<[^>]+>', text):
            return False, "HTML tags are not allowed"
        
        # Custom patterns
        for pattern in self.custom_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Input matches blocked pattern: {pattern}"
        
        return True, None
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize and validate input
        
        Raises:
            ValueError: If input is invalid
        """
        is_valid, error = self.validate(text)
        if not is_valid:
            raise ValueError(error)
        
        # Remove null bytes and normalize
        text = text.replace('\x00', '')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
