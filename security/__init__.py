"""
Independently Modular Security (IMS) Package

Centralized security utilities for the Indonesian Legal RAG System.
This package provides reusable security components that can be used
across all services: FastAPI, Gradio UI, CLI tools, etc.

Components:
- authentication: API Key validation and token management
- input_safety: XSS, injection, and malicious content prevention
- rate_limiting: Request throttling and abuse prevention
- file_protection: Secure file upload handling

File: security/__init__.py
"""

from .authentication import validate_api_key, APIKeyValidator
from .input_safety import sanitize_query, is_safe_input, check_for_injection
from .rate_limiting import RateLimiter, check_rate_limit
from .file_protection import validate_upload, is_safe_filename, FileValidator

__all__ = [
    # Authentication
    'validate_api_key',
    'APIKeyValidator',
    
    # Input Safety
    'sanitize_query',
    'is_safe_input',
    'check_for_injection',
    
    # Rate Limiting
    'RateLimiter',
    'check_rate_limit',
    
    # File Protection
    'validate_upload',
    'is_safe_filename',
    'FileValidator',
]

__version__ = '1.0.0'
