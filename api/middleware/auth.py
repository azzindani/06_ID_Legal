"""
API Authentication Middleware

FastAPI middleware that enforces API Key authentication.
Uses the centralized security.authentication module.

File: api/middleware/auth.py
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from security.authentication import validate_api_key
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API keys on all /api/v1/* endpoints
    
    Exempts health check and documentation endpoints
    """
    
    def __init__(self, app: ASGIApp, exempt_paths: list = None):
        """
        Initialize middleware
        
        Args:
            app: ASGI application
            exempt_paths: List of path prefixes to exempt from auth
        """
        super().__init__(app)
        self.exempt_paths = exempt_paths or [
            '/docs',
            '/redoc',
            '/openapi.json',
            '/api/v1/health'
        ]
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and validate API key
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response or 401 error
        """
        # Check if path is exempt
        path = request.url.path
        if any(path.startswith(exempt) for exempt in self.exempt_paths):
            return await call_next(request)
        
        # Extract API key from header
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning(f"API key missing for {path}")
            return JSONResponse(
                status_code=401,
                content={
                    'error': 'Unauthorized',
                    'message': 'API key is required. Include X-API-Key header.'
                }
            )
        
        # Validate using security module
        if not validate_api_key(api_key):
            logger.warning(f"Invalid API key for {path}")
            return JSONResponse(
                status_code=401,
                content={
                    'error': 'Unauthorized',
                    'message': 'Invalid API key'
                }
            )
        
        # Key is valid, proceed
        return await call_next(request)


def require_api_key(request: Request) -> None:
    """
    Dependency function to require API key
    Can be used in specific routes
    
    Args:
        request: FastAPI request
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail='API key is required. Include X-API-Key header.'
        )
    
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail='Invalid API key'
        )
