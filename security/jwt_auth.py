"""
JWT Authentication Module

Provides JWT token creation and verification for API authentication.
Uses PyJWT for token handling.

File: security/jwt_auth.py
"""

import os
import time
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from utils.logger_utils import get_logger

logger = get_logger(__name__)

# Try to import PyJWT, fall back to simple token if not available
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. Using simple token auth. Install with: pip install pyjwt")


# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))


# In-memory user store (for demo/development)
# In production, replace with database
_users: Dict[str, Dict[str, Any]] = {}
_user_lock = None

try:
    from threading import Lock
    _user_lock = Lock()
except ImportError:
    pass


def _hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = JWT_SECRET_KEY[:16]
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def register_user(username: str, password: str) -> Dict[str, Any]:
    """
    Register a new user
    
    Args:
        username: Unique username
        password: User password
        
    Returns:
        User info dict
        
    Raises:
        ValueError: If username already exists
    """
    if _user_lock:
        _user_lock.acquire()
    
    try:
        if username in _users:
            raise ValueError(f"Username '{username}' already exists")
        
        user = {
            "username": username,
            "password_hash": _hash_password(password),
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        _users[username] = user
        
        logger.info(f"User registered: {username}")
        return {"username": username, "created_at": user["created_at"]}
    finally:
        if _user_lock:
            _user_lock.release()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate user with username and password
    
    Args:
        username: Username
        password: Password
        
    Returns:
        User info if valid, None otherwise
    """
    user = _users.get(username)
    if not user:
        return None
    
    if user["password_hash"] != _hash_password(password):
        return None
    
    if not user.get("is_active", True):
        return None
    
    return {"username": username, "created_at": user["created_at"]}


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow()
    })
    
    if JWT_AVAILABLE:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    else:
        # Fallback: Simple token (not secure, for development only)
        token_data = f"{data.get('sub', '')}:{int(expire.timestamp())}"
        token_hash = hashlib.sha256(f"{token_data}:{JWT_SECRET_KEY}".encode()).hexdigest()
        return f"{token_data}:{token_hash}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        if JWT_AVAILABLE:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        else:
            # Fallback: Simple token verification
            parts = token.split(":")
            if len(parts) != 3:
                return None
            
            username, expire_ts, provided_hash = parts
            
            # Check expiration
            if int(expire_ts) < int(time.time()):
                return None
            
            # Verify hash
            token_data = f"{username}:{expire_ts}"
            expected_hash = hashlib.sha256(f"{token_data}:{JWT_SECRET_KEY}".encode()).hexdigest()
            
            if provided_hash != expected_hash:
                return None
            
            return {"sub": username, "exp": int(expire_ts)}
            
    except Exception as e:
        logger.debug(f"Token verification failed: {e}")
        return None


def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """
    Get current user from token
    
    Args:
        token: JWT token
        
    Returns:
        User info or None
    """
    payload = verify_token(token)
    if not payload:
        return None
    
    username = payload.get("sub")
    if not username:
        return None
    
    user = _users.get(username)
    if not user or not user.get("is_active", True):
        return None
    
    return {"username": username}


# Pre-register demo users
def _init_demo_users():
    """Initialize demo users for testing"""
    try:
        register_user("demo", "demo123")
        register_user("admin", "admin123")
        logger.info("Demo users initialized: demo, admin")
    except ValueError:
        pass  # Already registered


# Initialize demo users on module load
_init_demo_users()
