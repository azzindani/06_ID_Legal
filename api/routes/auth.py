"""
Authentication API Routes

Provides user registration, login, and token management endpoints.

File: api/routes/auth.py
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional

from security.jwt_auth import (
    register_user,
    authenticate_user,
    create_access_token,
    verify_token,
    get_current_user
)
from utils.logger_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


# Request/Response models
class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    username: str
    message: str = ""


# Dependency: Get current user from token
async def get_current_user_dep(authorization: Optional[str] = Header(None)):
    """Extract and verify user from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Extract token from "Bearer <token>"
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return user


@router.post("/register", response_model=UserResponse)
async def api_register(user_data: UserRegister):
    """
    Register a new user
    
    - **username**: Unique username
    - **password**: Password (will be hashed)
    """
    try:
        result = register_user(user_data.username, user_data.password)
        logger.info(f"User registered via API: {user_data.username}")
        return UserResponse(username=result["username"], message="Registration successful")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def api_login(user_data: UserLogin):
    """
    Login and get access token
    
    - **username**: Username
    - **password**: Password
    
    Returns JWT access token valid for 60 minutes
    """
    user = authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = create_access_token(data={"sub": user["username"]})
    logger.info(f"User logged in: {user_data.username}")
    
    return TokenResponse(access_token=access_token)


@router.get("/me", response_model=UserResponse)
async def api_me(current_user: dict = Depends(get_current_user_dep)):
    """
    Get current user info
    
    Requires Authorization header with Bearer token
    """
    return UserResponse(username=current_user["username"], message="Authenticated")


@router.post("/verify")
async def api_verify_token(authorization: Optional[str] = Header(None)):
    """
    Verify if a token is valid
    
    Returns token validity status
    """
    if not authorization:
        return {"valid": False, "error": "No token provided"}
    
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    
    payload = verify_token(token)
    if payload:
        return {"valid": True, "username": payload.get("sub")}
    else:
        return {"valid": False, "error": "Invalid or expired token"}
