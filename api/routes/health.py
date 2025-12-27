"""
Health Check Routes

Endpoints for system health monitoring.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any
import time

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    components: Dict[str, str]


class ReadyResponse(BaseModel):
    ready: bool
    message: str


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Basic health check endpoint

    Returns system status and component health.

    FIXED: Uses dependency injection instead of global state import.
    """
    from ..server import get_pipeline, get_conversation_manager

    # Try to get components from app state (dependency injection)
    try:
        pipeline = get_pipeline(request)
        pipeline_status = "healthy"
    except RuntimeError:
        pipeline_status = "not_initialized"

    try:
        manager = get_conversation_manager(request)
        conversation_status = "healthy"
    except RuntimeError:
        conversation_status = "not_initialized"

    components = {
        "api": "healthy",
        "pipeline": pipeline_status,
        "conversation": conversation_status
    }

    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        components=components
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check(request: Request):
    """
    Readiness check for load balancers

    Returns whether the service is ready to accept requests.

    FIXED: Uses dependency injection instead of global state import.
    """
    from ..server import get_pipeline

    try:
        pipeline = get_pipeline(request)
        return ReadyResponse(
            ready=True,
            message="Service is ready"
        )
    except RuntimeError:
        return ReadyResponse(
            ready=False,
            message="Pipeline not initialized"
        )


@router.get("/live")
async def liveness_check():
    """
    Liveness check for container orchestration

    Simple check that the service is running.
    """
    return {"alive": True}


@router.post("/memory/cleanup")
async def cleanup_memory():
    """
    Force memory cleanup (GPU and RAM)
    
    Clears CUDA cache and runs garbage collection.
    Useful between test turns to prevent OOM.
    
    Returns:
        Memory statistics before and after cleanup
    """
    from utils.memory_utils import aggressive_cleanup, get_memory_stats
    
    # Get stats before
    before = get_memory_stats()
    
    # Run aggressive cleanup
    cleanup_stats = aggressive_cleanup(reason="API cleanup request")
    
    # Get stats after
    after = get_memory_stats()
    
    return {
        "success": True,
        "before": before,
        "after": after,
        "cleanup": cleanup_stats
    }

