"""
Health Check Routes

Endpoints for system health monitoring.
"""

from fastapi import APIRouter
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
async def health_check():
    """
    Basic health check endpoint

    Returns system status and component health.
    """
    from ..server import pipeline, conversation_manager

    components = {
        "api": "healthy",
        "pipeline": "healthy" if pipeline else "not_initialized",
        "conversation": "healthy" if conversation_manager else "not_initialized"
    }

    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        components=components
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """
    Readiness check for load balancers

    Returns whether the service is ready to accept requests.
    """
    from ..server import pipeline

    if pipeline is None:
        return ReadyResponse(
            ready=False,
            message="Pipeline not initialized"
        )

    return ReadyResponse(
        ready=True,
        message="Service is ready"
    )


@router.get("/live")
async def liveness_check():
    """
    Liveness check for container orchestration

    Simple check that the service is running.
    """
    return {"alive": True}
