"""
FastAPI Server - Indonesian Legal RAG System API

Main server configuration and application factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline
from conversation import ConversationManager
from utils.logger_utils import get_logger

logger = get_logger(__name__)

# FIXED: Use application state instead of global variables for multi-worker support
# Application state is stored per-worker and properly isolated


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - FIXED: Uses app.state instead of globals"""

    logger.info("Starting Indonesian Legal RAG API...")

    # Initialize pipeline and store in app.state (worker-safe)
    app.state.pipeline = RAGPipeline()
    if not app.state.pipeline.initialize():
        logger.error("Failed to initialize RAG pipeline")
        raise RuntimeError("Pipeline initialization failed")

    # Initialize conversation manager and store in app.state
    app.state.conversation_manager = ConversationManager()

    logger.info("API ready to serve requests")

    yield

    # Cleanup
    logger.info("Shutting down API...")
    if hasattr(app.state, 'pipeline') and app.state.pipeline:
        app.state.pipeline.shutdown()
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="Indonesian Legal RAG API",
        description="REST API for Indonesian Legal Document Consultation System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # CORS middleware - SECURITY FIX: Whitelist specific origins instead of "*"
    # NOTE: allow_origins=["*"] with allow_credentials=True is a security anti-pattern
    # that enables CSRF attacks. We now use a whitelist of trusted origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:7860",  # Gradio UI default port
            "http://localhost:3000",   # Common development frontend port
            os.getenv("FRONTEND_URL", "http://localhost:8000")  # Production frontend from environment
        ],
        allow_credentials=True,  # Now safe with origin whitelist
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # FIXED: Add rate limiting middleware
    from .middleware.rate_limiter import SimpleRateLimiter
    app.add_middleware(
        SimpleRateLimiter,
        requests_per_minute=60,  # 60 requests per minute per IP
        requests_per_hour=1000   # 1000 requests per hour per IP
    )

    # Include routers
    from .routes import search_router, generate_router, health_router, session_router

    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(search_router, prefix="/api/v1", tags=["Search"])
    app.include_router(generate_router, prefix="/api/v1", tags=["Generate"])
    app.include_router(session_router, prefix="/api/v1", tags=["Session"])

    return app


# FIXED: Use dependency injection for worker-safe access
from fastapi import Request


def get_pipeline(request: Request) -> RAGPipeline:
    """Get the pipeline instance from app state (dependency injection)"""
    if not hasattr(request.app.state, 'pipeline') or request.app.state.pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return request.app.state.pipeline


def get_conversation_manager(request: Request) -> ConversationManager:
    """Get the conversation manager instance from app state (dependency injection)"""
    if not hasattr(request.app.state, 'conversation_manager') or request.app.state.conversation_manager is None:
        raise RuntimeError("Conversation manager not initialized")
    return request.app.state.conversation_manager


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
