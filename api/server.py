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
from logger_utils import get_logger

logger = get_logger(__name__)

# Global instances
pipeline: Optional[RAGPipeline] = None
conversation_manager: Optional[ConversationManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global pipeline, conversation_manager

    logger.info("Starting Indonesian Legal RAG API...")

    # Initialize pipeline
    pipeline = RAGPipeline()
    if not pipeline.initialize():
        logger.error("Failed to initialize RAG pipeline")
        raise RuntimeError("Pipeline initialization failed")

    # Initialize conversation manager
    conversation_manager = ConversationManager()

    logger.info("API ready to serve requests")

    yield

    # Cleanup
    logger.info("Shutting down API...")
    if pipeline:
        pipeline.shutdown()
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

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from .routes import search_router, generate_router, health_router, session_router

    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(search_router, prefix="/api/v1", tags=["Search"])
    app.include_router(generate_router, prefix="/api/v1", tags=["Generate"])
    app.include_router(session_router, prefix="/api/v1", tags=["Session"])

    return app


def get_pipeline() -> RAGPipeline:
    """Get the global pipeline instance"""
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance"""
    if conversation_manager is None:
        raise RuntimeError("Conversation manager not initialized")
    return conversation_manager


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
