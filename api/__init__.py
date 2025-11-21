"""
API Module - FastAPI REST API

Provides REST endpoints for the Indonesian Legal RAG System.
"""

from .server import create_app, app

__all__ = ['create_app', 'app']
