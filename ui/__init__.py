"""
UI Module - Gradio Interface

Provides web-based user interface for the Indonesian Legal RAG System.

Two interfaces available:
- gradio_app: Conversational chat UI with full RAG + conversation history
- search_app: Search engine UI focused on document retrieval
"""

from .gradio_app import create_gradio_interface, launch_app

# Alias for backwards compatibility
create_demo = create_gradio_interface
from .search_app import create_search_demo, launch_search_app

__all__ = [
    'create_demo',
    'launch_app',
    'create_search_demo',
    'launch_search_app'
]
