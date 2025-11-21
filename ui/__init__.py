"""
UI Module - Gradio Interface

Provides web-based user interface for the Indonesian Legal RAG System.
"""

from .gradio_app import create_demo, launch_app

__all__ = ['create_demo', 'launch_app']
