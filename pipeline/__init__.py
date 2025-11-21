"""
Pipeline Module - High-level RAG Pipeline API

Provides simplified interface for complete RAG workflow:
query -> retrieval -> generation -> answer
"""

from .rag_pipeline import RAGPipeline

__all__ = ['RAGPipeline']
