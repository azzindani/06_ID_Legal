"""
Core Search Module for KG-Enhanced Indonesian Legal RAG System
Implements multi-stage, multi-persona search with LangGraph orchestration
"""

from .query_detection import QueryDetector
from .hybrid_search import HybridSearchEngine
from .stages_research import StagesResearchEngine
from .consensus import ConsensusBuilder
from .reranking import RerankerEngine

__all__ = [
    'QueryDetector',
    'HybridSearchEngine', 
    'StagesResearchEngine',
    'ConsensusBuilder',
    'RerankerEngine'
]

__version__ = "1.0.0"