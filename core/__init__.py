"""
Core Module - Search and Generation Components

This module provides the core RAG functionality including:
- Query detection and analysis
- Hybrid search (semantic + keyword)
- Multi-stage research with researcher personas
- Consensus building
- LLM-based generation
"""

from .search.query_detection import QueryDetector
from .search.hybrid_search import HybridSearchEngine
from .search.stages_research import StagesResearch
from .search.consensus import ConsensusBuilder
from .search.reranking import Reranker
from .search.langgraph_orchestrator import LangGraphOrchestrator

from .generation.llm_engine import LLMEngine
from .generation.generation_engine import GenerationEngine
from .generation.prompt_builder import PromptBuilder
from .generation.citation_formatter import CitationFormatter
from .generation.response_validator import ResponseValidator

__all__ = [
    # Search components
    'QueryDetector',
    'HybridSearchEngine',
    'StagesResearch',
    'ConsensusBuilder',
    'Reranker',
    'LangGraphOrchestrator',

    # Generation components
    'LLMEngine',
    'GenerationEngine',
    'PromptBuilder',
    'CitationFormatter',
    'ResponseValidator',
]
