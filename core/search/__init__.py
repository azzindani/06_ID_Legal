"""
Core Search Module for KG-Enhanced Indonesian Legal RAG System
Implements multi-stage, multi-persona search with LangGraph orchestration
"""

# Lazy imports to avoid torch dependency during testing
def __getattr__(name):
    if name == 'QueryDetector':
        from .query_detection import QueryDetector
        return QueryDetector
    elif name == 'HybridSearchEngine':
        from .hybrid_search import HybridSearchEngine
        return HybridSearchEngine
    elif name == 'StagesResearchEngine':
        from .stages_research import StagesResearchEngine
        return StagesResearchEngine
    elif name == 'ConsensusBuilder':
        from .consensus import ConsensusBuilder
        return ConsensusBuilder
    elif name == 'RerankerEngine':
        from .reranking import RerankerEngine
        return RerankerEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'QueryDetector',
    'HybridSearchEngine',
    'StagesResearchEngine',
    'ConsensusBuilder',
    'RerankerEngine'
]

__version__ = "1.0.0"