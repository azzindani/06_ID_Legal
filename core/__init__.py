"""
Core Module - Search and Generation Components

This module provides the core RAG functionality including:
- Query detection and analysis
- Hybrid search (semantic + keyword)
- Multi-stage research with researcher personas
- Consensus building
- LLM-based generation
"""

# Lazy imports to avoid torch dependency during testing
_import_map = {
    # Search components
    'QueryDetector': ('.search.query_detection', 'QueryDetector'),
    'HybridSearchEngine': ('.search.hybrid_search', 'HybridSearchEngine'),
    'StagesResearch': ('.search.stages_research', 'StagesResearch'),
    'ConsensusBuilder': ('.search.consensus', 'ConsensusBuilder'),
    'Reranker': ('.search.reranking', 'Reranker'),
    'LangGraphOrchestrator': ('.search.langgraph_orchestrator', 'LangGraphOrchestrator'),
    # Generation components
    'LLMEngine': ('.generation.llm_engine', 'LLMEngine'),
    'GenerationEngine': ('.generation.generation_engine', 'GenerationEngine'),
    'PromptBuilder': ('.generation.prompt_builder', 'PromptBuilder'),
    'CitationFormatter': ('.generation.citation_formatter', 'CitationFormatter'),
    'ResponseValidator': ('.generation.response_validator', 'ResponseValidator'),
}

def __getattr__(name):
    if name in _import_map:
        module_path, attr_name = _import_map[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
