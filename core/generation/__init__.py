"""
Generation Module for Indonesian Legal RAG System
Complete LLM-based answer generation with citations and validation

File: core/generation/__init__.py
"""

# Lazy imports to avoid torch dependency during testing
def __getattr__(name):
    if name == 'LLMEngine':
        from .llm_engine import LLMEngine
        return LLMEngine
    elif name == 'get_llm_engine':
        from .llm_engine import get_llm_engine
        return get_llm_engine
    elif name == 'PromptBuilder':
        from .prompt_builder import PromptBuilder
        return PromptBuilder
    elif name == 'CitationFormatter':
        from .citation_formatter import CitationFormatter
        return CitationFormatter
    elif name == 'ResponseValidator':
        from .response_validator import ResponseValidator
        return ResponseValidator
    elif name == 'GenerationEngine':
        from .generation_engine import GenerationEngine
        return GenerationEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'LLMEngine',
    'get_llm_engine',
    'PromptBuilder',
    'CitationFormatter',
    'ResponseValidator',
    'GenerationEngine'
]

__version__ = "1.0.0"