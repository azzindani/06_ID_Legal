"""
Generation Module for Indonesian Legal RAG System
Complete LLM-based answer generation with citations and validation

File: core/generation/__init__.py
"""

from .llm_engine import LLMEngine, get_llm_engine
from .prompt_builder import PromptBuilder
from .citation_formatter import CitationFormatter
from .response_validator import ResponseValidator
from .generation_engine import GenerationEngine

__all__ = [
    'LLMEngine',
    'get_llm_engine',
    'PromptBuilder',
    'CitationFormatter',
    'ResponseValidator',
    'GenerationEngine'
]

__version__ = "1.0.0"