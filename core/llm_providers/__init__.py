"""
LLM Providers Module - Flexible LLM Backend Abstraction

Supports multiple LLM providers:
- OpenRouter: Cloud API gateway (200+ models)
- Local: HuggingFace transformers (Qwen VL)
- LlamaCpp: GGUF model inference (CPU/GPU hybrid)
- None: RAG-only mode (no LLM generation)

File: core/llm_providers/__init__.py
"""

from .base import LLMProviderBase
from .factory import LLMProviderFactory, get_provider
from .none import NoneProvider
from .openrouter import OpenRouterProvider
from .local import LocalProvider
from .llamacpp import LlamaCppProvider

__all__ = [
    'LLMProviderBase',
    'LLMProviderFactory',
    'get_provider',
    'NoneProvider',
    'OpenRouterProvider',
    'LocalProvider',
    'LlamaCppProvider',
]
