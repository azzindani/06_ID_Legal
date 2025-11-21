"""
LLM Providers Module

Provides unified interface for multiple LLM backends:
- Local inference (HuggingFace, quantized)
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- Google (Gemini)
- OpenRouter (multiple providers)
"""

from .base import BaseLLMProvider
from .local import LocalLLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .openrouter_provider import OpenRouterProvider
from .factory import create_provider, get_provider, list_providers, switch_provider, PROVIDERS

__all__ = [
    'BaseLLMProvider',
    'LocalLLMProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'GoogleProvider',
    'OpenRouterProvider',
    'create_provider',
    'get_provider',
    'list_providers',
    'switch_provider',
    'PROVIDERS',
]
