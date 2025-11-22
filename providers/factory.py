"""
Provider Factory - Creates LLM providers based on configuration
"""

from typing import Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import BaseLLMProvider
from .local import LocalLLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .openrouter_provider import OpenRouterProvider
from logger_utils import get_logger

logger = get_logger(__name__)

# Provider registry
PROVIDERS = {
    'local': LocalLLMProvider,
    'openai': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'google': GoogleProvider,
    'openrouter': OpenRouterProvider,
}

# Global provider instance
_current_provider: Optional[BaseLLMProvider] = None


def create_provider(
    provider_type: str = None,
    config: Optional[Dict[str, Any]] = None
) -> BaseLLMProvider:
    """
    Create an LLM provider instance

    Args:
        provider_type: Provider type (local, openai, anthropic, google, openrouter)
        config: Provider configuration

    Returns:
        Provider instance
    """
    from config import LLM_PROVIDER

    if provider_type is None:
        provider_type = LLM_PROVIDER

    provider_type = provider_type.lower()

    if provider_type not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_type}. "
            f"Available: {list(PROVIDERS.keys())}"
        )

    provider_class = PROVIDERS[provider_type]
    provider = provider_class(config)

    logger.info(f"Created {provider_type} provider")
    return provider


def get_provider(
    provider_type: str = None,
    config: Optional[Dict[str, Any]] = None,
    reinitialize: bool = False
) -> BaseLLMProvider:
    """
    Get or create the global LLM provider

    Args:
        provider_type: Provider type
        config: Provider configuration
        reinitialize: Force reinitialization

    Returns:
        Initialized provider instance
    """
    global _current_provider

    if _current_provider is not None and not reinitialize:
        return _current_provider

    # Shutdown existing provider
    if _current_provider is not None:
        _current_provider.shutdown()
        _current_provider = None

    # Create new provider
    _current_provider = create_provider(provider_type, config)

    # Initialize
    if not _current_provider.initialize():
        raise RuntimeError(f"Failed to initialize {provider_type} provider")

    return _current_provider


def list_providers() -> Dict[str, str]:
    """List available providers with descriptions"""
    return {
        'local': 'Local HuggingFace models with quantization support',
        'openai': 'OpenAI GPT models (GPT-4, GPT-3.5, etc.)',
        'anthropic': 'Anthropic Claude models',
        'google': 'Google Gemini models',
        'openrouter': 'OpenRouter (access multiple providers)'
    }


def switch_provider(
    provider_type: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseLLMProvider:
    """
    Switch to a different provider

    Args:
        provider_type: New provider type
        config: Provider configuration

    Returns:
        New provider instance
    """
    return get_provider(provider_type, config, reinitialize=True)


if __name__ == "__main__":
    print("=" * 60)
    print("PROVIDER FACTORY TEST")
    print("=" * 60)

    # List available providers
    print("\nAvailable Providers:")
    providers = list_providers()
    for name, desc in providers.items():
        print(f"  - {name}: {desc}")

    # Test provider creation (without initialization)
    print("\n" + "-" * 60)
    print("Testing Provider Creation")
    print("-" * 60)

    for provider_name in ['local', 'openai', 'anthropic', 'google', 'openrouter']:
        try:
            provider = create_provider(provider_name)
            print(f"  ✓ {provider_name}: {provider.__class__.__name__}")
        except Exception as e:
            print(f"  ✗ {provider_name}: {e}")

    # Test invalid provider
    print("\n" + "-" * 60)
    print("Testing Invalid Provider (should fail)")
    print("-" * 60)

    try:
        create_provider('invalid_provider')
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
