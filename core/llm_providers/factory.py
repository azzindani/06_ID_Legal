"""
LLM Provider Factory - Creates and Manages Providers

Singleton factory for creating and switching between LLM providers.
Supports runtime provider switching with smart context transfer.

File: core/llm_providers/factory.py
"""

from typing import Dict, Any, Optional, Type
from .base import LLMProviderBase
from .none import NoneProvider
from .openrouter import OpenRouterProvider
from .local import LocalProvider
from .llamacpp import LlamaCppProvider

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("LLMProviderFactory")
except ImportError:
    import logging
    logger = logging.getLogger("LLMProviderFactory")

# Import config
try:
    from config import (
        LLM_PROVIDER,
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL,
        get_default_config,
    )
except ImportError:
    LLM_PROVIDER = "local"
    OPENROUTER_API_KEY = ""
    OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    def get_default_config():
        return {}


# Provider registry
PROVIDER_REGISTRY: Dict[str, Type[LLMProviderBase]] = {
    "local": LocalProvider,
    "openrouter": OpenRouterProvider,
    "llamacpp": LlamaCppProvider,
    "none": NoneProvider,
}


class LLMProviderFactory:
    """
    Factory for creating and managing LLM providers.
    
    Implements singleton pattern for the current provider,
    but allows runtime switching between providers.
    
    Usage:
        # Get current provider
        provider = LLMProviderFactory.get_provider()
        
        # Switch provider
        provider = LLMProviderFactory.get_provider("openrouter", api_key="...")
        
        # Force reinitialize
        provider = LLMProviderFactory.get_provider("local", force_reinit=True)
    """
    
    _instance: Optional[LLMProviderBase] = None
    _current_type: Optional[str] = None
    _config: Dict[str, Any] = {}
    
    @classmethod
    def get_provider(
        cls,
        provider_type: str = None,
        config: Dict[str, Any] = None,
        force_reinit: bool = False,
        **kwargs
    ) -> LLMProviderBase:
        """
        Get or create LLM provider.
        
        Args:
            provider_type: Provider type ('local', 'openrouter', 'none')
                          If None, uses LLM_PROVIDER from config
            config: Configuration dictionary
            force_reinit: Force reinitialization even if same type
            **kwargs: Provider-specific parameters (api_key, model, etc.)
            
        Returns:
            LLMProviderBase instance
            
        Raises:
            ValueError: If provider type is unknown
        """
        # Determine provider type
        if provider_type is None:
            provider_type = config.get('llm_provider') if config else LLM_PROVIDER
        
        provider_type = provider_type.lower()
        
        # Validate provider type
        if provider_type not in PROVIDER_REGISTRY:
            available = list(PROVIDER_REGISTRY.keys())
            raise ValueError(f"Unknown provider: {provider_type}. Available: {available}")
        
        # Return cached if same type and not forcing reinit
        if (cls._instance is not None 
            and cls._current_type == provider_type 
            and not force_reinit):
            return cls._instance
        
        # Store previous provider for context transfer
        previous_provider = cls._instance
        previous_type = cls._current_type
        
        # Create new provider
        logger.info(f"Creating {provider_type} provider")
        
        full_config = config or get_default_config()
        full_config.update(kwargs)
        
        if provider_type == "none":
            cls._instance = NoneProvider()
            
        elif provider_type == "openrouter":
            # Get API key from kwargs, config, or environment
            api_key = kwargs.get('api_key') or \
                      full_config.get('openrouter_api_key') or \
                      OPENROUTER_API_KEY
            
            if not api_key:
                raise ValueError(
                    "OpenRouter API key required. Set OPENROUTER_API_KEY "
                    "environment variable or pass api_key parameter."
                )
            
            model = kwargs.get('model') or \
                    full_config.get('openrouter_model') or \
                    OPENROUTER_MODEL
            
            cls._instance = OpenRouterProvider(
                api_key=api_key,
                model=model,
                timeout=kwargs.get('timeout', 120),
                max_retries=kwargs.get('max_retries', 3)
            )
            
        elif provider_type == "local":
            cls._instance = LocalProvider(full_config)
            
            # Auto-load model unless explicitly disabled
            if kwargs.get('auto_load', True):
                if not cls._instance.load_model():
                    logger.warning("Failed to load local model")
        
        elif provider_type == "llamacpp":
            cls._instance = LlamaCppProvider(**kwargs)
            
            # Auto-load model unless explicitly disabled
            if kwargs.get('auto_load', True):
                if not cls._instance.load_model():
                    logger.warning("Failed to load llamacpp model")
        
        else:
            # Use registry for any future providers
            provider_class = PROVIDER_REGISTRY[provider_type]
            cls._instance = provider_class(**full_config, **kwargs)
        
        cls._current_type = provider_type
        cls._config = full_config
        
        # Log switch
        if previous_type and previous_type != provider_type:
            logger.info(f"Switched provider: {previous_type} -> {provider_type}")
        
        return cls._instance
    
    @classmethod
    def get_current_provider(cls) -> Optional[LLMProviderBase]:
        """Get current provider without creating new one"""
        return cls._instance
    
    @classmethod
    def get_current_type(cls) -> Optional[str]:
        """Get current provider type"""
        return cls._current_type
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if a provider is initialized"""
        return cls._instance is not None
    
    @classmethod
    def shutdown(cls):
        """Shutdown and cleanup current provider"""
        if cls._instance is not None:
            # Unload local model if applicable
            if hasattr(cls._instance, 'unload_model'):
                cls._instance.unload_model()
            
            cls._instance = None
            cls._current_type = None
            cls._config = {}
            
            logger.info("Provider shutdown complete")
    
    @classmethod
    def list_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        List available providers with metadata.
        
        Returns:
            Dict mapping provider names to their metadata
        """
        return {
            "local": {
                "name": "Local LLM",
                "description": "HuggingFace transformers model (GPU required)",
                "requires_api_key": False,
                "cost": "Free (uses local GPU)",
            },
            "llamacpp": {
                "name": "LlamaCpp",
                "description": "GGUF model inference (CPU/GPU hybrid)",
                "requires_api_key": False,
                "cost": "Free (local inference)",
            },
            "openrouter": {
                "name": "OpenRouter",
                "description": "Cloud API gateway (200+ models)",
                "requires_api_key": True,
                "cost": "Per-token (varies by model)",
            },
            "none": {
                "name": "None",
                "description": "RAG-only mode (no LLM generation)",
                "requires_api_key": False,
                "cost": "Free",
            },
        }


# Convenience function
def get_provider(
    provider_type: str = None,
    config: Dict[str, Any] = None,
    **kwargs
) -> LLMProviderBase:
    """
    Get LLM provider instance.
    
    Convenience function wrapping LLMProviderFactory.get_provider().
    
    Args:
        provider_type: 'local', 'openrouter', or 'none'
        config: Configuration dictionary
        **kwargs: Provider-specific parameters
        
    Returns:
        LLMProviderBase instance
    """
    return LLMProviderFactory.get_provider(
        provider_type=provider_type,
        config=config,
        **kwargs
    )
