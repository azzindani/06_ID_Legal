"""
Unit Tests for LLM Providers

Tests the provider abstraction layer including:
- NoneProvider for RAG-only mode
- OpenRouterProvider for cloud API
- LocalProvider wrapper
- LLMProviderFactory for provider creation and switching

File: tests/unit/test_llm_providers.py
"""

import sys
import os
import time
import pytest
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestNoneProvider:
    """Test NoneProvider for RAG-only mode"""
    
    def test_none_provider_import(self):
        """Test that NoneProvider can be imported"""
        from core.llm_providers import NoneProvider
        assert NoneProvider is not None
    
    def test_none_provider_init(self):
        """Test NoneProvider initialization"""
        from core.llm_providers import NoneProvider
        provider = NoneProvider()
        assert provider.provider_name == "none"
        assert provider.model_name == "none"
    
    def test_none_provider_always_available(self):
        """Test that NoneProvider is always available"""
        from core.llm_providers import NoneProvider
        provider = NoneProvider()
        assert provider.is_available() is True
    
    def test_none_provider_generate_returns_message(self):
        """Test that generate returns RAG-only message"""
        from core.llm_providers import NoneProvider
        provider = NoneProvider()
        result = provider.generate("test prompt")
        
        assert result['success'] is True
        assert 'generated_text' in result
        assert 'RAG-Only' in result['generated_text'] or 'disabled' in result['generated_text'].lower()
        assert result['provider'] == 'none'
    
    def test_none_provider_stream_returns_message(self):
        """Test that generate_stream yields message chunks"""
        from core.llm_providers import NoneProvider
        provider = NoneProvider()
        
        chunks = list(provider.generate_stream("test prompt"))
        assert len(chunks) > 0
        
        # Last chunk should be done
        last_chunk = chunks[-1]
        assert last_chunk['done'] is True
        assert last_chunk['success'] is True
    
    def test_none_provider_info(self):
        """Test get_info returns correct metadata"""
        from core.llm_providers import NoneProvider
        provider = NoneProvider()
        info = provider.get_info()
        
        assert info['provider'] == 'none'
        assert info['available'] is True
        assert info['cost_per_token'] == 0.0


class TestOpenRouterProvider:
    """Test OpenRouterProvider for cloud API"""
    
    def test_openrouter_provider_import(self):
        """Test that OpenRouterProvider can be imported"""
        from core.llm_providers import OpenRouterProvider
        assert OpenRouterProvider is not None
    
    def test_openrouter_provider_init_requires_key(self):
        """Test that API key is required"""
        from core.llm_providers import OpenRouterProvider
        
        with pytest.raises(ValueError, match="API key"):
            OpenRouterProvider(api_key="")
    
    def test_openrouter_provider_init_with_key(self):
        """Test initialization with API key"""
        from core.llm_providers import OpenRouterProvider
        
        provider = OpenRouterProvider(api_key="test-key-12345")
        assert provider.provider_name == "openrouter"
        assert provider._model == "nvidia/nemotron-3-nano-30b-a3b:free"
    
    def test_openrouter_provider_custom_model(self):
        """Test initialization with custom model"""
        from core.llm_providers import OpenRouterProvider
        
        provider = OpenRouterProvider(
            api_key="test-key",
            model="anthropic/claude-sonnet-4"
        )
        assert provider.model_name == "anthropic/claude-sonnet-4"
    
    def test_openrouter_model_presets(self):
        """Test model presets are available"""
        from core.llm_providers.openrouter import get_model_presets
        
        presets = get_model_presets()
        assert 'free_default' in presets
        assert 'free_google' in presets
        assert presets['free_default'] == "nvidia/nemotron-3-nano-30b-a3b:free"
    
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="No API key")
    def test_openrouter_real_request(self):
        """Test real API request (requires OPENROUTER_API_KEY env var)"""
        from core.llm_providers import OpenRouterProvider
        
        provider = OpenRouterProvider(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="openai/gpt-4o-mini"  # Use cheap model for testing
        )
        
        result = provider.generate("Say 'hello' and nothing else.")
        assert result['success'] is True
        assert 'hello' in result['generated_text'].lower()


class TestLocalProvider:
    """Test LocalProvider wrapper"""
    
    def test_local_provider_import(self):
        """Test that LocalProvider can be imported"""
        from core.llm_providers import LocalProvider
        assert LocalProvider is not None
    
    def test_local_provider_init(self):
        """Test LocalProvider initialization (without loading model)"""
        from core.llm_providers import LocalProvider
        
        provider = LocalProvider(config={})
        assert provider.provider_name == "local"
        assert provider.is_available() is False  # Not loaded yet


class TestLLMProviderFactory:
    """Test LLMProviderFactory for provider creation"""
    
    def test_factory_import(self):
        """Test factory can be imported"""
        from core.llm_providers import LLMProviderFactory, get_provider
        assert LLMProviderFactory is not None
        assert get_provider is not None
    
    def test_factory_creates_none_provider(self):
        """Test factory creates NoneProvider"""
        from core.llm_providers import LLMProviderFactory, NoneProvider
        
        provider = LLMProviderFactory.get_provider("none")
        assert isinstance(provider, NoneProvider)
    
    def test_factory_list_providers(self):
        """Test factory lists available providers"""
        from core.llm_providers import LLMProviderFactory
        
        providers = LLMProviderFactory.list_providers()
        assert 'local' in providers
        assert 'openrouter' in providers
        assert 'none' in providers
    
    def test_factory_unknown_provider_raises(self):
        """Test factory raises for unknown provider"""
        from core.llm_providers import LLMProviderFactory
        
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.get_provider("unknown_provider")
    
    def test_factory_singleton_behavior(self):
        """Test factory returns same instance for same type"""
        from core.llm_providers import LLMProviderFactory
        
        p1 = LLMProviderFactory.get_provider("none")
        p2 = LLMProviderFactory.get_provider("none")
        assert p1 is p2
    
    def test_factory_shutdown(self):
        """Test factory shutdown clears provider"""
        from core.llm_providers import LLMProviderFactory
        
        LLMProviderFactory.get_provider("none")
        assert LLMProviderFactory.is_initialized()
        
        LLMProviderFactory.shutdown()
        assert not LLMProviderFactory.is_initialized()


class TestSecureKeyStore:
    """Test encrypted API key storage"""
    
    def test_keystore_import(self):
        """Test keystore can be imported"""
        from core.llm_providers.keystore import SecureKeyStore, get_keystore
        assert SecureKeyStore is not None
        assert get_keystore is not None
    
    def test_keystore_save_and_load(self):
        """Test saving and loading a key"""
        from core.llm_providers.keystore import SecureKeyStore
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            keystore = SecureKeyStore(storage_dir=Path(tmpdir))
            
            # Save key
            result = keystore.save_key("test_provider", "test_api_key_12345")
            assert result is True
            
            # Load key
            loaded = keystore.load_key("test_provider")
            assert loaded == "test_api_key_12345"
    
    def test_keystore_delete(self):
        """Test deleting a key"""
        from core.llm_providers.keystore import SecureKeyStore
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            keystore = SecureKeyStore(storage_dir=Path(tmpdir))
            
            keystore.save_key("test", "key123")
            keystore.delete_key("test")
            
            loaded = keystore.load_key("test")
            assert loaded is None


class TestResponseCache:
    """Test response caching"""
    
    def test_cache_import(self):
        """Test cache can be imported"""
        from core.llm_providers.cache import ResponseCache, get_response_cache
        assert ResponseCache is not None
    
    def test_cache_set_and_get(self):
        """Test caching and retrieving responses"""
        from core.llm_providers.cache import ResponseCache
        
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        
        response = {'generated_text': 'test response', 'success': True}
        cache.set("test prompt", response, model="test-model")
        
        cached = cache.get("test prompt", model="test-model")
        assert cached is not None
        assert cached['generated_text'] == 'test response'
    
    def test_cache_miss(self):
        """Test cache miss returns None"""
        from core.llm_providers.cache import ResponseCache
        
        cache = ResponseCache(max_size=10)
        result = cache.get("nonexistent prompt")
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        from core.llm_providers.cache import ResponseCache
        
        cache = ResponseCache(max_size=10)
        cache.get("miss1")
        cache.get("miss2")
        
        stats = cache.get_stats()
        assert stats['misses'] == 2
        assert stats['hits'] == 0


class TestUsageTracker:
    """Test token usage tracking"""
    
    def test_tracker_import(self):
        """Test tracker can be imported"""
        from core.llm_providers.usage_tracker import UsageTracker, get_usage_tracker
        assert UsageTracker is not None
    
    def test_tracker_record(self):
        """Test recording usage"""
        from core.llm_providers.usage_tracker import UsageTracker
        
        tracker = UsageTracker(persist=False)
        tracker.record(
            provider="test",
            model="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        stats = tracker.get_session_stats()
        assert stats['total_tokens'] == 150
        assert stats['request_count'] == 1


# Run tests with: pytest tests/unit/test_llm_providers.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
