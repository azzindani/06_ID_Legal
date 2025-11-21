"""
Unit tests for LLM providers

Run with: pytest tests/unit/test_providers.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from providers import create_provider, list_providers, PROVIDERS
from providers.base import BaseLLMProvider
from providers.local import LocalLLMProvider
from providers.factory import PROVIDERS


@pytest.mark.unit
class TestProviderFactory:
    """Test provider factory functions"""

    def test_list_providers(self):
        """Test listing available providers"""
        providers = list_providers()

        assert 'local' in providers
        assert 'openai' in providers
        assert 'anthropic' in providers
        assert 'google' in providers
        assert 'openrouter' in providers

    def test_create_local_provider(self):
        """Test creating local provider"""
        provider = create_provider('local')

        assert provider is not None
        assert isinstance(provider, LocalLLMProvider)
        assert provider.model_name is not None

    def test_create_unknown_provider(self):
        """Test error on unknown provider"""
        with pytest.raises(ValueError) as exc_info:
            create_provider('unknown_provider')

        assert 'Unknown provider' in str(exc_info.value)

    def test_provider_config(self):
        """Test provider with custom config"""
        config = {'model': 'test-model', 'device': 'cpu'}
        provider = create_provider('local', config)

        assert provider.config.get('model') == 'test-model'


@pytest.mark.unit
class TestBaseProvider:
    """Test base provider functionality"""

    def test_messages_to_prompt(self):
        """Test message conversion to prompt"""
        from providers.local import LocalLLMProvider

        provider = LocalLLMProvider()
        messages = [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'},
            {'role': 'user', 'content': 'How are you?'}
        ]

        prompt = provider._messages_to_prompt(messages)

        assert 'System: You are helpful.' in prompt
        assert 'User: Hello' in prompt
        assert 'Assistant: Hi there' in prompt
        assert 'User: How are you?' in prompt

    def test_get_info(self):
        """Test provider info"""
        provider = create_provider('local')
        info = provider.get_info()

        assert 'provider' in info
        assert 'model' in info
        assert 'initialized' in info
        assert info['initialized'] == False


@pytest.mark.unit
class TestLocalProvider:
    """Test local provider specifics"""

    def test_quantization_config(self):
        """Test quantization configuration"""
        config = {'load_in_4bit': True, 'load_in_8bit': False}
        provider = LocalLLMProvider(config)

        assert provider.load_in_4bit == True
        assert provider.load_in_8bit == False

    def test_device_config(self):
        """Test device configuration"""
        config = {'device': 'cpu'}
        provider = LocalLLMProvider(config)

        assert provider.device == 'cpu'


@pytest.mark.unit
class TestAPIProviders:
    """Test API provider configurations"""

    def test_openai_config(self):
        """Test OpenAI provider config"""
        from providers.openai_provider import OpenAIProvider

        config = {'api_key': 'test-key', 'model': 'gpt-4'}
        provider = OpenAIProvider(config)

        assert provider.api_key == 'test-key'
        assert provider.model_name == 'gpt-4'

    def test_anthropic_config(self):
        """Test Anthropic provider config"""
        from providers.anthropic_provider import AnthropicProvider

        config = {'api_key': 'test-key', 'model': 'claude-3'}
        provider = AnthropicProvider(config)

        assert provider.api_key == 'test-key'
        assert provider.model_name == 'claude-3'

    def test_openrouter_base_url(self):
        """Test OpenRouter uses correct base URL"""
        from providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()

        assert provider.base_url == "https://openrouter.ai/api/v1"
