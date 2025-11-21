"""
Anthropic Provider - Claude models via API
"""

from typing import Dict, Any, Optional, Generator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import BaseLLMProvider
from logger_utils import get_logger

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider for Claude models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, API_TIMEOUT

        self.api_key = self.config.get('api_key', ANTHROPIC_API_KEY)
        self.model_name = self.config.get('model', ANTHROPIC_MODEL)
        self.timeout = self.config.get('timeout', API_TIMEOUT)
        self.client = None

    def initialize(self) -> bool:
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic

            if not self.api_key:
                logger.error("Anthropic API key not configured")
                return False

            self.client = Anthropic(api_key=self.api_key)
            self._initialized = True
            logger.info(f"Anthropic provider initialized with model: {self.model_name}")
            return True

        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate using Anthropic API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream using Anthropic API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text

    def chat(
        self,
        messages: list,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Native chat with message history"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        # Extract system message if present
        system = None
        chat_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            else:
                chat_messages.append(msg)

        create_kwargs = {
            'model': self.model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': chat_messages
        }

        if system:
            create_kwargs['system'] = system

        response = self.client.messages.create(**create_kwargs)
        return response.content[0].text
