"""
Base LLM Provider - Abstract interface for all providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_utils import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers

    All providers must implement this interface for plug-and-play compatibility.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.config.get('model', '')
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider (load model, setup API, etc.)

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text response

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream text response token by token

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Text chunks
        """
        pass

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Chat completion with message history

        Args:
            messages: List of {'role': 'user/assistant', 'content': str}
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Assistant response
        """
        # Default: convert to prompt
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, max_tokens, temperature, **kwargs)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream chat completion

        Args:
            messages: Message history
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Yields:
            Response chunks
        """
        prompt = self._messages_to_prompt(messages)
        yield from self.generate_stream(prompt, max_tokens, temperature, **kwargs)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert message list to prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def shutdown(self) -> None:
        """Cleanup resources"""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            'provider': self.__class__.__name__,
            'model': self.model_name,
            'initialized': self._initialized
        }
