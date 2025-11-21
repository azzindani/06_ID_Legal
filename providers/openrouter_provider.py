"""
OpenRouter Provider - Multiple models via unified API
"""

from typing import Dict, Any, Optional, Generator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import BaseLLMProvider
from logger_utils import get_logger

logger = get_logger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter API provider for multiple LLM backends"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, API_TIMEOUT

        self.api_key = self.config.get('api_key', OPENROUTER_API_KEY)
        self.model_name = self.config.get('model', OPENROUTER_MODEL)
        self.timeout = self.config.get('timeout', API_TIMEOUT)
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = None

    def initialize(self) -> bool:
        """Initialize OpenRouter client (uses OpenAI SDK)"""
        try:
            from openai import OpenAI

            if not self.api_key:
                logger.error("OpenRouter API key not configured")
                return False

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )

            self._initialized = True
            logger.info(f"OpenRouter provider initialized with model: {self.model_name}")
            return True

        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate using OpenRouter API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers={
                "HTTP-Referer": "https://github.com/azzindani/06_ID_Legal",
                "X-Title": "Indonesian Legal RAG"
            }
        )

        return response.choices[0].message.content

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream using OpenRouter API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_headers={
                "HTTP-Referer": "https://github.com/azzindani/06_ID_Legal",
                "X-Title": "Indonesian Legal RAG"
            }
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat(
        self,
        messages: list,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Native chat completion"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers={
                "HTTP-Referer": "https://github.com/azzindani/06_ID_Legal",
                "X-Title": "Indonesian Legal RAG"
            }
        )

        return response.choices[0].message.content
