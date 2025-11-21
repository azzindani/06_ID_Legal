"""
Google Provider - Gemini models via API
"""

from typing import Dict, Any, Optional, Generator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import BaseLLMProvider
from logger_utils import get_logger

logger = get_logger(__name__)


class GoogleProvider(BaseLLMProvider):
    """Google AI provider for Gemini models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        from config import GOOGLE_API_KEY, GOOGLE_MODEL

        self.api_key = self.config.get('api_key', GOOGLE_API_KEY)
        self.model_name = self.config.get('model', GOOGLE_MODEL)
        self.model = None

    def initialize(self) -> bool:
        """Initialize Google AI client"""
        try:
            import google.generativeai as genai

            if not self.api_key:
                logger.error("Google API key not configured")
                return False

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self._initialized = True
            logger.info(f"Google provider initialized with model: {self.model_name}")
            return True

        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Google AI: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate using Google AI API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        generation_config = {
            'max_output_tokens': max_tokens,
            'temperature': temperature,
        }

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )

        return response.text

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream using Google AI API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        generation_config = {
            'max_output_tokens': max_tokens,
            'temperature': temperature,
        }

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text
