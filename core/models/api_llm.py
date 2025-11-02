# core/models/api_llm.py
"""
API-based LLM wrapper supporting multiple providers.
"""
import time
import requests
from typing import Dict, Any, Optional, Iterator
from enum import Enum

from core.models.base_llm import BaseLLM, LLMType, GenerationConfig, LLMResponse
from utils.logging_config import get_logger, log_performance
from utils.retry_handler import RetryHandler

logger = get_logger(__name__)

class APIProvider(Enum):
    """Supported API providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class APILLM(BaseLLM):
    """
    Wrapper for API-based LLMs.
    Supports OpenAI, Anthropic, and custom APIs.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        api_key: str,
        api_endpoint: Optional[str] = None,
        provider: APIProvider = APIProvider.OPENAI
    ):
        """
        Initialize API LLM.
        
        Args:
            model_name: Model identifier
            config: Configuration
            api_key: API authentication key
            api_endpoint: Custom API endpoint (optional)
            provider: API provider type
        """
        super().__init__(model_name, config)
        
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.provider = provider
        
        # Set LLM type based on provider
        if provider == APIProvider.OPENAI:
            self.llm_type = LLMType.OPENAI
            if not api_endpoint:
                self.api_endpoint = "https://api.openai.com/v1/chat/completions"
        elif provider == APIProvider.ANTHROPIC:
            self.llm_type = LLMType.ANTHROPIC
            if not api_endpoint:
                self.api_endpoint = "https://api.anthropic.com/v1/messages"
        elif provider == APIProvider.HUGGINGFACE:
            self.llm_type = LLMType.HUGGINGFACE_API
            if not api_endpoint:
                self.api_endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Retry handler for API calls
        self.retry_handler = RetryHandler(
            max_retries=3,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        )
        
        logger.info(
            f"APILLM initialized: provider={provider.value}, "
            f"model={model_name}, endpoint={self.api_endpoint}"
        )
    
    @log_performance(logger)
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text via API.
        
        Args:
            prompt: Input prompt
            generation_config: Generation parameters
            **kwargs: Additional API-specific parameters
            
        Returns:
            LLMResponse with generated text
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        logger.info(f"Generating via API: provider={self.provider.value}")
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Build request based on provider
            if self.provider == APIProvider.OPENAI:
                response = self._generate_openai(prompt, generation_config, **kwargs)
            elif self.provider == APIProvider.ANTHROPIC:
                response = self._generate_anthropic(prompt, generation_config, **kwargs)
            elif self.provider == APIProvider.HUGGINGFACE:
                response = self._generate_huggingface(prompt, generation_config, **kwargs)
            else:
                response = self._generate_custom(prompt, generation_config, **kwargs)
            
            # Update stats
            elapsed = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['total_generation_time'] += elapsed
            
            if response.tokens_used:
                self.stats['total_tokens_used'] += response.tokens_used
            
            logger.info(f"API generation complete: {elapsed:.2f}s")
            
            return response
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"API generation failed: {e}", exc_info=True)
            
            return LLMResponse(
                text="",
                finish_reason='error',
                model=self.model_name,
                metadata={'error': str(e)}
            )
    
    def _generate_openai(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        **kwargs
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        logger.debug("Using OpenAI API")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': generation_config.max_new_tokens,
            'temperature': generation_config.temperature,
            'top_p': generation_config.top_p,
            **kwargs
        }
        
        def make_request():
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        # Make request with retry
        data = self.retry_handler.execute(make_request)
        
        # Parse response
        generated_text = data['choices'][0]['message']['content']
        finish_reason = data['choices'][0]['finish_reason']
        tokens_used = data.get('usage', {}).get('total_tokens')
        
        return LLMResponse(
            text=generated_text,
            finish_reason=finish_reason,
            tokens_used=tokens_used,
            model=data.get('model', self.model_name),
            metadata=data.get('usage', {})
        )
    
    def _generate_anthropic(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        **kwargs
    ) -> LLMResponse:
        """Generate using Anthropic API."""
        logger.debug("Using Anthropic API")
        
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': generation_config.max_new_tokens,
            'temperature': generation_config.temperature,
            'top_p': generation_config.top_p,
            **kwargs
        }
        
        def make_request():
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        data = self.retry_handler.execute(make_request)
        
        # Parse response
        generated_text = data['content'][0]['text']
        finish_reason = data.get('stop_reason', 'stop')
        
        # Calculate tokens
        tokens_used = data.get('usage', {}).get('input_tokens', 0)
        tokens_used += data.get('usage', {}).get('output_tokens', 0)
        
        return LLMResponse(
            text=generated_text,
            finish_reason=finish_reason,
            tokens_used=tokens_used,
            model=data.get('model', self.model_name),
            metadata=data.get('usage', {})
        )
    
    def _generate_huggingface(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        **kwargs
    ) -> LLMResponse:
        """Generate using HuggingFace Inference API."""
        logger.debug("Using HuggingFace Inference API")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': generation_config.max_new_tokens,
                'temperature': generation_config.temperature,
                'top_p': generation_config.top_p,
                'top_k': generation_config.top_k,
                'do_sample': generation_config.do_sample,
                **kwargs
            }
        }
        
        def make_request():
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        data = self.retry_handler.execute(make_request)
        
        # Parse response
        if isinstance(data, list) and len(data) > 0:
            generated_text = data[0].get('generated_text', '')
        else:
            generated_text = str(data)
        
        return LLMResponse(
            text=generated_text,
            finish_reason='stop',
            model=self.model_name
        )
    
    def _generate_custom(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        **kwargs
    ) -> LLMResponse:
        """Generate using custom API."""
        logger.debug("Using custom API endpoint")
        
        # Generic API call - adjust as needed
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'prompt': prompt,
            'max_tokens': generation_config.max_new_tokens,
            'temperature': generation_config.temperature,
            **kwargs
        }
        
        def make_request():
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        data = self.retry_handler.execute(make_request)
        
        # Try to extract text (adjust based on your API)
        generated_text = data.get('text', data.get('output', str(data)))
        
        return LLMResponse(
            text=generated_text,
            finish_reason='stop',
            model=self.model_name
        )
    
    def generate_stream(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming (if supported by API).
        
        Note: Streaming support varies by provider.
        """
        logger.warning("Streaming not yet implemented for API LLM, using sync generation")
        
        # Fallback to synchronous generation
        response = self.generate(prompt, generation_config, **kwargs)
        yield response.text
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.
        
        Note: Exact counting requires provider-specific tokenizers.
        """
        # Rough approximation: 1 token ≈ 4 characters
        estimated = len(text) // 4
        logger.debug(f"Estimated tokens: {estimated}")
        return estimated
    
    def is_available(self) -> bool:
        """Check if API is accessible."""
        try:
            # Quick test request
            response = self.generate("test", GenerationConfig(max_new_tokens=1))
            return response.finish_reason != 'error'
        except Exception as e:
            logger.error(f"API availability check failed: {e}")
            return False