"""
OpenRouter Provider - Cloud LLM API Gateway

Provides access to 200+ models through OpenRouter's unified API:
- GPT-4, Claude, Gemini, DeepSeek, Qwen, etc.
- OpenAI-compatible API format
- SSE streaming support
- Token usage and cost tracking

File: core/llm_providers/openrouter.py
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional, Generator
from .base import LLMProviderBase

# Import logger - handle case where it doesn't exist yet
try:
    from utils.logger_utils import get_logger
    logger = get_logger("OpenRouterProvider")
except ImportError:
    import logging
    logger = logging.getLogger("OpenRouterProvider")


# Model presets with free models prioritized
# User-tested free models that work well:
# - nvidia/nemotron-3-nano-30b-a3b:free (default, fast, 30B)
# - deepseek/deepseek-r1-0528:free (reasoning, good for complex questions)
# - openai/gpt-oss-20b:free (smaller, faster)
MODEL_PRESETS = {
    "free_default": "nvidia/nemotron-3-nano-30b-a3b:free",
    "free_google": "google/gemini-2.0-flash-exp:free",
    "free_deepseek": "deepseek/deepseek-r1-0528:free",
    "free_openai": "openai/gpt-oss-20b:free",
    "premium_claude": "anthropic/claude-sonnet-4",
    "premium_gpt4": "openai/gpt-4o",
    "reasoning": "deepseek/deepseek-r1",
}

# Context windows for popular models (in tokens)
MODEL_CONTEXT_WINDOWS = {
    "nvidia/nemotron-3-nano-30b-a3b:free": 32768,
    "google/gemini-2.0-flash-exp:free": 1048576,  # 1M context
    "deepseek/deepseek-r1-0528:free": 65536,
    "openai/gpt-oss-20b:free": 32768,
    "anthropic/claude-sonnet-4": 200000,
    "openai/gpt-4o": 128000,
    "deepseek/deepseek-r1": 65536,
}


class OpenRouterProvider(LLMProviderBase):
    """
    OpenRouter API provider for cloud LLM access.
    
    Features:
    - OpenAI-compatible chat completions API
    - SSE streaming for real-time token output
    - Automatic retry with exponential backoff
    - Token usage and cost tracking
    - Support for all OpenRouter models (200+)
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = MODEL_PRESETS["free_default"]
    
    def __init__(
        self,
        api_key: str,
        model: str = None,
        base_url: str = None,
        timeout: int = 120,
        max_retries: int = 3,
        app_name: str = "Indonesian Legal RAG",
        app_url: str = "https://github.com/azzindani/06_ID_Legal"
    ):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (starts with sk-or-)
            model: Model ID (e.g., 'openai/gpt-4o')
            base_url: API base URL (default: OpenRouter)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts on failure
            app_name: App name for OpenRouter dashboard
            app_url: App URL for OpenRouter dashboard
        """
        # Validate API key
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("OpenRouter API key cannot be empty or whitespace")
        
        # OpenRouter keys are typically long (50+ chars)
        if len(api_key) < 20:
            raise ValueError("OpenRouter API key is too short (minimum 20 characters)")

        self.api_key = api_key
        self._model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.app_name = app_name
        self.app_url = app_url
        
        # Session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": app_url,
            "X-Title": app_name,
        })
        
        # Usage tracking
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        
        logger.info(f"OpenRouter provider initialized", {
            "model": self._model,
            "base_url": self.base_url
        })
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @model_name.setter
    def model_name(self, value: str):
        """Allow changing model at runtime"""
        self._model = value
        logger.info(f"Model changed to: {value}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response synchronously.
        
        Args:
            prompt: User message/prompt
            max_new_tokens: Max tokens (default: 4096)
            temperature: Sampling temp (default: 0.7)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            stop_sequences: Stop strings
            
        Returns:
            Dict with generated_text, tokens, cost, etc.
        """
        if not self.validate_prompt(prompt):
            return {
                'generated_text': '',
                'success': False,
                'error': 'Invalid or empty prompt',
                'provider': self.provider_name,
                'model': self._model,
            }
        
        start_time = time.time()
        
        # Build request body
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens or 4096,
            "temperature": temperature if temperature is not None else 0.7,
            "stream": False,
        }
        
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop_sequences:
            body["stop"] = stop_sequences
        
        # Add any extra kwargs
        body.update(kwargs)
        
        # Make request with retry
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    generation_time = time.time() - start_time
                    
                    # Extract response
                    generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})
                    
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    
                    # Track usage
                    self.total_tokens_used += total_tokens
                    
                    tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
                    
                    logger.info(f"Generation completed", {
                        "tokens": completion_tokens,
                        "time": f"{generation_time:.2f}s",
                        "tps": f"{tokens_per_second:.1f}"
                    })
                    
                    return {
                        'generated_text': generated_text,
                        'success': True,
                        'error': None,
                        'tokens_generated': completion_tokens,
                        'prompt_tokens': prompt_tokens,
                        'total_tokens': total_tokens,
                        'generation_time': generation_time,
                        'tokens_per_second': tokens_per_second,
                        'cost_usd': 0.0,  # Cost calculated separately
                        'provider': self.provider_name,
                        'model': self._model,
                    }
                    
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    last_error = f"API error {response.status_code}: {response.text[:200]}"
                    logger.error(last_error)
                    
            except requests.exceptions.Timeout:
                last_error = f"Request timeout after {self.timeout}s"
                logger.warning(f"Timeout on attempt {attempt}")
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                logger.error(last_error)
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(last_error)
        
        # All retries failed
        return {
            'generated_text': '',
            'success': False,
            'error': last_error,
            'provider': self.provider_name,
            'model': self._model,
        }
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate response with SSE streaming.
        
        Yields token chunks as they arrive from the API.
        Final chunk contains full_text and usage stats.
        """
        if not self.validate_prompt(prompt):
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': 'Invalid or empty prompt',
            }
            return
        
        start_time = time.time()
        tokens_generated = 0
        full_text = ""
        
        # Build request body
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens or 4096,
            "temperature": temperature if temperature is not None else 0.7,
            "stream": True,
        }
        
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop_sequences:
            body["stop"] = stop_sequences
        
        body.update(kwargs)
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=body,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code != 200:
                yield {
                    'token': '',
                    'done': True,
                    'success': False,
                    'error': f"API error {response.status_code}: {response.text[:200]}",
                }
                return
            
            # Parse SSE stream
            for line in response.iter_lines():
                if not line:
                    continue
                
                line = line.decode('utf-8')
                
                # Skip comments (OpenRouter sends these sometimes)
                if line.startswith(':'):
                    continue
                
                # Parse data line
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    # Check for stream end
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Extract token from delta
                        choices = data.get('choices', [])
                        if choices:
                            delta = choices[0].get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                full_text += content
                                tokens_generated += 1
                                
                                yield {
                                    'token': content,
                                    'done': False,
                                    'success': True,
                                    'error': None,
                                    'tokens_generated': tokens_generated,
                                }
                                
                    except json.JSONDecodeError:
                        # Ignore malformed JSON (happens occasionally)
                        continue
            
            # Final chunk with stats
            generation_time = time.time() - start_time
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            self.total_tokens_used += tokens_generated
            
            logger.info(f"Streaming completed", {
                "tokens": tokens_generated,
                "time": f"{generation_time:.2f}s",
                "tps": f"{tokens_per_second:.1f}"
            })
            
            yield {
                'token': '',
                'done': True,
                'success': True,
                'error': None,
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_second,
                'full_text': full_text,
            }
            
        except requests.exceptions.Timeout:
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': f"Stream timeout after {self.timeout}s",
                'full_text': full_text,
            }
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': str(e),
                'full_text': full_text,
            }
    
    def is_available(self) -> bool:
        """Check if OpenRouter API is reachable"""
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            'provider': self.provider_name,
            'model': self._model,
            'available': self.is_available(),
            'base_url': self.base_url,
            'supports_streaming': True,
            'context_window': self.get_context_window(),
            'total_tokens_used': self.total_tokens_used,
            'total_cost_usd': self.total_cost_usd,
        }
    
    def get_context_window(self) -> int:
        """Return context window for current model"""
        return MODEL_CONTEXT_WINDOWS.get(self._model, 32768)
    
    @classmethod
    def list_models(cls, api_key: str) -> List[Dict[str, Any]]:
        """
        Fetch available models from OpenRouter.
        
        Args:
            api_key: OpenRouter API key
            
        Returns:
            List of model dictionaries with id, name, context_length, pricing
        """
        try:
            response = requests.get(
                f"{cls.BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                
                # Sort by name for better UX
                models.sort(key=lambda m: m.get('id', ''))
                
                return models
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    @classmethod
    def get_free_models(cls, api_key: str) -> List[Dict[str, Any]]:
        """Get only free models from OpenRouter"""
        all_models = cls.list_models(api_key)
        free_models = []
        
        for model in all_models:
            pricing = model.get('pricing', {})
            # Free if both prompt and completion are 0
            if pricing.get('prompt', '0') == '0' and pricing.get('completion', '0') == '0':
                free_models.append(model)
        
        return free_models
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate OpenRouter API key.
        
        Args:
            api_key: Key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not api_key or not api_key.startswith('sk-or-'):
            return False
        
        try:
            response = requests.get(
                f"https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# Convenience function
def get_model_presets() -> Dict[str, str]:
    """Get available model presets"""
    return MODEL_PRESETS.copy()
