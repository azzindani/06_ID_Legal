"""
LLM Provider Base Class - Abstract Interface

Defines the contract for all LLM providers ensuring consistent
interface across local, cloud, and mock providers.

File: core/llm_providers/base.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Standard result from LLM generation"""
    generated_text: str
    success: bool
    error: Optional[str] = None
    tokens_generated: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    cost_usd: float = 0.0
    provider: str = ""
    model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'generated_text': self.generated_text,
            'success': self.success,
            'error': self.error,
            'tokens_generated': self.tokens_generated,
            'prompt_tokens': self.prompt_tokens,
            'total_tokens': self.total_tokens,
            'generation_time': self.generation_time,
            'tokens_per_second': self.tokens_per_second,
            'cost_usd': self.cost_usd,
            'provider': self.provider,
            'model': self.model,
        }


@dataclass
class StreamChunk:
    """Single chunk from streaming generation"""
    token: str
    done: bool = False
    success: bool = True
    error: Optional[str] = None
    tokens_generated: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    full_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'token': self.token,
            'done': self.done,
            'success': self.success,
            'error': self.error,
            'tokens_generated': self.tokens_generated,
            'generation_time': self.generation_time,
            'tokens_per_second': self.tokens_per_second,
            'full_text': self.full_text,
        }


class LLMProviderBase(ABC):
    """
    Abstract base class for all LLM providers.
    
    Ensures consistent interface across:
    - Local models (transformers)
    - Cloud APIs (OpenRouter, OpenAI, etc.)
    - Mock/None providers (RAG-only mode)
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'openrouter', 'local', 'none')"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the current model name"""
        pass
    
    @abstractmethod
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
            prompt: Input prompt/context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences that stop generation
            **kwargs: Provider-specific parameters
            
        Returns:
            Dict with keys: generated_text, success, error, tokens_generated, etc.
        """
        pass
    
    @abstractmethod
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
        Generate response with streaming.
        
        Yields dictionaries with keys:
        - token: The new token/text chunk
        - done: Whether generation is complete
        - success: Whether this chunk was successful
        - error: Error message if any
        
        Final chunk includes:
        - full_text: Complete generated text
        - tokens_generated: Total tokens
        - generation_time: Total time
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is ready to serve requests.
        
        Returns:
            True if ready, False otherwise
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get provider metadata and status.
        
        Returns:
            Dict with provider details (name, model, loaded, config, etc.)
        """
        pass
    
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming. Default True."""
        return True
    
    def get_context_window(self) -> int:
        """Return context window size in tokens. Default 8192."""
        return 8192
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate prompt before generation.
        
        Args:
            prompt: The prompt to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not prompt or not prompt.strip():
            return False
        return True
