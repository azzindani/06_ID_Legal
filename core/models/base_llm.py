# core/models/base_llm.py
"""
Abstract base class for all LLM implementations.
Ensures consistent interface for local and API-based models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, List
from dataclasses import dataclass
from enum import Enum
from utils.logging_config import get_logger

logger = get_logger(__name__)

class LLMType(Enum):
    """Types of LLM implementations."""
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE_API = "huggingface_api"
    CUSTOM = "custom"

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 20
    min_p: float = 0.1
    repetition_penalty: float = 1.0
    do_sample: bool = True
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

@dataclass
class LLMResponse:
    """Standardized LLM response."""
    text: str
    finish_reason: str  # 'stop', 'length', 'error'
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return self.text

class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.
    
    All LLM wrappers must implement:
    - generate(): Synchronous generation
    - generate_stream(): Streaming generation
    - count_tokens(): Token counting
    - is_available(): Health check
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize LLM.
        
        Args:
            model_name: Model identifier
            config: Model configuration
        """
        self.model_name = model_name
        self.config = config
        self.llm_type: LLMType = LLMType.CUSTOM
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_generation_time': 0.0
        }
        
        logger.info(f"{self.__class__.__name__} initialized: {model_name}")
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text synchronously.
        
        Args:
            prompt: Input prompt
            generation_config: Generation parameters
            **kwargs: Additional model-specific parameters
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming.
        
        Args:
            prompt: Input prompt
            generation_config: Generation parameters
            **kwargs: Additional model-specific parameters
            
        Yields:
            Text chunks as they're generated
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if model is available and working.
        
        Returns:
            True if model is ready
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        avg_time = (
            self.stats['total_generation_time'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0
        )
        
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_requests'] / self.stats['total_requests'] * 100
                if self.stats['total_requests'] > 0 else 0
            ),
            'avg_generation_time': avg_time
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_generation_time': 0.0
        }
        logger.info("Statistics reset")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, type={self.llm_type.value})"