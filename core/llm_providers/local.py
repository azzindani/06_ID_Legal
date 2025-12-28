"""
Local Provider - Wrapper for Existing LLMEngine

Wraps the existing transformers-based LLMEngine to
conform to the unified provider interface.

File: core/llm_providers/local.py
"""

from typing import Dict, Any, List, Optional, Generator
from .base import LLMProviderBase

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("LocalProvider")
except ImportError:
    import logging
    logger = logging.getLogger("LocalProvider")


class LocalProvider(LLMProviderBase):
    """
    Local LLM provider wrapping existing LLMEngine.
    
    Uses HuggingFace transformers for local inference.
    Maintains backward compatibility with existing codebase.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize local provider.
        
        Args:
            config: Configuration dictionary for LLMEngine
        """
        self.config = config or {}
        self._engine = None
        self._model_loaded = False
        
        logger.info("Local provider initialized (model not loaded yet)")
    
    @property
    def provider_name(self) -> str:
        return "local"
    
    @property
    def model_name(self) -> str:
        if self._engine:
            return self._engine.model_name
        return self.config.get('llm_model', 'unknown')
    
    def load_model(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Load the local LLM model.
        
        Args:
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            
        Returns:
            True if successful, False otherwise
        """
        if self._model_loaded and self._engine is not None:
            logger.debug("Model already loaded")
            return True
        
        try:
            # Import here to avoid circular imports and allow lazy loading
            from core.generation.llm_engine import LLMEngine
            
            logger.info("Loading local LLM model...")
            self._engine = LLMEngine(self.config)
            
            if self._engine.load_model(max_retries=max_retries, retry_delay=retry_delay):
                self._model_loaded = True
                logger.info(f"Local model loaded: {self._engine.model_name}")
                return True
            else:
                logger.error("Failed to load local model")
                return False
                
        except ImportError as e:
            logger.error(f"LLMEngine not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def unload_model(self):
        """Unload model to free memory"""
        if self._engine:
            self._engine.unload_model()
            self._engine = None
            self._model_loaded = False
            logger.info("Local model unloaded")
    
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
        Generate response using local model.
        
        Delegates to underlying LLMEngine.
        """
        if not self._model_loaded or self._engine is None:
            return {
                'generated_text': '',
                'success': False,
                'error': 'Local model not loaded. Call load_model() first.',
                'provider': self.provider_name,
                'model': self.model_name,
            }
        
        if not self.validate_prompt(prompt):
            return {
                'generated_text': '',
                'success': False,
                'error': 'Invalid or empty prompt',
                'provider': self.provider_name,
                'model': self.model_name,
            }
        
        # Call underlying engine
        result = self._engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences
        )
        
        # Add provider info
        result['provider'] = self.provider_name
        result['model'] = self.model_name
        result['cost_usd'] = 0.0  # Local = free
        
        return result
    
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
        Stream response using local model.
        
        Delegates to underlying LLMEngine's generate_stream.
        """
        if not self._model_loaded or self._engine is None:
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': 'Local model not loaded. Call load_model() first.',
            }
            return
        
        if not self.validate_prompt(prompt):
            yield {
                'token': '',
                'done': True,
                'success': False,
                'error': 'Invalid or empty prompt',
            }
            return
        
        # Stream from underlying engine
        for chunk in self._engine.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences
        ):
            yield chunk
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        return self._model_loaded and self._engine is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        info = {
            'provider': self.provider_name,
            'model': self.model_name,
            'available': self.is_available(),
            'model_loaded': self._model_loaded,
            'supports_streaming': True,
            'cost_per_token': 0.0,  # Local = free
        }
        
        if self._engine:
            engine_info = self._engine.get_model_info()
            info.update({
                'device': engine_info.get('device', 'unknown'),
                'max_length': engine_info.get('max_length', 0),
                'max_new_tokens': engine_info.get('max_new_tokens', 0),
            })
        
        return info
    
    def get_context_window(self) -> int:
        """Return context window from config"""
        if self._engine:
            return self._engine.max_length
        return self.config.get('max_length', 32768)
