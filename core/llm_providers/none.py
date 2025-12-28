"""
None Provider - RAG-Only Mode

Returns informative messages when LLM generation is requested.
Useful for retrieval-only scenarios to save GPU memory.

File: core/llm_providers/none.py
"""

from typing import Dict, Any, List, Optional, Generator
from .base import LLMProviderBase

# Import logger with fallback
try:
    from utils.logger_utils import get_logger
    logger = get_logger("NoneProvider")
except ImportError:
    import logging
    logger = logging.getLogger("NoneProvider")

class NoneProvider(LLMProviderBase):
    """
    No-op provider for RAG-only mode.
    
    When LLM generation is requested, returns an informative message
    indicating that only document retrieval is available.
    
    Benefits:
    - No LLM model loaded = saves GPU memory
    - Fast startup time
    - Still allows full retrieval/search functionality
    """
    
    RAG_ONLY_MESSAGE = (
        "⚠️ **Mode RAG-Only Aktif**\n\n"
        "Generasi LLM dinonaktifkan. Anda dapat:\n"
        "- Mencari dokumen hukum\n"
        "- Melihat hasil retrieval\n"
        "- Mengekspor hasil pencarian\n\n"
        "Untuk mengaktifkan generasi jawaban, ubah provider ke 'local' atau 'openrouter' "
        "di pengaturan atau restart dengan `--llm-provider openrouter`."
    )
    
    def __init__(self):
        """Initialize None provider - no setup needed"""
        pass
    
    @property
    def provider_name(self) -> str:
        return "none"
    
    @property
    def model_name(self) -> str:
        return "none"
    
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
        """Return informative message about RAG-only mode"""
        return {
            'generated_text': self.RAG_ONLY_MESSAGE,
            'success': True,
            'error': None,
            'tokens_generated': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
            'generation_time': 0.0,
            'tokens_per_second': 0.0,
            'cost_usd': 0.0,
            'provider': 'none',
            'model': 'none',
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
        """Stream the RAG-only message token by token for UI consistency"""
        # Yield message in chunks for natural streaming feel
        words = self.RAG_ONLY_MESSAGE.split(' ')
        full_text = ""
        
        for i, word in enumerate(words):
            token = word + (' ' if i < len(words) - 1 else '')
            full_text += token
            yield {
                'token': token,
                'done': False,
                'success': True,
                'error': None,
                'tokens_generated': i + 1,
            }
        
        # Final chunk
        yield {
            'token': '',
            'done': True,
            'success': True,
            'error': None,
            'tokens_generated': len(words),
            'generation_time': 0.0,
            'tokens_per_second': 0.0,
            'full_text': full_text,
        }
    
    def is_available(self) -> bool:
        """Always available - no setup required"""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Return provider information"""
        return {
            'provider': 'none',
            'model': 'none',
            'available': True,
            'description': 'RAG-only mode - no LLM generation',
            'supports_streaming': True,
            'context_window': 0,
            'cost_per_token': 0.0,
        }
    
    def supports_streaming(self) -> bool:
        """Supports streaming for UI consistency"""
        return True
    
    def get_context_window(self) -> int:
        """No context window in RAG-only mode"""
        return 0
