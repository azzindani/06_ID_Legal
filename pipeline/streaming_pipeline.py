"""
Streaming Pipeline - Real-time Response Generation

Provides streaming response generation for better UX.
"""

from typing import Dict, Any, List, Optional, Generator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .rag_pipeline import RAGPipeline
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class StreamingPipeline(RAGPipeline):
    """
    Streaming-optimized RAG pipeline

    Extends RAGPipeline with enhanced streaming support
    for real-time token generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.buffer_size = self.config.get('buffer_size', 10)

    def stream_query(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response tokens with metadata

        Args:
            question: User question
            conversation_history: Previous conversation

        Yields:
            Token chunks with metadata
        """
        if not self._initialized:
            yield {'type': 'error', 'content': 'Pipeline not initialized'}
            return

        try:
            # Yield search phase
            yield {'type': 'status', 'content': 'Searching documents...'}

            # Get results from parent
            for chunk in self.query(question, conversation_history, stream=True):
                if isinstance(chunk, dict):
                    yield {'type': 'final', **chunk}
                else:
                    yield {'type': 'token', 'content': chunk}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {'type': 'error', 'content': str(e)}

    def stream_with_sources(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response with source documents interleaved

        Yields source documents before the answer for transparency.
        """
        if not self._initialized:
            yield {'type': 'error', 'content': 'Pipeline not initialized'}
            return

        try:
            # First, yield that we're searching
            yield {'type': 'phase', 'content': 'retrieval'}

            # Get the full result first to extract sources
            result = self.query(question, conversation_history, stream=False)

            # Yield sources
            sources = result.get('metadata', {}).get('sources', [])
            if sources:
                yield {'type': 'sources', 'content': sources[:5]}

            # Now stream the answer
            yield {'type': 'phase', 'content': 'generation'}

            # Stream the answer character by character for demo
            # In production, you'd use actual LLM streaming
            answer = result.get('answer', '')
            buffer = ''

            for char in answer:
                buffer += char
                if len(buffer) >= self.buffer_size or char in '.!?\n':
                    yield {'type': 'token', 'content': buffer}
                    buffer = ''

            if buffer:
                yield {'type': 'token', 'content': buffer}

            # Final metadata
            yield {
                'type': 'complete',
                'metadata': result.get('metadata', {})
            }

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {'type': 'error', 'content': str(e)}


def create_streaming_pipeline(config: Optional[Dict[str, Any]] = None) -> StreamingPipeline:
    """Factory function for streaming pipeline"""
    return StreamingPipeline(config)
