"""
Conversational RAG Service

This module provides a reusable service for conversational RAG interactions,
separating core conversation logic from UI-specific concerns.

This service can be used by:
- Gradio UI (conversational interface)
- API endpoints (REST/GraphQL)
- CLI tools
- Tests
- Batch processing
- Other UIs

File: conversation/conversational_service.py
"""

from typing import Dict, Any, List, Optional, Callable, Generator
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class ConversationalRAGService:
    """
    Service for managing conversational RAG interactions

    This class handles the core logic of conversational RAG, including:
    - Query analysis
    - Context retrieval
    - RAG pipeline execution
    - Progress tracking via callbacks
    - Result formatting

    Different UIs can provide their own progress callbacks to handle
    updates in their own way (Gradio yields, API events, CLI prints, etc.)
    """

    def __init__(
        self,
        pipeline,
        conversation_manager,
        current_provider: str = 'local'
    ):
        """
        Initialize conversational service

        Args:
            pipeline: RAG pipeline instance
            conversation_manager: ConversationManager or MemoryManager instance
            current_provider: Current LLM provider name
        """
        self.pipeline = pipeline
        self.manager = conversation_manager
        self.current_provider = current_provider
        self.logger = logger

        # Detect if using MemoryManager or ConversationManager
        # MemoryManager has get_context() and save_turn()
        # ConversationManager has get_context_for_query() and add_turn()
        self.is_memory_manager = hasattr(conversation_manager, 'save_turn') and hasattr(conversation_manager, 'get_context')
        if self.is_memory_manager:
            self.logger.info("Using MemoryManager (unified with caching)")
        else:
            self.logger.info("Using ConversationManager (legacy mode)")

    def process_query(
        self,
        message: str,
        session_id: str,
        config_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_mode: str = 'low'
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process a conversational query with streaming support

        Args:
            message: User query text
            session_id: Conversation session ID
            config_dict: Configuration dictionary
            progress_callback: Optional callback for progress updates (progress_callback(message))
            stream_callback: Optional callback for streaming tokens (stream_callback(token))
            thinking_mode: Thinking mode ('low', 'medium', 'high')

        Yields:
            Dictionary with progress updates and results:
            {
                'type': 'progress' | 'query_analysis' | 'streaming_chunk' | 'final_result' | 'error',
                'data': {...}
            }
        """
        if not message.strip():
            yield {'type': 'error', 'data': {'error': 'Empty message'}}
            return

        try:
            # Step 1: Query Analysis
            yield {'type': 'progress', 'data': {'message': '🚀 Memulai analisis query...'}}

            query_analysis = self._analyze_query(message, progress_callback)
            if query_analysis:
                yield {'type': 'query_analysis', 'data': query_analysis}

            # Step 2: Get Conversation Context
            context = self._get_conversation_context(session_id)

            # Step 3: Execute RAG Pipeline
            # Update pipeline configuration with user settings
            if config_dict:
                self.pipeline.update_config(**config_dict)

            yield {'type': 'progress', 'data': {'message': '🔍 Conducting intelligent search...'}}

            # Show team assembly
            team_size = config_dict.get('research_team_size', 4)
            yield {'type': 'progress', 'data': {'message': f'👥 Assembling research team ({team_size} members)...'}}

            # Execute pipeline with streaming
            use_streaming = (self.current_provider == 'local')

            if use_streaming:
                # Stream results
                for chunk in self._execute_with_streaming(message, context, config_dict, stream_callback, thinking_mode):
                    yield chunk
            else:
                # Non-streaming execution
                result = self._execute_without_streaming(message, context, config_dict, thinking_mode)
                yield {'type': 'final_result', 'data': result}

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            yield {'type': 'error', 'data': {'error': str(e), 'traceback': traceback.format_exc()}}

    def _analyze_query(
        self,
        message: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze query and extract information

        Args:
            message: User query
            progress_callback: Optional progress callback

        Returns:
            Query analysis dictionary or None
        """
        try:
            if hasattr(self.pipeline, 'search_orchestrator') and \
               hasattr(self.pipeline.search_orchestrator, 'query_analyzer'):

                analysis = self.pipeline.search_orchestrator.query_analyzer.analyze_query(message)

                # Report analysis to callback
                if progress_callback and analysis:
                    strategy = analysis.get('search_strategy', 'unknown')
                    confidence = analysis.get('confidence', 0)
                    progress_callback(f"🧠 Strategy: {strategy} ({confidence:.0%})")

                    if analysis.get('reasoning'):
                        progress_callback(f"💡 {analysis['reasoning']}")

                    if analysis.get('key_phrases'):
                        phrases = [p['phrase'] for p in analysis['key_phrases']]
                        progress_callback(f"🎯 Key phrases: {', '.join(phrases)}")

                    if analysis.get('law_name_detected'):
                        law_name = analysis['specific_entities'][0]['name']
                        progress_callback(f"📜 Law name detected: {law_name}")

                return analysis

            return None

        except Exception as e:
            self.logger.debug(f"Query analysis skipped: {e}")
            return None

    def _get_conversation_context(self, session_id: str) -> Optional[List[Dict]]:
        """
        Get conversation context for the session

        Works with both MemoryManager and ConversationManager

        Args:
            session_id: Session ID

        Returns:
            List of context messages or None
        """
        try:
            if not session_id:
                self.logger.info("No session ID provided")
                return None

            # Use appropriate method based on manager type
            if self.is_memory_manager:
                # MemoryManager has get_context() with caching
                context = self.manager.get_context(session_id)
            else:
                # ConversationManager has get_context_for_query()
                context = self.manager.get_context_for_query(session_id)

            if context:
                self.logger.info(f"Using {len(context)} messages from conversation history")
            else:
                self.logger.info("No conversation context available (first message)")

            return context

        except Exception as e:
            self.logger.error(f"Error getting context: {e}")
            return None

    def _execute_with_streaming(
        self,
        message: str,
        context: Optional[List[Dict]],
        config_dict: Dict[str, Any],
        stream_callback: Optional[Callable[[str], None]] = None,
        thinking_mode: str = 'low'
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Execute RAG pipeline with streaming

        Args:
            message: User query
            context: Conversation context
            config_dict: Configuration
            stream_callback: Optional callback for streaming tokens
            thinking_mode: Thinking mode ('low', 'medium', 'high')

        Yields:
            Progress and result dictionaries
        """
        # Clear GPU cache before generation
        self._clear_gpu_cache()

        streamed_answer = ""
        chunk_count = 0
        result = None
        all_phase_metadata = {}

        try:
            # Execute pipeline with streaming generator
            for chunk in self.pipeline.query(
                question=message,
                conversation_history=context,
                stream=True,
                thinking_mode=thinking_mode
            ):
                if not isinstance(chunk, dict):
                    continue

                chunk_type = chunk.get('type', '')

                if chunk_type == 'token':
                    # Streaming token from LLM
                    token = chunk.get('token', '')
                    streamed_answer += token
                    chunk_count += 1

                    # Call stream callback
                    if stream_callback:
                        stream_callback(token)

                    # Yield streaming chunk
                    yield {
                        'type': 'streaming_chunk',
                        'data': {
                            'chunk': token,
                            'accumulated': streamed_answer,
                            'chunk_count': chunk_count
                        }
                    }

                elif chunk_type == 'thinking':
                    # Streaming thinking token (CoT) - add to streamed_answer
                    # so gradio_app.py can parse <think> tags from accumulated text
                    token = chunk.get('token', '')
                    streamed_answer += token  # Include thinking in accumulated stream
                    chunk_count += 1
                    
                    # Call stream callback for thinking tokens too
                    if stream_callback:
                        stream_callback(token)
                    
                    # Yield as streaming_chunk so gradio_app.py processes it
                    # The <think> tag parsing in gradio_app.py handles display
                    yield {
                        'type': 'streaming_chunk',
                        'data': {
                            'chunk': token,
                            'accumulated': streamed_answer,
                            'chunk_count': chunk_count,
                            'is_thinking': True  # Flag for context
                        }
                    }

                elif chunk_type == 'complete':
                    # Final result with all metadata
                    result = chunk

                    # Extract phase_metadata if present
                    if 'phase_metadata' in chunk:
                        all_phase_metadata = chunk['phase_metadata']

                    # Use the answer from pipeline (not accumulated tokens, they may differ)
                    final_answer = chunk.get('answer', streamed_answer)

                    # Build final result
                    yield {
                        'type': 'final_result',
                        'data': {
                            'success': chunk.get('success', True),
                            'answer': final_answer,
                            'thinking': chunk.get('thinking', ''),
                            'sources': chunk.get('sources', []),
                            'citations': chunk.get('citations', []),
                            'metadata': chunk.get('metadata', {}),
                            'phase_metadata': all_phase_metadata,
                            'all_retrieved_metadata': chunk.get('all_retrieved_metadata', {}),
                            'consensus_data': chunk.get('consensus_data', {}),
                            'research_data': chunk.get('research_data', {}),
                            'research_log': chunk.get('research_log', {}),
                            'communities': chunk.get('communities', []),
                            'chunk_count': chunk_count
                        }
                    }

                elif chunk_type == 'error':
                    # Error occurred
                    yield {
                        'type': 'error',
                        'data': {
                            'error': chunk.get('error', 'Unknown error'),
                            'answer': streamed_answer if streamed_answer else 'Error occurred during processing'
                        }
                    }

        finally:
            # Clean up GPU memory
            self._clear_gpu_cache()

    def _execute_without_streaming(
        self,
        message: str,
        context: Optional[List[Dict]],
        config_dict: Dict[str, Any],
        thinking_mode: str = 'low'
    ) -> Dict[str, Any]:
        """
        Execute RAG pipeline without streaming (for non-local providers)

        Args:
            message: User query
            context: Conversation context
            config_dict: Configuration
            thinking_mode: Thinking mode ('low', 'medium', 'high')

        Returns:
            Result dictionary
        """
        try:
            result = self.pipeline.query(
                question=message,
                conversation_history=context,
                stream=False,
                thinking_mode=thinking_mode
            )
            return result

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent OOM"""
        try:
            import gc
            gc.collect()

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.debug("Cleared CUDA cache")

        except Exception as e:
            self.logger.debug(f"Cache clearing skipped: {e}")

    def update_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update conversation history

        Works with both MemoryManager and ConversationManager

        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Optional metadata to store
        """
        try:
            # Use appropriate method based on manager type
            if self.is_memory_manager:
                # MemoryManager has save_turn() with auto-caching
                self.manager.save_turn(
                    session_id,
                    user_message,
                    assistant_message,
                    metadata
                )
            else:
                # ConversationManager has add_turn()
                self.manager.add_turn(
                    session_id,
                    user_message,
                    assistant_message,
                    metadata
                )
        except Exception as e:
            self.logger.error(f"Error updating conversation: {e}")


def create_conversational_service(
    pipeline,
    conversation_manager,
    current_provider: str = 'local'
) -> ConversationalRAGService:
    """
    Factory function to create a conversational service

    Args:
        pipeline: RAG pipeline instance
        conversation_manager: Conversation manager instance
        current_provider: Current LLM provider

    Returns:
        ConversationalRAGService instance
    """
    return ConversationalRAGService(
        pipeline,
        conversation_manager,
        current_provider
    )
