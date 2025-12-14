"""
System Service Module

This module provides service functions for system initialization,
provider management, and conversation session management.

File: ui/services/system_service.py
"""

from typing import Dict, Any, Optional, Tuple
from logger_utils import get_logger

logger = get_logger(__name__)


def initialize_rag_system(
    pipeline_class,
    manager_class,
    provider_type: Optional[str] = None,
    current_provider: Optional[str] = None
) -> Tuple[Any, Any, Any, str, Dict[str, Any]]:
    """
    Initialize the RAG system with specified provider

    Args:
        pipeline_class: RAGPipeline class to instantiate
        manager_class: ConversationManager class to instantiate
        provider_type: Optional provider type to use
        current_provider: Current provider (used if provider_type not specified)

    Returns:
        Tuple of (pipeline, manager, session, provider, components_dict)
        where components_dict contains:
        {
            'search_engine': ...,
            'knowledge_graph': ...,
            'reranker': ...,
            'llm_generator': ...,
            'llm_model': ...,
            'llm_tokenizer': ...,
            'dataset_loader': ...
        }
    """
    from config import EMBEDDING_DEVICE, LLM_DEVICE

    if provider_type:
        current_provider = provider_type

    logger.info(f"Initializing RAG system with provider: {current_provider}")

    # Initialize pipeline
    pipeline = pipeline_class({'llm_provider': current_provider})
    if not pipeline.initialize():
        raise RuntimeError("Failed to initialize pipeline")

    logger.info("Pipeline initialized")

    # Extract component references for direct access
    components = {}
    if hasattr(pipeline, 'search_orchestrator'):
        components['search_engine'] = pipeline.search_orchestrator
    if hasattr(pipeline, 'knowledge_graph'):
        components['knowledge_graph'] = pipeline.knowledge_graph
    if hasattr(pipeline, 'reranker'):
        components['reranker'] = pipeline.reranker
    if hasattr(pipeline, 'generator'):
        components['llm_generator'] = pipeline.generator
    if hasattr(pipeline, 'llm_model'):
        components['llm_model'] = pipeline.llm_model
    if hasattr(pipeline, 'llm_tokenizer'):
        components['llm_tokenizer'] = pipeline.llm_tokenizer
    if hasattr(pipeline, 'data_loader'):
        components['dataset_loader'] = pipeline.data_loader

    # Initialize conversation manager
    manager = manager_class()

    # Start session
    session = manager.start_session()

    device_info = f"Embedding: {EMBEDDING_DEVICE}, LLM: {LLM_DEVICE}"
    status_message = f"Initialized with {current_provider} provider. {device_info}"

    return pipeline, manager, session, current_provider, components


def change_llm_provider(
    pipeline,
    provider_type: str,
    current_provider: str
) -> Tuple[Any, str, str]:
    """
    Switch to a different LLM provider

    Args:
        pipeline: Current RAG pipeline instance
        provider_type: New provider type to switch to
        current_provider: Current provider name

    Returns:
        Tuple of (new_pipeline, new_provider, status_message)
    """
    try:
        if pipeline:
            pipeline.shutdown()

        # Reinitialize with new provider
        from pipeline import RAGPipeline

        logger.info(f"Switching to {provider_type} provider")
        new_pipeline = RAGPipeline({'llm_provider': provider_type})

        if not new_pipeline.initialize():
            raise RuntimeError(f"Failed to initialize {provider_type} provider")

        new_provider = provider_type
        message = f"Successfully switched to {provider_type} provider"

        logger.info(message)
        return new_pipeline, new_provider, message

    except Exception as e:
        error_msg = f"Error switching provider: {e}"
        logger.error(error_msg)
        # Return original pipeline and provider on error
        return pipeline, current_provider, error_msg


def clear_conversation_session(manager) -> Tuple[Any, list, str]:
    """
    Clear conversation history and start new session

    Args:
        manager: Conversation manager instance

    Returns:
        Tuple of (new_session, empty_history, empty_string)
        for Gradio interface update
    """
    try:
        if manager:
            new_session = manager.start_session()
            return new_session, [], ""
        return None, [], ""
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return None, [], ""
