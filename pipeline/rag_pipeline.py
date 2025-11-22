"""
RAG Pipeline - Complete Retrieval-Augmented Generation Pipeline

High-level API that orchestrates the complete RAG workflow:
1. Query analysis
2. Multi-stage retrieval with consensus
3. Reranking
4. LLM generation

File: pipeline/rag_pipeline.py
"""

import time
from typing import Dict, Any, List, Optional, Generator
from logger_utils import get_logger
from config import (
    get_default_config,
    DEFAULT_SEARCH_PHASES,
    DATASET_NAME,
    EMBEDDING_DIM,
    ENABLE_CONTEXT_CACHE
)
from conversation.context_cache import get_context_cache


class RAGPipeline:
    """
    Complete RAG Pipeline with simplified interface

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()
        result = pipeline.query("What are the labor law sanctions?")
        print(result['answer'])
        pipeline.shutdown()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG Pipeline

        Args:
            config: Optional configuration dict. If None, uses defaults.
        """
        self.logger = get_logger("RAGPipeline")

        # Merge with defaults
        self.config = get_default_config()
        if config:
            self.config.update(config)

        # Ensure search phases are set
        if 'search_phases' not in self.config:
            self.config['search_phases'] = DEFAULT_SEARCH_PHASES

        # Component references (initialized later)
        self.embedding_model = None
        self.reranker_model = None
        self.data_loader = None
        self.orchestrator = None
        self.generation_engine = None

        # State
        self._initialized = False
        self._initialization_time = 0

        # Context cache
        self._context_cache = None
        if ENABLE_CONTEXT_CACHE:
            self._context_cache = get_context_cache()

        self.logger.info("RAGPipeline created", {
            "config_keys": list(self.config.keys()),
            "context_cache_enabled": ENABLE_CONTEXT_CACHE
        })

    def initialize(self, progress_callback: Optional[callable] = None) -> bool:
        """
        Initialize all pipeline components

        Args:
            progress_callback: Optional callback for progress updates
                              Called with (step_name, step_number, total_steps)

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            self.logger.warning("Pipeline already initialized")
            return True

        self.logger.info("Initializing RAG Pipeline...")
        start_time = time.time()

        total_steps = 5

        try:
            # Step 1: Load embedding and reranker models
            if progress_callback:
                progress_callback("Loading models", 1, total_steps)

            self.logger.info("Step 1/5: Loading models...")
            from model_manager import load_models
            self.embedding_model, self.reranker_model = load_models()

            if self.embedding_model is None or self.reranker_model is None:
                self.logger.error("Failed to load models")
                return False

            # Step 2: Load dataset
            if progress_callback:
                progress_callback("Loading dataset", 2, total_steps)

            self.logger.info("Step 2/5: Loading dataset...")
            from loader.dataloader import EnhancedKGDatasetLoader

            self.data_loader = EnhancedKGDatasetLoader(
                DATASET_NAME,
                EMBEDDING_DIM
            )
            self.data_loader.load_from_huggingface(
                progress_callback=lambda msg: self.logger.debug(f"Dataset: {msg}")
            )

            stats = self.data_loader.get_statistics()
            self.logger.info("Dataset loaded", {
                "total_records": stats.get('total_records', 0),
                "kg_enhanced": stats.get('kg_enhanced_records', 0)
            })

            # Step 3: Create RAG orchestrator
            if progress_callback:
                progress_callback("Creating orchestrator", 3, total_steps)

            self.logger.info("Step 3/5: Creating RAG orchestrator...")
            from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator

            self.orchestrator = LangGraphRAGOrchestrator(
                data_loader=self.data_loader,
                embedding_model=self.embedding_model,
                reranker_model=self.reranker_model,
                config=self.config
            )

            # Step 4: Initialize generation engine
            if progress_callback:
                progress_callback("Initializing LLM", 4, total_steps)

            self.logger.info("Step 4/5: Initializing generation engine...")
            from core.generation.generation_engine import GenerationEngine

            self.generation_engine = GenerationEngine(self.config)
            if not self.generation_engine.initialize():
                self.logger.error("Failed to initialize generation engine")
                return False

            # Step 5: Finalize
            if progress_callback:
                progress_callback("Finalizing", 5, total_steps)

            self._initialized = True
            self._initialization_time = time.time() - start_time

            self.logger.success("RAG Pipeline initialized successfully", {
                "initialization_time": f"{self._initialization_time:.2f}s"
            })

            return True

        except Exception as e:
            self.logger.error("Pipeline initialization failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()
            })
            return False

    def query(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete RAG query

        Args:
            question: User question
            conversation_history: Optional conversation context
            stream: Whether to stream the response

        Returns:
            Dictionary with:
                - success: bool
                - answer: str (final answer)
                - sources: List[Dict] (source documents)
                - metadata: Dict (timing, scores, etc.)
                - error: str (if failed)
        """
        if not self._initialized:
            return {
                'success': False,
                'error': 'Pipeline not initialized. Call initialize() first.',
                'answer': '',
                'sources': [],
                'metadata': {}
            }

        self.logger.info("Processing query", {
            "question_length": len(question),
            "has_history": conversation_history is not None,
            "stream": stream
        })

        start_time = time.time()

        try:
            # Check context cache for similar recent queries
            cache_key = None
            if self._context_cache:
                cache_key = f"query:{hash(question)}"
                cached = self._context_cache.get(cache_key)
                if cached and cached.get('answer'):
                    self.logger.info("Cache hit for query")
                    cached['metadata']['from_cache'] = True
                    return cached

            # Step 1: Run RAG orchestrator (retrieval)
            self.logger.info("Running retrieval...")
            retrieval_start = time.time()

            rag_result = self.orchestrator.run(
                query=question,
                conversation_history=conversation_history or []
            )

            retrieval_time = time.time() - retrieval_start

            final_results = rag_result.get('final_results', [])

            self.logger.info("Retrieval completed", {
                "results_count": len(final_results),
                "retrieval_time": f"{retrieval_time:.2f}s"
            })

            if not final_results:
                self.logger.warning("No results retrieved")
                return {
                    'success': True,
                    'answer': 'Maaf, tidak ditemukan dokumen yang relevan untuk pertanyaan Anda.',
                    'sources': [],
                    'metadata': {
                        'retrieval_time': retrieval_time,
                        'total_time': time.time() - start_time,
                        'results_count': 0
                    }
                }

            # Step 2: Generate answer
            self.logger.info("Generating answer...")

            query_analysis = rag_result.get('metadata', {}).get('query_analysis', {})

            if stream:
                return self._generate_streaming(
                    question=question,
                    retrieved_results=final_results,
                    query_analysis=query_analysis,
                    conversation_history=conversation_history,
                    retrieval_time=retrieval_time,
                    start_time=start_time
                )
            else:
                generation_result = self.generation_engine.generate_answer(
                    query=question,
                    retrieved_results=final_results,
                    query_analysis=query_analysis,
                    conversation_history=conversation_history,
                    stream=False
                )

                total_time = time.time() - start_time

                if generation_result['success']:
                    self.logger.success("Query completed successfully", {
                        "total_time": f"{total_time:.2f}s",
                        "answer_length": len(generation_result['answer'])
                    })

                    result = {
                        'success': True,
                        'answer': generation_result['answer'],
                        'sources': generation_result.get('sources', []),
                        'citations': generation_result.get('citations', []),
                        'metadata': {
                            'question': question,
                            'retrieval_time': retrieval_time,
                            'generation_time': generation_result['metadata'].get('generation_time', 0),
                            'total_time': total_time,
                            'results_count': len(final_results),
                            'tokens_generated': generation_result['metadata'].get('tokens_generated', 0),
                            'query_type': query_analysis.get('query_type', 'general'),
                            'rag_metadata': rag_result.get('metadata', {}),
                            'from_cache': False
                        }
                    }

                    # Store in cache
                    if self._context_cache and cache_key:
                        self._context_cache.put(cache_key, result)

                    return result
                else:
                    self.logger.error("Generation failed", {
                        "error": generation_result.get('error')
                    })
                    return {
                        'success': False,
                        'error': generation_result.get('error', 'Generation failed'),
                        'answer': '',
                        'sources': [],
                        'metadata': {
                            'retrieval_time': retrieval_time,
                            'total_time': time.time() - start_time
                        }
                    }

        except Exception as e:
            self.logger.error("Query execution failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()
            })

            return {
                'success': False,
                'error': str(e),
                'answer': '',
                'sources': [],
                'metadata': {
                    'total_time': time.time() - start_time
                }
            }

    def _generate_streaming(
        self,
        question: str,
        retrieved_results: List[Dict],
        query_analysis: Dict,
        conversation_history: Optional[List],
        retrieval_time: float,
        start_time: float
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming response"""

        self.logger.info("Starting streaming generation")

        try:
            for chunk in self.generation_engine.generate_answer(
                query=question,
                retrieved_results=retrieved_results,
                query_analysis=query_analysis,
                conversation_history=conversation_history,
                stream=True
            ):
                if chunk.get('type') == 'token':
                    yield {
                        'type': 'token',
                        'token': chunk['token'],
                        'done': False
                    }
                elif chunk.get('type') == 'complete':
                    total_time = time.time() - start_time
                    yield {
                        'type': 'complete',
                        'success': True,
                        'answer': chunk['answer'],
                        'sources': [],  # TODO: format sources
                        'metadata': {
                            'question': question,
                            'retrieval_time': retrieval_time,
                            'generation_time': chunk.get('generation_time', 0),
                            'total_time': total_time,
                            'results_count': len(retrieved_results),
                            'tokens_generated': chunk.get('tokens_generated', 0)
                        },
                        'done': True
                    }
                elif chunk.get('type') == 'error':
                    yield {
                        'type': 'error',
                        'error': chunk.get('error'),
                        'done': True
                    }

        except Exception as e:
            self.logger.error("Streaming generation failed", {
                "error": str(e)
            })
            yield {
                'type': 'error',
                'error': str(e),
                'done': True
            }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information and status"""

        info = {
            'initialized': self._initialized,
            'initialization_time': self._initialization_time,
            'config': {
                'final_top_k': self.config.get('final_top_k'),
                'max_rounds': self.config.get('max_rounds'),
                'consensus_threshold': self.config.get('consensus_threshold'),
                'temperature': self.config.get('temperature'),
                'max_new_tokens': self.config.get('max_new_tokens')
            }
        }

        if self._initialized:
            info['dataset_stats'] = self.data_loader.get_statistics()
            info['generation_info'] = self.generation_engine.get_engine_info()

        return info

    def update_config(self, **kwargs):
        """
        Update pipeline configuration

        Args:
            **kwargs: Configuration key-value pairs to update
        """
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                self.logger.info(f"Config updated: {key}", {
                    "old": old_value,
                    "new": value
                })
            else:
                self.logger.warning(f"Unknown config key: {key}")

    def shutdown(self):
        """Shutdown pipeline and cleanup resources"""

        self.logger.info("Shutting down RAG Pipeline...")

        try:
            if self.generation_engine:
                self.generation_engine.shutdown()

            # Clear references
            self.embedding_model = None
            self.reranker_model = None
            self.data_loader = None
            self.orchestrator = None
            self.generation_engine = None

            self._initialized = False

            # Clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.success("RAG Pipeline shut down successfully")

        except Exception as e:
            self.logger.error("Error during shutdown", {
                "error": str(e)
            })

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False


# Convenience function
def create_pipeline(config: Optional[Dict[str, Any]] = None) -> RAGPipeline:
    """
    Create and return a RAG Pipeline instance

    Args:
        config: Optional configuration

    Returns:
        RAGPipeline instance (not initialized)
    """
    return RAGPipeline(config)
