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
from utils.logger_utils import get_logger
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
            from core.model_manager import load_models
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

    def initialize_retrieval_only(self, progress_callback: Optional[callable] = None) -> bool:
        """
        Initialize ONLY retrieval components (no LLM loading)
        Perfect for search UI that doesn't need answer generation
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            self.logger.warning("Pipeline already initialized")
            return True
            
        self.logger.info("Initializing RAG Pipeline (RETRIEVAL ONLY - NO LLM)...")
        start_time = time.time()
        
        total_steps = 3  # Only 3 steps instead of 5
        
        try:
            # Step 1: Load embedding and reranker models
            if progress_callback:
                progress_callback("Loading models", 1, total_steps)
                
            self.logger.info("Step 1/3: Loading embedding and reranker models...")
            from core.model_manager import load_models
            self.embedding_model, self.reranker_model = load_models()
            
            if self.embedding_model is None or self.reranker_model is None:
                self.logger.error("Failed to load models")
                return False
                
            # Step 2: Load dataset
            if progress_callback:
                progress_callback("Loading dataset", 2, total_steps)
                
            self.logger.info("Step 2/3: Loading dataset...")
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
            
            # Step 3: Create RAG orchestrator (NO LLM!)
            if progress_callback:
                progress_callback("Creating orchestrator", 3, total_steps)
                
            self.logger.info("Step 3/3: Creating RAG orchestrator (retrieval only)...")
            from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator
            
            self.orchestrator = LangGraphRAGOrchestrator(
                data_loader=self.data_loader,
                embedding_model=self.embedding_model,
                reranker_model=self.reranker_model,
                config=self.config
            )
            
            # SKIP Step 4 (GenerationEngine) - NO LLM!
            self.generation_engine = None
            self.logger.info("SKIPPED: LLM initialization (retrieval-only mode)")
            
            self._initialized = True
            self._initialization_time = time.time() - start_time
            
            self.logger.success("RAG Pipeline initialized (RETRIEVAL ONLY)", {
                "initialization_time": f"{self._initialization_time:.2f}s",
                "llm_loaded": False
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
        stream: bool = False,
        thinking_mode: str = 'low'
    ) -> Dict[str, Any]:
        """
        Execute complete RAG query

        Args:
            question: User question
            conversation_history: Optional conversation context
            stream: Whether to stream the response
            thinking_mode: Thinking mode ('low', 'medium', 'high')

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
                no_results_response = {
                    'success': True,
                    'answer': 'Maaf, tidak ditemukan dokumen yang relevan untuk pertanyaan Anda.',
                    'sources': [],
                    'metadata': {
                        'retrieval_time': retrieval_time,
                        'total_time': time.time() - start_time,
                        'results_count': 0
                    }
                }

                # Handle streaming case
                if stream:
                    def no_results_generator():
                        # Yield answer as tokens
                        answer = no_results_response['answer']
                        for char in answer:
                            yield {
                                'type': 'token',
                                'token': char,
                                'done': False
                            }
                        # Yield complete
                        yield {
                            'type': 'complete',
                            'done': True,
                            **no_results_response
                        }
                    return no_results_generator()
                else:
                    return no_results_response


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
                    start_time=start_time,
                    rag_result=rag_result,  # Pass full rag_result for metadata
                    thinking_mode=thinking_mode
                )
            else:
                generation_result = self.generation_engine.generate_answer(
                    query=question,
                    retrieved_results=final_results,
                    query_analysis=query_analysis,
                    conversation_history=conversation_history,
                    stream=False,
                    thinking_mode=thinking_mode
                )

                total_time = time.time() - start_time

                if generation_result['success']:
                    self.logger.success("Query completed successfully", {
                        "total_time": f"{total_time:.2f}s",
                        "answer_length": len(generation_result['answer'])
                    })

                    # Extract phase metadata from orchestrator result
                    phase_metadata = {}
                    consensus_data = rag_result.get('consensus_data', {})
                    research_data = rag_result.get('research_data', {})

                    # Transform research_data into format expected by UI
                    # The UI expects: {key: {phase, researcher, researcher_name, candidates, confidence}}
                    if research_data:
                        # Get phase_results and persona_results
                        phase_results = research_data.get('phase_results', {})
                        persona_results = research_data.get('persona_results', {})
                        rounds = research_data.get('rounds', [])

                        # Build phase metadata by combining phase and persona info
                        from config import RESEARCH_TEAM_PERSONAS

                        entry_idx = 0
                        for phase_name, results in phase_results.items():
                            # Group results by persona within each phase
                            persona_groups = {}
                            for result in results:
                                persona = result.get('metadata', {}).get('persona', 'unknown')
                                if persona not in persona_groups:
                                    persona_groups[persona] = []
                                persona_groups[persona].append(result)

                            # Create entry for each persona in this phase
                            for persona_name, persona_results_list in persona_groups.items():
                                key = f"{entry_idx}_{phase_name}_{persona_name}"
                                researcher_info = RESEARCH_TEAM_PERSONAS.get(persona_name, {})

                                # Transform results to candidates format
                                candidates = []
                                for r in persona_results_list:
                                    candidates.append({
                                        'record': r.get('record', {}),
                                        'composite_score': r.get('scores', {}).get('final', 0),
                                        'semantic_score': r.get('scores', {}).get('semantic', 0),
                                        'keyword_score': r.get('scores', {}).get('keyword', 0),
                                        'kg_score': r.get('scores', {}).get('kg', 0),
                                        'team_consensus': r.get('team_consensus', False),
                                        'researcher_agreement': r.get('researcher_agreement', 0)
                                    })

                                phase_metadata[key] = {
                                    'phase': phase_name,
                                    'researcher': persona_name,
                                    'researcher_name': researcher_info.get('name', persona_name),
                                    'candidates': candidates,
                                    'confidence': 1.0,  # Default confidence
                                    'results': candidates  # Alias
                                }
                                entry_idx += 1

                    # Also include from metadata if available
                    if rag_result.get('metadata', {}).get('research_phases'):
                        phase_metadata.update(rag_result['metadata']['research_phases'])

                    result = {
                        'success': True,
                        'answer': generation_result['answer'],
                        'thinking': generation_result.get('thinking', ''),  # Pass through thinking
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
                        },
                        # Include research process metadata for detailed display
                        'phase_metadata': phase_metadata,
                        'all_retrieved_metadata': phase_metadata,  # Alias for compatibility
                        'consensus_data': consensus_data,
                        'research_data': research_data,
                        'communities': rag_result.get('communities', []),
                        'clusters': rag_result.get('clusters', [])
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

    def retrieve_documents(
        self,
        question: str,
        top_k: int = 10,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents WITHOUT LLM generation (for search UI)
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            conversation_history: Optional conversation context
            
        Returns:
            Dictionary with sources, metadata, and phase_metadata
        """
        if not self._initialized:
            return {
                'success': False,
                'error': 'Pipeline not initialized',
                'sources': [],
                'metadata': {}
            }

        self.logger.info("Retrieving documents (no LLM)", {"top_k": top_k})
        start_time = time.time()

        try:
            # Run retrieval
            rag_result = self.orchestrator.run(
                query=question,
                conversation_history=conversation_history or [],
                top_k=top_k
            )

            retrieval_time = time.time() - start_time
            final_results = rag_result.get('final_results', [])[:top_k]

            # Build phase metadata
            phase_metadata = {}
            research_data = rag_result.get('research_data', {})

            if research_data:
                phase_results = research_data.get('phase_results', {})
                from config import RESEARCH_TEAM_PERSONAS

                entry_idx = 0
                for phase_name, results in phase_results.items():
                    persona_groups = {}
                    for result in results:
                        persona = result.get('metadata', {}).get('persona', 'unknown')
                        if persona not in persona_groups:
                            persona_groups[persona] = []
                        persona_groups[persona].append(result)

                    for persona_name, persona_results_list in persona_groups.items():
                        key = f"{entry_idx}_{phase_name}_{persona_name}"
                        researcher_info = RESEARCH_TEAM_PERSONAS.get(persona_name, {})

                        candidates = []
                        for r in persona_results_list:
                            candidates.append({
                                'record': r.get('record', {}),
                                'scores': r.get('scores', {}),
                                'final_score': r.get('scores', {}).get('final', 0),
                                '_phase': phase_name,
                                '_researcher': persona_name
                            })

                        phase_metadata[key] = {
                            'phase': phase_name,
                            'researcher': persona_name,
                            'researcher_name': researcher_info.get('name', persona_name),
                            'candidates': candidates,
                            'confidence': 1.0,
                            'results': candidates
                        }
                        entry_idx += 1

            # Format sources
            sources = []
            for r in final_results:
                record = r.get('record', r)
                scores = r.get('scores', {})
                sources.append({
                    'regulation_type': record.get('regulation_type', ''),
                    'regulation_number': record.get('regulation_number', ''),
                    'year': record.get('year', ''),
                    'about': record.get('about', ''),
                    'content': record.get('content', ''),
                    'enacting_body': record.get('enacting_body', ''),
                    'chapter': record.get('chapter', ''),
                    'article': record.get('article', ''),
                    'score': scores.get('final', 0),
                    'scores': scores,
                    'record': record,
                    '_phase': r.get('_phase', ''),
                    '_researcher': r.get('_researcher', '')
                })

            return {
                'success': True,
                'sources': sources,
                'metadata': {
                    'query': question,
                    'retrieval_time': retrieval_time,
                    'total_time': retrieval_time,
                    'results_count': len(sources)
                },
                'phase_metadata': phase_metadata,
                'consensus_data': rag_result.get('consensus_data', {})
            }

        except Exception as e:
            self.logger.error("Retrieval failed", {"error": str(e)})
            return {
                'success': False,
                'error': str(e),
                'sources': [],
                'metadata': {}
            }

    def _generate_streaming(
        self,
        question: str,
        retrieved_results: List[Dict],
        query_analysis: Dict,
        conversation_history: Optional[List],
        retrieval_time: float,
        start_time: float,
        rag_result: Optional[Dict] = None,
        thinking_mode: str = 'low'
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate streaming response with full metadata"""

        self.logger.info("Starting streaming generation", {
            "thinking_mode": thinking_mode
        })

        # Pre-build phase_metadata from rag_result for inclusion in complete chunk
        phase_metadata = {}
        consensus_data = {}
        research_data = {}

        if rag_result:
            consensus_data = rag_result.get('consensus_data', {})
            research_data = rag_result.get('research_data', {})

            # Transform research_data into phase_metadata format (same as non-streaming)
            if research_data:
                phase_results = research_data.get('phase_results', {})
                from config import RESEARCH_TEAM_PERSONAS

                entry_idx = 0
                for phase_name, results in phase_results.items():
                    # Group results by persona within each phase
                    persona_groups = {}
                    for result in results:
                        persona = result.get('metadata', {}).get('persona', 'unknown')
                        if persona not in persona_groups:
                            persona_groups[persona] = []
                        persona_groups[persona].append(result)

                    # Create entry for each persona in this phase
                    for persona_name, persona_results_list in persona_groups.items():
                        key = f"{entry_idx}_{phase_name}_{persona_name}"
                        researcher_info = RESEARCH_TEAM_PERSONAS.get(persona_name, {})

                        # Transform results to candidates format with ALL scores
                        candidates = []
                        for r in persona_results_list:
                            candidates.append({
                                'record': r.get('record', {}),
                                'scores': r.get('scores', {}),
                                'composite_score': r.get('scores', {}).get('final', 0),
                                'semantic_score': r.get('scores', {}).get('semantic', 0),
                                'keyword_score': r.get('scores', {}).get('keyword', 0),
                                'kg_score': r.get('scores', {}).get('kg', 0),
                                'authority_score': r.get('scores', {}).get('authority', 0),
                                'temporal_score': r.get('scores', {}).get('temporal', 0),
                                'completeness_score': r.get('scores', {}).get('completeness', 0),
                                'team_consensus': r.get('team_consensus', False),
                                'researcher_agreement': r.get('researcher_agreement', 0)
                            })

                        phase_metadata[key] = {
                            'phase': phase_name,
                            'researcher': persona_name,
                            'researcher_name': researcher_info.get('name', persona_name),
                            'candidates': candidates,
                            'confidence': 1.0,
                            'results': candidates
                        }
                        entry_idx += 1

        try:
            for chunk in self.generation_engine.generate_answer(
                query=question,
                retrieved_results=retrieved_results,
                query_analysis=query_analysis,
                conversation_history=conversation_history,
                stream=True,
                thinking_mode=thinking_mode
            ):
                if chunk.get('type') == 'token':
                    yield {
                        'type': 'token',
                        'token': chunk['token'],
                        'done': False
                    }
                elif chunk.get('type') == 'thinking':
                    yield {
                        'type': 'thinking',
                        'token': chunk['token'],
                        'done': False
                    }
                elif chunk.get('type') == 'complete':
                    total_time = time.time() - start_time

                    # Format sources from retrieved results with COMPLETE metadata
                    sources = []
                    citations = []
                    for r in retrieved_results:
                        record = r.get('record', r)
                        scores = r.get('scores', {})

                        # Build complete citation with all required fields for formatting
                        citation = {
                            # Basic regulation info
                            'regulation_type': record.get('regulation_type', ''),
                            'regulation_number': record.get('regulation_number', ''),
                            'year': record.get('year', ''),
                            'about': record.get('about', ''),
                            'content': record.get('content', ''),
                            'enacting_body': record.get('enacting_body', ''),
                            'global_id': record.get('global_id', ''),

                            # Article/Chapter location
                            'chapter': record.get('chapter', record.get('bab', '')),
                            'article': record.get('article', record.get('pasal', '')),
                            'article_number': record.get('article_number', ''),
                            'section': record.get('section', record.get('bagian', '')),
                            'paragraph': record.get('paragraph', record.get('ayat', '')),

                            # Effective date
                            'effective_date': record.get('effective_date', record.get('tanggal_penetapan', '')),
                            'tanggal_penetapan': record.get('tanggal_penetapan', record.get('effective_date', '')),

                            # All scores
                            'final_score': scores.get('final', r.get('final_score', 0)),
                            'score': scores.get('final', r.get('final_score', 0)),
                            'semantic_score': scores.get('semantic', r.get('semantic_score', 0)),
                            'keyword_score': scores.get('keyword', r.get('keyword_score', 0)),
                            'kg_score': scores.get('kg', r.get('kg_score', 0)),
                            'authority_score': scores.get('authority', r.get('authority_score', 0)),
                            'temporal_score': scores.get('temporal', r.get('temporal_score', 0)),
                            'completeness_score': scores.get('completeness', r.get('completeness_score', 0)),
                            'rerank_score': r.get('rerank_score', 0),
                            'composite_score': r.get('composite_score', 0),

                            # KG metadata
                            'kg_primary_domain': record.get('kg_primary_domain', ''),
                            'kg_hierarchy_level': record.get('kg_hierarchy_level', 0),
                            'kg_cross_ref_count': record.get('kg_cross_ref_count', 0),
                            'kg_pagerank': record.get('kg_pagerank', 0),

                            # Team consensus info (if available)
                            'team_consensus': r.get('team_consensus', False),
                            'researcher_agreement': r.get('researcher_agreement', 0),
                            'supporting_researchers': r.get('supporting_researchers', [])
                        }

                        sources.append(citation)
                        citations.append(citation)

                    yield {
                        'type': 'complete',
                        'success': True,
                        'answer': chunk['answer'],
                        'thinking': chunk.get('thinking', ''),  # Include thinking process
                        'sources': sources,
                        'citations': citations,
                        'metadata': {
                            'question': question,
                            'retrieval_time': retrieval_time,
                            'generation_time': chunk.get('generation_time', 0),
                            'total_time': total_time,
                            'results_count': len(retrieved_results),
                            'tokens_generated': chunk.get('tokens_generated', 0),
                            'query_type': query_analysis.get('query_type', 'general')
                        },
                        # Include all research metadata for transparency
                        'phase_metadata': phase_metadata,
                        'all_retrieved_metadata': phase_metadata,
                        'consensus_data': consensus_data,
                        'research_data': research_data,
                        'research_log': {
                            'phase_results': phase_metadata,
                            'team_members': list(set(
                                pm.get('researcher', '') for pm in phase_metadata.values()
                            )) if phase_metadata else [],
                            'total_documents_retrieved': sum(
                                len(pm.get('candidates', []))
                                for pm in phase_metadata.values()
                            ) if phase_metadata else 0
                        },
                        'communities': rag_result.get('communities', []) if rag_result else [],
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
