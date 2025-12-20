"""
LangGraph Orchestrator for RAG System - FIXED VERSION
Proper state handling and error management
"""

import time
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from utils.logger_utils import get_logger

from .query_detection import QueryDetector
from .hybrid_search import HybridSearchEngine
from .stages_research import StagesResearchEngine
from .consensus import ConsensusBuilder
from .reranking import RerankerEngine
from core.knowledge_graph.kg_core import KnowledgeGraphCore


class RAGState(TypedDict, total=False):
    """State for RAG workflow - allows partial updates"""
    # Input
    query: str
    conversation_history: Optional[List[Dict[str, str]]]
    top_k: Optional[int]
    
    # Query Analysis
    query_analysis: Optional[Dict[str, Any]]
    enhanced_query: Optional[str]
    
    # Research
    research_data: Optional[Dict[str, Any]]
    research_summary: Optional[Dict[str, Any]]
    
    # Consensus
    consensus_data: Optional[Dict[str, Any]]
    
    # Reranking
    rerank_data: Optional[Dict[str, Any]]
    final_results: Optional[List[Dict[str, Any]]]
    
    # Metadata
    metadata: Optional[Dict[str, Any]]
    errors: Optional[List[str]]


class LangGraphRAGOrchestrator:
    """
    Orchestrates RAG workflow using LangGraph state machine
    """
    
    def __init__(
        self,
        data_loader,
        embedding_model,
        reranker_model,
        config: Dict[str, Any]
    ):
        self.logger = get_logger("RAGOrchestrator")
        self.config = config  # Store config for access in nodes

        # Initialize components
        self.query_detector = QueryDetector()
        self.hybrid_search = HybridSearchEngine(
            data_loader=data_loader,
            embedding_model=embedding_model,
            reranker_model=reranker_model
        )
        self.stages_research = StagesResearchEngine(
            hybrid_search=self.hybrid_search,
            config=config
        )
        self.consensus_builder = ConsensusBuilder(config=config)
        self.reranker = RerankerEngine(
            reranker_model=reranker_model,
            config=config
        )

        # Initialize KG core for regulation reference extraction
        self.kg_core = KnowledgeGraphCore(config)

        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

        self.logger.info("LangGraph RAG Orchestrator initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("query_detection", self._query_detection_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("consensus", self._consensus_node)
        workflow.add_node("reranking", self._reranking_node)
        
        # Define edges
        workflow.set_entry_point("query_detection")
        workflow.add_edge("query_detection", "research")
        workflow.add_edge("research", "consensus")
        workflow.add_edge("consensus", "reranking")
        workflow.add_edge("reranking", END)
        
        self.logger.info("Workflow graph built", {
            "nodes": 4,
            "entry": "query_detection"
        })
        
        return workflow
    
    def _safe_get(self, state: RAGState, key: str, default: Any = None) -> Any:
        """Safely get value from state"""
        return state.get(key, default)
    
    def _merge_dict(self, existing: Optional[Dict], new: Dict) -> Dict:
        """Safely merge dictionaries"""
        result = existing.copy() if existing else {}
        result.update(new)
        return result
    
    def _merge_list(self, existing: Optional[List], new: List) -> List:
        """Safely merge lists"""
        result = existing.copy() if existing else []
        result.extend(new)
        return result
    
    def _query_detection_node(self, state: RAGState) -> Dict[str, Any]:
        """Query Detection Node"""
        self.logger.info("Executing query detection node")
        
        try:
            query = state['query']
            conversation_history = self._safe_get(state, 'conversation_history', [])
            
            # Analyze query
            query_analysis = self.query_detector.analyze_query(
                query=query,
                conversation_history=conversation_history
            )
            
            # Enhance query
            enhanced_query = self.query_detector.enhance_query(
                original_query=query,
                analysis=query_analysis
            )
            
            self.logger.success("Query detection completed", {
                "type": query_analysis['query_type'],
                "complexity": f"{query_analysis['complexity_score']:.2f}"
            })
            
            new_metadata = {
                "query_detection": {
                    "type": query_analysis['query_type'],
                    "complexity": query_analysis['complexity_score'],
                    "team_size": len(query_analysis['team_composition'])
                }
            }
            
            return {
                "query_analysis": query_analysis,
                "enhanced_query": enhanced_query,
                "metadata": self._merge_dict(self._safe_get(state, 'metadata'), new_metadata)
            }
            
        except Exception as e:
            self.logger.error("Query detection failed", {"error": str(e)})
            return {
                "errors": self._merge_list(self._safe_get(state, 'errors'), [f"Query detection error: {str(e)}"])
            }
    
    def _research_node(self, state: RAGState) -> Dict[str, Any]:
        """Multi-Stage Research Node with Metadata-First Search"""
        self.logger.info("Executing research node")

        try:
            query = self._safe_get(state, 'enhanced_query') or state['query']
            query_analysis = state['query_analysis']

            # Step 1: Extract regulation references and do metadata-first search
            metadata_results = []
            reg_refs = self.kg_core.extract_regulation_references_with_confidence(query)

            if reg_refs:
                self.logger.info("Regulation references found", {
                    "count": len(reg_refs),
                    "top_confidence": reg_refs[0]['confidence'] if reg_refs else 0
                })

                # Perform metadata-first search for exact matches
                metadata_results = self.hybrid_search.metadata_first_search(
                    regulation_references=reg_refs,
                    top_k=20,
                    query=query
                )

                self.logger.info("Metadata-first search completed", {
                    "results": len(metadata_results),
                    "perfect_matches": sum(1 for r in metadata_results if r['scores']['final'] == 1.0)
                })

            # Step 2: Conduct regular multi-stage research
            research_data = self.stages_research.conduct_research(
                query=query,
                query_analysis=query_analysis,
                team_composition=query_analysis['team_composition']
            )

            # Step 3: Merge metadata-first results (they get priority)
            if metadata_results:
                # Get IDs already in research results
                research_ids = {r.get('metadata', {}).get('global_id', i)
                               for i, r in enumerate(research_data['all_results'])}

                # Add metadata results that aren't duplicates
                for meta_result in metadata_results:
                    meta_id = meta_result['metadata'].get('global_id')
                    if meta_id not in research_ids:
                        research_data['all_results'].insert(0, meta_result)

                # Re-sort by score (perfect matches first)
                research_data['all_results'].sort(
                    key=lambda x: x.get('scores', {}).get('final', 0),
                    reverse=True
                )

            # Get summary
            research_summary = self.stages_research.get_research_summary(research_data)

            self.logger.success("Research completed", {
                "rounds": research_data['rounds_executed'],
                "results": len(research_data['all_results']),
                "metadata_first_hits": len(metadata_results)
            })

            new_metadata = {
                "research": {
                    "rounds": research_data['rounds_executed'],
                    "total_results": len(research_data['all_results']),
                    "candidates_evaluated": research_data['total_candidates_evaluated'],
                    "metadata_first_results": len(metadata_results),
                    "regulation_references": len(reg_refs)
                }
            }

            return {
                "research_data": research_data,
                "research_summary": research_summary,
                "metadata": self._merge_dict(self._safe_get(state, 'metadata'), new_metadata)
            }
            
        except Exception as e:
            self.logger.error("Research failed", {"error": str(e)})
            import traceback
            self.logger.debug("Traceback", {"trace": traceback.format_exc()[:500]})
            return {
                "errors": self._merge_list(self._safe_get(state, 'errors'), [f"Research error: {str(e)}"])
            }
    
    def _consensus_node(self, state: RAGState) -> Dict[str, Any]:
        """Consensus Building Node"""
        self.logger.info("Executing consensus node")
        
        try:
            research_data = state['research_data']
            query_analysis = state['query_analysis']
            
            # Build consensus
            consensus_data = self.consensus_builder.build_consensus(
                research_data=research_data,
                team_composition=query_analysis['team_composition']
            )
            
            self.logger.success("Consensus built", {
                "validated": len(consensus_data['validated_results']),
                "agreement": f"{consensus_data['agreement_level']:.2%}"
            })
            
            new_metadata = {
                "consensus": {
                    "validated": len(consensus_data['validated_results']),
                    "agreement_level": consensus_data['agreement_level'],
                    "cross_validated": len(consensus_data['cross_validation_passed']),
                    "flags": len(consensus_data['devil_advocate_flags'])
                }
            }
            
            return {
                "consensus_data": consensus_data,
                "metadata": self._merge_dict(self._safe_get(state, 'metadata'), new_metadata)
            }
            
        except Exception as e:
            self.logger.error("Consensus building failed", {"error": str(e)})
            return {
                "errors": self._merge_list(self._safe_get(state, 'errors'), [f"Consensus error: {str(e)}"])
            }
    
    def _reranking_node(self, state: RAGState) -> Dict[str, Any]:
        """Reranking Node"""
        self.logger.info("Executing reranking node")
        
        try:
            query = state['query']
            consensus_data = state['consensus_data']
            
            # Rerank results
            expected_top_k = state.get('top_k') or self.config.get('final_top_k', 3)
            rerank_data = self.reranker.rerank(
                query=query,
                consensus_results=consensus_data['validated_results'],
                top_k=expected_top_k
            )

            final_results = rerank_data['reranked_results']

            # SAFETY: Ensure exactly expected_top_k documents (defense in depth)
            if len(final_results) > expected_top_k:
                self.logger.warning(f"Reranker returned {len(final_results)} docs, trimming to {expected_top_k}")
                final_results = final_results[:expected_top_k]
                # Update rerank_data to reflect trimmed results
                rerank_data['reranked_results'] = final_results
                rerank_data['metadata']['reranked_count'] = len(final_results)
            # FIXED: Validate and warn if we got fewer than expected
            elif len(final_results) < expected_top_k:
                self.logger.error(
                    f"Pipeline returned {len(final_results)}/{expected_top_k} documents - "
                    f"consensus may be too strict"
                )
                self.logger.info(
                    "RECOMMENDATION: Lower consensus_threshold in config "
                    "(current: {:.2f}) or check if search is finding relevant results".format(
                        self.config.get('consensus_threshold', 0.6)
                    )
                )

            self.logger.success("Reranking completed", {
                "final_count": len(final_results),
                "expected_top_k": expected_top_k
            })

            new_metadata = {
                "reranking": {
                    "final_count": len(final_results),
                    "avg_score": sum(r['final_score'] for r in final_results) / len(final_results) if final_results else 0
                }
            }

            return {
                "rerank_data": rerank_data,
                "final_results": final_results,
                "metadata": self._merge_dict(self._safe_get(state, 'metadata'), new_metadata)
            }
            
        except Exception as e:
            self.logger.error("Reranking failed", {"error": str(e)})
            return {
                "errors": self._merge_list(self._safe_get(state, 'errors'), [f"Reranking error: {str(e)}"])
            }
    
    def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete RAG workflow
        
        Args:
            query: User query
            conversation_history: Optional conversation context
            
        Returns:
            Complete workflow results
        """
        self.logger.info("Starting RAG workflow", {
            "query": query[:50] + "..." if len(query) > 50 else query
        })
        
        # Initialize state
        initial_state: RAGState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "top_k": top_k,
            "metadata": {
                "query": query,
                "start_time": time.time()
            },
            "errors": []
        }
        
        try:
            # Execute workflow
            final_state = self.app.invoke(initial_state)
            
            # Check for errors
            if final_state.get('errors'):
                self.logger.warning("Workflow completed with errors", {
                    "error_count": len(final_state['errors'])
                })
            else:
                self.logger.success("Workflow completed successfully")
            
            return final_state
            
        except Exception as e:
            self.logger.error("Workflow execution failed", {
                "error": str(e)
            })
            
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()[:1000]
            })
            
            return {
                "query": query,
                "errors": [f"Workflow error: {str(e)}"],
                "final_results": [],
                "metadata": {}
            }
    
    def stream_run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ):
        """
        Stream workflow execution (yields intermediate states)
        """
        self.logger.info("Starting streaming RAG workflow")
        
        initial_state: RAGState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "top_k": top_k,
            "metadata": {
                "query": query,
                "start_time": time.time()
            },
            "errors": []
        }
        
        try:
            for state in self.app.stream(initial_state):
                self.logger.debug("Workflow state update", {
                    "keys": list(state.keys())
                })
                yield state
        except Exception as e:
            self.logger.error("Streaming workflow failed", {
                "error": str(e)
            })
            yield {
                "errors": [f"Streaming error: {str(e)}"]
            }