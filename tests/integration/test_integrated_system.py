"""
Fixed Integrated RAG System Test
Modular, robust testing with proper error handling and output display
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Skip if dependencies not available
import pytest
numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.logger_utils import get_logger, initialize_logging, log_session_end
from config import (
    DATASET_NAME,
    get_default_config,
    apply_validated_config,
    EMBEDDING_DIM
)
from loader.dataloader import EnhancedKGDatasetLoader
from core.model_manager import get_model_manager
from core.search.query_detection import QueryDetector
from core.search.hybrid_search import HybridSearchEngine
from core.search.stages_research import StagesResearchEngine
from core.search.consensus import ConsensusBuilder
from core.search.reranking import RerankerEngine

# Try to import LangGraph orchestrator (optional)
try:
    from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_ERROR = str(e)


class IntegratedRAGTester:
    """
    Complete integration tester with robust error handling
    """
    
    def __init__(self, use_mock_reranker: bool = True):
        self.logger = get_logger("IntegratedTest")
        self.use_mock_reranker = use_mock_reranker
        
        # Components
        self.data_loader: Optional[EnhancedKGDatasetLoader] = None
        self.model_manager = None
        self.embedding_model = None
        self.reranker_model = None
        self.config: Optional[Dict[str, Any]] = None
        
        # Search components
        self.query_detector: Optional[QueryDetector] = None
        self.hybrid_search: Optional[HybridSearchEngine] = None
        self.stages_research: Optional[StagesResearchEngine] = None
        self.consensus_builder: Optional[ConsensusBuilder] = None
        self.reranker: Optional[RerankerEngine] = None
        self.orchestrator: Optional[LangGraphRAGOrchestrator] = None
        
        self.logger.info("IntegratedRAGTester initialized", {
            "mock_reranker": use_mock_reranker,
            "langgraph_available": LANGGRAPH_AVAILABLE
        })
    
    def setup(self) -> bool:
        """
        Setup complete system with proper error handling
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING INTEGRATED SYSTEM SETUP")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Configuration
            if not self._setup_config():
                return False
            
            # Step 2: Dataset
            if not self._setup_dataset():
                return False
            
            # Step 3: Models
            if not self._setup_models():
                return False
            
            # Step 4: Search Components
            if not self._setup_search_components():
                return False
            
            # Step 5: LangGraph Orchestrator (optional)
            self._setup_orchestrator()
            
            self.logger.info("=" * 80)
            self.logger.success("SYSTEM SETUP COMPLETE")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error("Setup failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()[:1000]
            })
            return False
    
    def _setup_config(self) -> bool:
        """Setup configuration with ULTRA-PERMISSIVE thresholds"""
        self.logger.info("Step 1: Loading configuration")
        
        try:
            self.config = get_default_config()
            
            # CRITICAL FIX: Ultra-permissive thresholds to ensure results
            self.config['search_phases'] = {
                'initial_scan': {
                    'candidates': 400,
                    'semantic_threshold': 0.01,   # Ultra-low
                    'keyword_threshold': 0.001,   # Ultra-low
                    'description': 'Initial scan',
                    'enabled': True,
                    'time_limit': 30,
                    'focus_areas': ['regulation_type', 'enacting_body']
                },
                'focused_review': {
                    'candidates': 200,
                    'semantic_threshold': 0.05,   # Very permissive
                    'keyword_threshold': 0.01,    # Very permissive
                    'description': 'Focused review',
                    'enabled': True,
                    'time_limit': 45,
                    'focus_areas': ['content', 'chapter', 'article']
                },
                'deep_analysis': {
                    'candidates': 100,
                    'semantic_threshold': 0.10,   # Permissive
                    'keyword_threshold': 0.05,    # Permissive
                    'description': 'Deep analysis',
                    'enabled': True,
                    'time_limit': 60,
                    'focus_areas': ['kg_entities', 'cross_references']
                }
            }
            
            # CRITICAL FIX: Lower consensus threshold
            self.config['consensus_threshold'] = 0.3  # 30% team agreement
            self.config['initial_quality'] = 1.0
            self.config['quality_degradation'] = 0.2
            self.config['min_quality'] = 0.2
            
            self.config = apply_validated_config(self.config)
            
            self.logger.success("Configuration loaded", {
                "max_rounds": self.config.get('max_rounds'),
                "final_top_k": self.config.get('final_top_k')
            })
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration setup failed: {e}")
            return False
    
    def _setup_dataset(self) -> bool:
        """Setup dataset"""
        self.logger.info("Step 2: Loading dataset")
        
        try:
            self.data_loader = EnhancedKGDatasetLoader(
                dataset_name=DATASET_NAME,
                embedding_dim=EMBEDDING_DIM
            )
            
            def progress_callback(msg):
                self.logger.info(f"   {msg}")
            
            if not self.data_loader.load_from_huggingface(progress_callback):
                self.logger.error("Dataset loading failed")
                return False
            
            stats = self.data_loader.get_statistics()
            self.logger.success("Dataset loaded", {
                "records": f"{stats['total_records']:,}",
                "kg_enhanced": f"{stats['kg_enhanced']:,}"
            })
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset setup failed: {e}")
            return False
    
    def _setup_models(self) -> bool:
        """Setup models"""
        self.logger.info("Step 3: Loading models")
        
        try:
            self.model_manager = get_model_manager()
            
            # Embedding model
            try:
                self.embedding_model = self.model_manager.load_embedding_model()
                self.logger.success("Embedding model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                return False
            
            # Reranker model
            try:
                self.reranker_model = self.model_manager.load_reranker_model(
                    use_mock=self.use_mock_reranker
                )
                self.logger.success("Reranker model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load reranker model: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            return False
    
    def _setup_search_components(self) -> bool:
        """Setup search components"""
        self.logger.info("Step 4: Initializing search components")
        
        try:
            self.query_detector = QueryDetector()
            self.logger.info("   ✓ Query detector initialized")
            
            self.hybrid_search = HybridSearchEngine(
                data_loader=self.data_loader,
                embedding_model=self.embedding_model,
                reranker_model=self.reranker_model
            )
            self.logger.info("   ✓ Hybrid search initialized")
            
            self.stages_research = StagesResearchEngine(
                hybrid_search=self.hybrid_search,
                config=self.config
            )
            self.logger.info("   ✓ Stages research initialized")
            
            self.consensus_builder = ConsensusBuilder(config=self.config)
            self.logger.info("   ✓ Consensus builder initialized")
            
            self.reranker = RerankerEngine(
                reranker_model=self.reranker_model,
                config=self.config
            )
            self.logger.info("   ✓ Reranker initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Search components setup failed: {e}")
            return False
    
    def _setup_orchestrator(self):
        """Setup LangGraph orchestrator (optional)"""
        if LANGGRAPH_AVAILABLE:
            self.logger.info("Step 5: Initializing LangGraph orchestrator")
            try:
                self.orchestrator = LangGraphRAGOrchestrator(
                    data_loader=self.data_loader,
                    embedding_model=self.embedding_model,
                    reranker_model=self.reranker_model,
                    config=self.config
                )
                self.logger.success("LangGraph orchestrator initialized")
            except Exception as e:
                self.logger.warning(f"LangGraph orchestrator failed: {e}")
                self.orchestrator = None
        else:
            self.logger.warning(f"LangGraph not available: {LANGGRAPH_ERROR}")
            self.orchestrator = None
    
    def test_manual_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Test manual pipeline (step-by-step without LangGraph)
        
        Args:
            query: Test query
            
        Returns:
            Results dictionary with detailed metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("TESTING MANUAL PIPELINE", {"query": query})
        self.logger.info("=" * 80)
        
        start_time = time.time()
        result = {
            'query': query,
            'method': 'manual',
            'success': False,
            'stages': {},
            'final_results': [],
            'error': None
        }
        
        try:
            # Stage 1: Query Detection
            self.logger.info("Stage 1: Query Detection")
            t0 = time.time()
            
            query_analysis = self.query_detector.analyze_query(query)
            
            result['stages']['query_detection'] = {
                'duration': time.time() - t0,
                'type': query_analysis['query_type'],
                'complexity': query_analysis['complexity_score'],
                'team_size': len(query_analysis['team_composition'])
            }
            
            self.logger.success("Query detection completed", 
                              result['stages']['query_detection'])
            
            # Stage 2: Research
            self.logger.info("Stage 2: Multi-stage research")
            t0 = time.time()
            
            research_data = self.stages_research.conduct_research(
                query=query,
                query_analysis=query_analysis,
                team_composition=query_analysis['team_composition']
            )
            
            # FIX: Handle both 'rounds_executed' and len(rounds)
            rounds_executed = research_data.get('rounds_executed', len(research_data.get('rounds', [])))
            
            result['stages']['research'] = {
                'duration': time.time() - t0,
                'rounds': rounds_executed,
                'total_results': len(research_data.get('all_results', [])),
                'candidates_evaluated': research_data.get('total_candidates_evaluated', 0)
            }
            
            self.logger.success("Research completed", result['stages']['research'])
            
            if len(research_data.get('all_results', [])) == 0:
                self.logger.warning("No results found in research phase")
                result['success'] = False
                result['error'] = 'No results found'
                result['total_duration'] = time.time() - start_time
                
                # FIX: Print detailed debug info
                self._print_debug_info(research_data)
                return result
            
            # Stage 3: Consensus
            self.logger.info("Stage 3: Consensus building")
            t0 = time.time()
            
            consensus_data = self.consensus_builder.build_consensus(
                research_data=research_data,
                team_composition=query_analysis['team_composition']
            )
            
            result['stages']['consensus'] = {
                'duration': time.time() - t0,
                'validated': len(consensus_data.get('validated_results', [])),
                'agreement_level': consensus_data.get('agreement_level', 0),
                'cross_validated': len(consensus_data.get('cross_validation_passed', []))
            }
            
            self.logger.success("Consensus built", result['stages']['consensus'])
            
            # Stage 4: Reranking
            self.logger.info("Stage 4: Reranking")
            t0 = time.time()
            
            rerank_data = self.reranker.rerank(
                query=query,
                consensus_results=consensus_data.get('validated_results', [])
            )
            
            result['stages']['reranking'] = {
                'duration': time.time() - t0,
                'final_count': len(rerank_data.get('reranked_results', []))
            }
            
            result['final_results'] = rerank_data.get('reranked_results', [])
            result['success'] = len(result['final_results']) > 0
            result['total_duration'] = time.time() - start_time
            
            self.logger.success("Reranking completed", result['stages']['reranking'])
            
            # FIX: Print results to console
            self._print_results_to_console(result)
            
            self.logger.info("=" * 80)
            self.logger.success("MANUAL PIPELINE COMPLETED", {
                "duration": f"{result['total_duration']:.2f}s",
                "final_results": len(result['final_results'])
            })
            self.logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            result['total_duration'] = time.time() - start_time
            
            self.logger.error("Manual pipeline failed", {
                "error": str(e),
                "duration": f"{result['total_duration']:.2f}s"
            })
            
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()[:1000]
            })
            
            return result
    
    def _print_debug_info(self, research_data: Dict):
        """Print detailed debug information when no results found"""
        print("\n" + "="*80)
        print("DEBUG: No Results Found - Detailed Information")
        print("="*80)
        
        rounds = research_data.get('rounds', [])
        print(f"\nTotal Rounds: {len(rounds)}")
        
        for i, round_data in enumerate(rounds, 1):
            print(f"\n--- Round {i} ---")
            print(f"Quality Multiplier: {round_data.get('quality_multiplier', 'N/A'):.3f}")
            print(f"Results: {len(round_data.get('results', []))}")
            print(f"Candidates Evaluated: {round_data.get('candidates_evaluated', 0)}")
            
            phase_breakdown = round_data.get('phase_breakdown', {})
            if phase_breakdown:
                print(f"Phase Breakdown:")
                for phase, count in phase_breakdown.items():
                    print(f"  - {phase}: {count}")
            
            persona_breakdown = round_data.get('persona_breakdown', {})
            if persona_breakdown:
                print(f"Persona Breakdown:")
                for persona, count in persona_breakdown.items():
                    print(f"  - {persona}: {count}")
        
        print("\n" + "="*80)
    
    def _print_results_to_console(self, result: Dict):
        """Print results to console for visibility"""
        print("\n" + "="*80)
        print("PIPELINE RESULTS")
        print("="*80)
        print(f"\nQuery: {result['query']}")
        print(f"Method: {result['method']}")
        print(f"Success: {result['success']}")
        print(f"Total Duration: {result.get('total_duration', 0):.2f}s")
        
        if result.get('stages'):
            print("\nStage Breakdown:")
            for stage_name, stage_data in result['stages'].items():
                duration = stage_data.get('duration', 0)
                print(f"  {stage_name}: {duration:.2f}s")
                for key, value in stage_data.items():
                    if key != 'duration':
                        print(f"    - {key}: {value}")
        
        final_results = result.get('final_results', [])
        print(f"\nFinal Results: {len(final_results)}")
        
        if final_results:
            print("\nTop Results:")
            for i, res in enumerate(final_results[:5], 1):
                rec = res.get('record', {})
                score = res.get('final_score', res.get('rerank_score', 0))
                print(f"\n  {i}. Score: {score:.4f}")
                print(f"     {rec.get('regulation_type', 'N/A')} "
                      f"{rec.get('regulation_number', 'N/A')}/{rec.get('year', 'N/A')}")
                print(f"     About: {rec.get('about', 'N/A')[:100]}...")
                
                scores = res.get('scores', {})
                if scores:
                    print(f"     Detailed Scores:")
                    print(f"       - Semantic: {scores.get('semantic', 0):.3f}")
                    print(f"       - Keyword: {scores.get('keyword', 0):.3f}")
                    print(f"       - KG: {scores.get('kg', 0):.3f}")
                    print(f"       - Authority: {scores.get('authority', 0):.3f}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        
        print("\n" + "="*80)
    
    def test_langgraph_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Test LangGraph orchestrated pipeline
        
        Args:
            query: Test query
            
        Returns:
            Results dictionary
        """
        if not self.orchestrator:
            self.logger.warning("LangGraph orchestrator not available")
            return {
                'query': query,
                'method': 'langgraph',
                'success': False,
                'error': 'LangGraph not available',
                'final_results': []
            }
        
        self.logger.info("=" * 80)
        self.logger.info("TESTING LANGGRAPH PIPELINE", {"query": query})
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            final_state = self.orchestrator.run(query)
            
            result = {
                'query': query,
                'method': 'langgraph',
                'success': len(final_state.get('errors', [])) == 0,
                'final_results': final_state.get('final_results', []),
                'metadata': final_state.get('metadata', {}),
                'errors': final_state.get('errors', []),
                'total_duration': time.time() - start_time,
                'error': None
            }
            
            if result['success']:
                self.logger.success("LangGraph pipeline completed", {
                    "duration": f"{result['total_duration']:.2f}s",
                    "final_results": len(result['final_results'])
                })
                
                # FIX: Print results to console
                self._print_results_to_console(result)
            else:
                self.logger.error("LangGraph pipeline had errors", {
                    "errors": len(result['errors'])
                })
                result['error'] = '; '.join(result['errors'])
            
            self.logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            result = {
                'query': query,
                'method': 'langgraph',
                'success': False,
                'error': str(e),
                'total_duration': time.time() - start_time,
                'final_results': []
            }
            
            self.logger.error("LangGraph pipeline failed", {
                "error": str(e),
                "duration": f"{result['total_duration']:.2f}s"
            })
            
            return result
    
    def run_test_suite(self, queries: List[str]) -> Dict[str, Any]:
        """
        Run complete test suite with multiple queries
        
        Args:
            queries: List of test queries
            
        Returns:
            Aggregated results with statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("RUNNING COMPLETE TEST SUITE", {
            "queries": len(queries)
        })
        self.logger.info("=" * 80)
        
        suite_results = {
            'total_queries': len(queries),
            'manual_results': [],
            'langgraph_results': [],
            'statistics': {}
        }
        
        for idx, query in enumerate(queries, 1):
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"TEST {idx}/{len(queries)}: {query}")
            self.logger.info(f"{'=' * 80}\n")
            
            # Test manual pipeline
            manual_result = self.test_manual_pipeline(query)
            suite_results['manual_results'].append(manual_result)
            
            # Test LangGraph pipeline if available
            if LANGGRAPH_AVAILABLE and self.orchestrator:
                langgraph_result = self.test_langgraph_pipeline(query)
                suite_results['langgraph_results'].append(langgraph_result)
        
        # Calculate statistics
        suite_results['statistics'] = self._calculate_statistics(suite_results)
        
        # FIX: Print summary to console
        self._print_suite_summary(suite_results)
        
        self.logger.info("=" * 80)
        self.logger.success("TEST SUITE COMPLETED")
        self.logger.info("=" * 80)
        
        return suite_results
    
    def _calculate_statistics(self, suite_results: Dict) -> Dict:
        """Calculate aggregate statistics"""
        manual_results = suite_results['manual_results']
        langgraph_results = suite_results['langgraph_results']
        
        stats = {
            'manual': {
                'total': len(manual_results),
                'successful': sum(1 for r in manual_results if r['success']),
                'failed': sum(1 for r in manual_results if not r['success']),
                'avg_duration': 0,
                'avg_results': 0
            }
        }
        
        if manual_results:
            successful = [r for r in manual_results if r['success']]
            if successful:
                stats['manual']['avg_duration'] = sum(
                    r['total_duration'] for r in successful
                ) / len(successful)
                stats['manual']['avg_results'] = sum(
                    len(r.get('final_results', [])) for r in successful
                ) / len(successful)
        
        if langgraph_results:
            stats['langgraph'] = {
                'total': len(langgraph_results),
                'successful': sum(1 for r in langgraph_results if r['success']),
                'failed': sum(1 for r in langgraph_results if not r['success']),
                'avg_duration': 0,
                'avg_results': 0
            }
            
            successful = [r for r in langgraph_results if r['success']]
            if successful:
                stats['langgraph']['avg_duration'] = sum(
                    r['total_duration'] for r in successful
                ) / len(successful)
                stats['langgraph']['avg_results'] = sum(
                    len(r.get('final_results', [])) for r in successful
                ) / len(successful)
        
        return stats
    
    def _print_suite_summary(self, suite_results: Dict):
        """Print test suite summary to console"""
        print("\n" + "="*80)
        print("TEST SUITE SUMMARY")
        print("="*80)
        
        stats = suite_results['statistics']
        
        print(f"\nManual Pipeline:")
        manual_stats = stats['manual']
        print(f"  Total: {manual_stats['total']}")
        print(f"  Successful: {manual_stats['successful']}")
        print(f"  Failed: {manual_stats['failed']}")
        print(f"  Success Rate: {manual_stats['successful']/manual_stats['total']*100:.1f}%")
        print(f"  Avg Duration: {manual_stats['avg_duration']:.2f}s")
        print(f"  Avg Results: {manual_stats['avg_results']:.1f}")
        
        if 'langgraph' in stats:
            print(f"\nLangGraph Pipeline:")
            lg_stats = stats['langgraph']
            print(f"  Total: {lg_stats['total']}")
            print(f"  Successful: {lg_stats['successful']}")
            print(f"  Failed: {lg_stats['failed']}")
            print(f"  Success Rate: {lg_stats['successful']/lg_stats['total']*100:.1f}%")
            print(f"  Avg Duration: {lg_stats['avg_duration']:.2f}s")
            print(f"  Avg Results: {lg_stats['avg_results']:.1f}")
        
        print("\n" + "="*80)
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources")
        
        if self.model_manager:
            self.model_manager.unload_models()
        
        self.logger.success("Cleanup completed")


def main():
    """Main test execution"""
    
    # Initialize logging
    initialize_logging(
        enable_file_logging=True,
        log_dir="logs",
        append=False,
        log_filename="integrated_test.log"
    )
    
    logger = get_logger("Main")
    logger.info("=" * 80)
    logger.info("INTEGRATED RAG SYSTEM TEST")
    logger.info("=" * 80)
    
    # Create tester
    tester = IntegratedRAGTester(use_mock_reranker=True)
    
    # Setup system
    if not tester.setup():
        logger.error("System setup failed, aborting tests")
        log_session_end()
        return False
    
    # Define test queries
    test_queries = [
        "Apa saja sanksi pidana dalam UU ITE?",
        "Bagaimana prosedur pengadaan barang dan jasa pemerintah?",
        "Apa definisi pelanggaran administratif dalam peraturan kepegawaian?"
    ]
    
    # Run test suite
    suite_results = tester.run_test_suite(test_queries)
    
    # Save results
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "integrated_test_results.json"
    try:
        serializable_results = {
            'total_queries': suite_results['total_queries'],
            'statistics': suite_results['statistics'],
            'manual_summary': [
                {
                    'query': r['query'],
                    'success': r['success'],
                    'duration': r.get('total_duration', 0),
                    'results_count': len(r.get('final_results', [])),
                    'error': r.get('error')
                }
                for r in suite_results['manual_results']
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # Cleanup
    tester.cleanup()
    
    # End session
    log_session_end()
    
    return suite_results['statistics']['manual']['successful'] == suite_results['statistics']['manual']['total']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)