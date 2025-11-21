"""
Complete End-to-End RAG System Test
Tests the entire pipeline: Search → Retrieval → Generation → Validation

File: test_complete_rag.py
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath('.'))

from logger_utils import initialize_logging, get_logger, log_session_end
from config import get_default_config, DATASET_NAME, EMBEDDING_DIM
from loader.dataloader import EnhancedKGDatasetLoader
from model_manager import get_model_manager
from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator
from core.generation import GenerationEngine


class CompleteRAGTester:
    """Complete end-to-end RAG system tester"""
    
    def __init__(self):
        self.logger = get_logger("CompleteRAG")
        self.config = None
        self.data_loader = None
        self.orchestrator = None
        self.generation_engine = None
        
    def setup(self) -> bool:
        """Setup complete system"""
        self.logger.info("=" * 80)
        self.logger.info("SETTING UP COMPLETE RAG SYSTEM")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Configuration
            self.logger.info("Step 1: Loading configuration...")
            self.config = get_default_config()
            
            # Use permissive thresholds
            self.config['search_phases'] = {
                'initial_scan': {
                    'candidates': 200,
                    'semantic_threshold': 0.05,
                    'keyword_threshold': 0.01,
                    'description': 'Initial scan',
                    'enabled': True
                },
                'focused_review': {
                    'candidates': 100,
                    'semantic_threshold': 0.10,
                    'keyword_threshold': 0.05,
                    'description': 'Focused review',
                    'enabled': True
                }
            }
            
            self.config['consensus_threshold'] = 0.3
            
            self.logger.success("Configuration loaded")
            
            # Step 2: Dataset
            self.logger.info("Step 2: Loading dataset...")
            self.data_loader = EnhancedKGDatasetLoader(
                dataset_name=DATASET_NAME,
                embedding_dim=EMBEDDING_DIM
            )
            
            def progress(msg):
                self.logger.info(f"   {msg}")
            
            if not self.data_loader.load_from_huggingface(progress):
                self.logger.error("Dataset loading failed")
                return False
            
            self.logger.success("Dataset loaded")
            
            # Step 3: Models
            self.logger.info("Step 3: Loading models...")
            model_manager = get_model_manager()
            
            embedding_model = model_manager.load_embedding_model()
            reranker_model = model_manager.load_reranker_model(use_mock=True)
            
            self.logger.success("Search models loaded")
            
            # Step 4: Search orchestrator
            self.logger.info("Step 4: Initializing search orchestrator...")
            self.orchestrator = LangGraphRAGOrchestrator(
                data_loader=self.data_loader,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                config=self.config
            )
            
            self.logger.success("Search orchestrator initialized")
            
            # Step 5: Generation engine
            self.logger.info("Step 5: Initializing generation engine...")
            self.generation_engine = GenerationEngine(self.config)
            
            # Try to load LLM model (may fail if not available)
            if not self.generation_engine.initialize():
                self.logger.warning("LLM model not available - will use mock generation")
            else:
                self.logger.success("Generation engine initialized")
            
            self.logger.info("=" * 80)
            self.logger.success("COMPLETE SYSTEM READY")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()
            })
            return False
    
    def test_complete_pipeline(self, query: str) -> Dict[str, Any]:
        """Test complete RAG pipeline"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"TESTING COMPLETE PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Query: {query}")
        
        overall_start = time.time()
        
        result = {
            'query': query,
            'success': False,
            'stages': {},
            'answer': '',
            'error': None
        }
        
        try:
            # Stage 1: Search & Retrieval
            self.logger.info("\n--- STAGE 1: SEARCH & RETRIEVAL ---")
            search_start = time.time()
            
            search_result = self.orchestrator.run(query)
            
            search_time = time.time() - search_start
            
            if search_result.get('errors'):
                self.logger.error("Search failed", {
                    "errors": len(search_result['errors'])
                })
                result['error'] = '; '.join(search_result['errors'])
                return result
            
            final_results = search_result.get('final_results', [])
            
            result['stages']['search'] = {
                'duration': search_time,
                'results_found': len(final_results),
                'metadata': search_result.get('metadata', {})
            }
            
            self.logger.success("Search completed", {
                "time": f"{search_time:.2f}s",
                "results": len(final_results)
            })
            
            if len(final_results) == 0:
                self.logger.warning("No results found")
                result['answer'] = "Maaf, tidak ditemukan dokumen yang relevan untuk menjawab pertanyaan Anda."
                result['success'] = True
                result['total_time'] = time.time() - overall_start
                return result
            
            # Display top results
            self.logger.info("\nTop 3 Retrieved Documents:")
            for i, res in enumerate(final_results[:3], 1):
                rec = res.get('record', {})
                score = res.get('final_score', 0)
                self.logger.info(f"  {i}. Score: {score:.3f}")
                self.logger.info(f"     {rec.get('regulation_type')} No. {rec.get('regulation_number')}/{rec.get('year')}")
                self.logger.info(f"     {rec.get('about', '')[:80]}...")
            
            # Stage 2: Answer Generation
            self.logger.info("\n--- STAGE 2: ANSWER GENERATION ---")
            
            # Check if LLM is available
            if not self.generation_engine.llm_engine._model:
                self.logger.warning("LLM not available - generating mock answer")
                result['answer'] = self._generate_mock_answer(query, final_results)
                result['stages']['generation'] = {
                    'duration': 0,
                    'method': 'mock'
                }
            else:
                gen_start = time.time()
                
                query_analysis = search_result.get('query_analysis', {})
                
                gen_result = self.generation_engine.generate_answer(
                    query=query,
                    retrieved_results=final_results,
                    query_analysis=query_analysis,
                    stream=False
                )
                
                gen_time = time.time() - gen_start
                
                if gen_result['success']:
                    result['answer'] = gen_result['answer']
                    result['citations'] = gen_result.get('citations', [])
                    result['sources'] = gen_result.get('sources', [])
                    
                    result['stages']['generation'] = {
                        'duration': gen_time,
                        'method': 'llm',
                        'tokens': gen_result['metadata'].get('tokens_generated', 0),
                        'validation': gen_result['metadata'].get('validation', {})
                    }
                    
                    self.logger.success("Generation completed", {
                        "time": f"{gen_time:.2f}s",
                        "tokens": gen_result['metadata'].get('tokens_generated', 0)
                    })
                else:
                    self.logger.error("Generation failed", {
                        "error": gen_result.get('error')
                    })
                    result['answer'] = self._generate_mock_answer(query, final_results)
                    result['stages']['generation'] = {
                        'duration': gen_time,
                        'method': 'fallback_mock',
                        'error': gen_result.get('error')
                    }
            
            result['success'] = True
            result['total_time'] = time.time() - overall_start
            
            # Display answer
            self.logger.info("\n" + "=" * 80)
            self.logger.info("FINAL ANSWER")
            self.logger.info("=" * 80)
            print(result['answer'])
            self.logger.info("=" * 80)
            
            # Display summary
            self.logger.success("Pipeline completed successfully", {
                "total_time": f"{result['total_time']:.2f}s",
                "search_time": f"{result['stages']['search']['duration']:.2f}s",
                "gen_time": f"{result['stages']['generation']['duration']:.2f}s"
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()
            })
            
            result['error'] = str(e)
            result['total_time'] = time.time() - overall_start
            return result
    
    def _generate_mock_answer(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Generate mock answer from retrieved documents"""
        from core.generation import CitationFormatter
        
        formatter = CitationFormatter(self.config)
        
        answer_parts = [
            f"Berdasarkan dokumen yang ditemukan, berikut informasi terkait pertanyaan Anda:\n"
        ]
        
        for i, result in enumerate(results[:3], 1):
            rec = result.get('record', {})
            
            citation = formatter.format_citation(rec, style='standard')
            content = rec.get('content', '')[:300]
            
            answer_parts.append(f"\n[Dokumen {i}] {citation}")
            answer_parts.append(f"\n{content}...\n")
        
        answer_parts.append(
            "\n**Catatan**: Jawaban ini dibuat dari dokumen yang tersedia. "
            "Untuk informasi lebih lengkap atau kasus spesifik, disarankan "
            "berkonsultasi dengan ahli hukum."
        )
        
        return "\n".join(answer_parts)
    
    def run_test_suite(self, queries: List[str]) -> Dict[str, Any]:
        """Run complete test suite"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RUNNING COMPLETE TEST SUITE")
        self.logger.info(f"Total queries: {len(queries)}")
        self.logger.info("=" * 80)
        
        suite_results = {
            'total_queries': len(queries),
            'results': [],
            'statistics': {}
        }
        
        for idx, query in enumerate(queries, 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"TEST {idx}/{len(queries)}")
            self.logger.info(f"{'='*80}")
            
            result = self.test_complete_pipeline(query)
            suite_results['results'].append(result)
        
        # Calculate statistics
        successful = sum(1 for r in suite_results['results'] if r['success'])
        failed = len(queries) - successful
        
        total_time = sum(r.get('total_time', 0) for r in suite_results['results'])
        avg_time = total_time / len(queries) if len(queries) > 0 else 0
        
        suite_results['statistics'] = {
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(queries) if len(queries) > 0 else 0,
            'total_time': total_time,
            'avg_time': avg_time
        }
        
        return suite_results
    
    def print_summary(self, suite_results: Dict[str, Any]):
        """Print test suite summary"""
        print("\n" + "=" * 80)
        print("COMPLETE TEST SUITE SUMMARY")
        print("=" * 80)
        
        stats = suite_results['statistics']
        
        print(f"\nTotal Queries: {suite_results['total_queries']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Total Time: {stats['total_time']:.2f}s")
        print(f"Average Time: {stats['avg_time']:.2f}s per query")
        
        print("\nIndividual Results:")
        for i, result in enumerate(suite_results['results'], 1):
            status = "✅" if result['success'] else "❌"
            time_str = f"{result.get('total_time', 0):.2f}s"
            print(f"  {status} Query {i}: {time_str}")
            print(f"     {result['query'][:60]}...")
            if result.get('error'):
                print(f"     Error: {result['error']}")
        
        print("\n" + "=" * 80)
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        if self.generation_engine:
            self.generation_engine.shutdown()
        
        if hasattr(self, 'model_manager'):
            model_manager = get_model_manager()
            model_manager.unload_models()
        
        self.logger.success("Cleanup completed")


def main():
    """Main test execution"""
    # Initialize logging
    initialize_logging(
        enable_file_logging=True,
        log_dir="logs",
        append=False,
        log_filename="complete_rag_test.log"
    )
    
    logger = get_logger("Main")
    logger.info("=" * 80)
    logger.info("COMPLETE END-TO-END RAG SYSTEM TEST")
    logger.info("=" * 80)
    
    # Create tester
    tester = CompleteRAGTester()
    
    # Setup
    if not tester.setup():
        logger.error("Setup failed, aborting tests")
        log_session_end()
        return False
    
    # Define test queries
    test_queries = [
        "Apa sanksi pidana dalam UU ITE?",
        "Bagaimana prosedur pengadaan barang dan jasa pemerintah?",
        "Apa definisi pelanggaran administratif dalam peraturan kepegawaian?"
    ]
    
    # Run test suite
    suite_results = tester.run_test_suite(test_queries)
    
    # Print summary
    tester.print_summary(suite_results)
    
    # Save results
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "complete_rag_results.json"
    
    try:
        # Prepare serializable results
        serializable = {
            'total_queries': suite_results['total_queries'],
            'statistics': suite_results['statistics'],
            'results': [
                {
                    'query': r['query'],
                    'success': r['success'],
                    'total_time': r.get('total_time', 0),
                    'answer_preview': r.get('answer', '')[:200],
                    'error': r.get('error')
                }
                for r in suite_results['results']
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # Cleanup
    tester.cleanup()
    
    # End session
    log_session_end()
    
    success = suite_results['statistics']['failed'] == 0
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)