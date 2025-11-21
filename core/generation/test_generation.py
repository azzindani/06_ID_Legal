"""
Comprehensive Test Suite for Generation Module
Tests all generation components with real data

File: core/generation/test_generation.py
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from logger_utils import get_logger, initialize_logging, log_session_end
from config import get_default_config, DATASET_NAME, EMBEDDING_DIM
from loader.dataloader import EnhancedKGDatasetLoader
from model_manager import get_model_manager
from core.search.query_detection import QueryDetector
from core.search.hybrid_search import HybridSearchEngine
from core.search.stages_research import StagesResearchEngine
from core.search.consensus import ConsensusBuilder
from core.search.reranking import RerankerEngine
from core.generation import (
    GenerationEngine,
    PromptBuilder,
    CitationFormatter,
    ResponseValidator
)


class GenerationTester:
    """Comprehensive tester for generation module"""
    
    def __init__(self):
        self.logger = get_logger("GenerationTester")
        self.config = None
        self.data_loader = None
        self.generation_engine = None
        
        # Mock retrieved results for testing
        self.mock_results = []
        
    def setup(self) -> bool:
        """Setup test environment"""
        self.logger.info("=" * 80)
        self.logger.info("SETTING UP GENERATION TEST ENVIRONMENT")
        self.logger.info("=" * 80)
        
        try:
            # Load config
            self.logger.info("Loading configuration...")
            self.config = get_default_config()
            self.logger.success("Configuration loaded")
            
            # Initialize generation engine
            self.logger.info("Initializing generation engine...")
            self.generation_engine = GenerationEngine(self.config)
            
            # Try to initialize (load model)
            if not self.generation_engine.initialize():
                self.logger.warning("Could not load LLM model - some tests will be skipped")
            else:
                self.logger.success("Generation engine initialized")
            
            # Create mock data
            self._create_mock_data()
            
            self.logger.success("Setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            import traceback
            self.logger.debug("Traceback", {
                "traceback": traceback.format_exc()
            })
            return False
    
    def _create_mock_data(self):
        """Create mock retrieved results for testing"""
        self.mock_results = [
            {
                'record': {
                    'global_id': 'test_001',
                    'regulation_type': 'Undang-Undang',
                    'regulation_number': '11',
                    'year': '2008',
                    'about': 'Informasi dan Transaksi Elektronik',
                    'enacting_body': 'Presiden Republik Indonesia',
                    'article': '27',
                    'chapter': 'VII',
                    'content': 'Setiap Orang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan yang melanggar kesusilaan.',
                    'effective_date': '2008-04-21'
                },
                'final_score': 0.89,
                'rerank_score': 0.85,
                'scores': {
                    'semantic': 0.82,
                    'keyword': 0.78,
                    'kg': 0.91,
                    'authority': 0.95,
                    'temporal': 0.85
                }
            },
            {
                'record': {
                    'global_id': 'test_002',
                    'regulation_type': 'Undang-Undang',
                    'regulation_number': '11',
                    'year': '2008',
                    'about': 'Informasi dan Transaksi Elektronik',
                    'enacting_body': 'Presiden Republik Indonesia',
                    'article': '45',
                    'chapter': 'XI',
                    'content': 'Setiap Orang yang memenuhi unsur sebagaimana dimaksud dalam Pasal 27 ayat (1), ayat (2), ayat (3), atau ayat (4) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah).',
                    'effective_date': '2008-04-21'
                },
                'final_score': 0.87,
                'rerank_score': 0.83,
                'scores': {
                    'semantic': 0.80,
                    'keyword': 0.76,
                    'kg': 0.89,
                    'authority': 0.95,
                    'temporal': 0.85
                }
            },
            {
                'record': {
                    'global_id': 'test_003',
                    'regulation_type': 'Undang-Undang',
                    'regulation_number': '19',
                    'year': '2016',
                    'about': 'Perubahan atas Undang-Undang Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik',
                    'enacting_body': 'Presiden Republik Indonesia',
                    'article': '5',
                    'chapter': 'II',
                    'content': 'Ketentuan Pasal 45 ayat (3) diubah sehingga berbunyi sebagai berikut: Setiap Orang yang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan penghinaan dan/atau pencemaran nama baik sebagaimana dimaksud dalam Pasal 27 ayat (3) dipidana dengan pidana penjara paling lama 4 (empat) tahun dan/atau denda paling banyak Rp750.000.000,00 (tujuh ratus lima puluh juta rupiah).',
                    'effective_date': '2016-11-25'
                },
                'final_score': 0.81,
                'rerank_score': 0.79,
                'scores': {
                    'semantic': 0.75,
                    'keyword': 0.72,
                    'kg': 0.85,
                    'authority': 0.92,
                    'temporal': 0.90
                }
            }
        ]
        
        self.logger.info("Mock data created", {
            "num_results": len(self.mock_results)
        })
    
    def test_prompt_builder(self) -> bool:
        """Test prompt builder component"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 1: PROMPT BUILDER")
        self.logger.info("=" * 80)
        
        try:
            prompt_builder = PromptBuilder(self.config)
            
            test_query = "Apa sanksi pidana untuk pelanggaran UU ITE?"
            
            # Test standard RAG prompt
            self.logger.info("Testing standard RAG prompt...")
            prompt = prompt_builder.build_prompt(
                query=test_query,
                retrieved_results=self.mock_results,
                template_type='rag_qa'
            )
            
            self.logger.success("Standard prompt built", {
                "prompt_length": len(prompt)
            })
            
            # Test with conversation history
            self.logger.info("Testing prompt with conversation history...")
            conv_history = [
                {'role': 'user', 'content': 'Apa itu UU ITE?'},
                {'role': 'assistant', 'content': 'UU ITE adalah...'}
            ]
            
            prompt_with_history = prompt_builder.build_prompt(
                query="Apa sanksi pelanggarannya?",
                retrieved_results=self.mock_results,
                conversation_history=conv_history,
                template_type='followup'
            )
            
            self.logger.success("Prompt with history built", {
                "prompt_length": len(prompt_with_history)
            })
            
            # Test context truncation
            self.logger.info("Testing context truncation...")
            truncated = prompt_builder.truncate_context(
                self.mock_results,
                max_tokens=1000
            )
            
            self.logger.success("Context truncation works", {
                "original": len(self.mock_results),
                "truncated": len(truncated)
            })
            
            # Display sample prompt
            self.logger.info("\n--- SAMPLE PROMPT (first 500 chars) ---")
            print(prompt[:500] + "...\n")
            
            self.logger.success("✅ Prompt Builder test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Prompt Builder test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_citation_formatter(self) -> bool:
        """Test citation formatter component"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 2: CITATION FORMATTER")
        self.logger.info("=" * 80)
        
        try:
            formatter = CitationFormatter(self.config)
            
            # Test different citation styles
            record = self.mock_results[0]['record']
            
            self.logger.info("Testing citation styles...")
            
            styles = ['standard', 'short', 'inline', 'bluebook']
            for style in styles:
                citation = formatter.format_citation(record, style=style)
                self.logger.info(f"  {style}: {citation}")
            
            # Test multiple citations
            self.logger.info("\nTesting multiple citations...")
            citations = formatter.format_citations(
                self.mock_results,
                style='standard',
                numbered=True
            )
            
            print("\n--- FORMATTED CITATIONS ---")
            print(citations)
            print()
            
            # Test citation extraction
            self.logger.info("Testing citation extraction...")
            test_text = "Berdasarkan UU No. 11/2008 dan PP Nomor 82 Tahun 2012..."
            extracted = formatter.extract_citations_from_text(test_text)
            
            self.logger.success("Citations extracted", {
                "found": len(extracted)
            })
            
            # Test bibliography generation
            self.logger.info("Testing bibliography generation...")
            bibliography = formatter.generate_bibliography(
                self.mock_results,
                style='standard'
            )
            
            print("\n--- BIBLIOGRAPHY ---")
            print(bibliography)
            print()
            
            self.logger.success("✅ Citation Formatter test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Citation Formatter test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_response_validator(self) -> bool:
        """Test response validator component"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 3: RESPONSE VALIDATOR")
        self.logger.info("=" * 80)
        
        try:
            validator = ResponseValidator(self.config)
            
            # Test valid response
            self.logger.info("Testing valid response...")
            valid_response = """Berdasarkan Undang-Undang Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik [Dokumen 1], sanksi pidana untuk pelanggaran yang dimaksud dalam Pasal 27 adalah sebagai berikut:

Setiap orang yang dengan sengaja dan tanpa hak mendistribusikan atau mentransmisikan informasi elektronik yang melanggar kesusilaan dapat dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 [Dokumen 2].

Untuk kasus atau situasi spesifik, disarankan untuk berkonsultasi dengan ahli hukum atau pengacara profesional."""
            
            validation = validator.validate_response(
                response=valid_response,
                query="Apa sanksi pidana UU ITE?",
                retrieved_results=self.mock_results,
                strict=False
            )
            
            self.logger.success("Valid response validated", {
                "is_valid": validation['is_valid'],
                "quality_score": f"{validation['quality_score']:.2f}",
                "checks_passed": len(validation['checks_passed'])
            })
            
            # Test invalid response (too short)
            self.logger.info("Testing invalid response...")
            invalid_response = "Tidak tahu."
            
            validation_invalid = validator.validate_response(
                response=invalid_response,
                query="Apa sanksi pidana UU ITE?",
                retrieved_results=self.mock_results,
                strict=True
            )
            
            self.logger.info("Invalid response validated", {
                "is_valid": validation_invalid['is_valid'],
                "errors": len(validation_invalid['errors'])
            })
            
            # Test quality report
            self.logger.info("Testing quality report generation...")
            report = validator.get_quality_report(validation)
            
            print("\n--- QUALITY REPORT ---")
            print(report)
            print()
            
            # Test sanitization
            self.logger.info("Testing response sanitization...")
            dirty_response = "<think>Internal reasoning...</think>\n\nThe answer is..."
            clean = validator.sanitize_response(dirty_response)
            
            self.logger.success("Response sanitized", {
                "original_length": len(dirty_response),
                "cleaned_length": len(clean)
            })
            
            self.logger.success("✅ Response Validator test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Response Validator test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_full_generation(self) -> bool:
        """Test complete generation pipeline"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 4: FULL GENERATION PIPELINE")
        self.logger.info("=" * 80)
        
        # Check if model is loaded
        if not self.generation_engine.llm_engine._model:
            self.logger.warning("LLM model not loaded - skipping full generation test")
            self.logger.info("To run this test, ensure LLM model is available")
            return True
        
        try:
            test_query = "Apa sanksi pidana untuk pelanggaran Pasal 27 UU ITE?"
            
            self.logger.info(f"Test query: {test_query}")
            self.logger.info("Generating answer...")
            
            start_time = time.time()
            
            result = self.generation_engine.generate_answer(
                query=test_query,
                retrieved_results=self.mock_results,
                stream=False
            )
            
            generation_time = time.time() - start_time
            
            if result['success']:
                self.logger.success("Answer generated successfully", {
                    "generation_time": f"{generation_time:.2f}s",
                    "answer_length": len(result['answer']),
                    "tokens_generated": result['metadata'].get('tokens_generated', 0)
                })
                
                # Display answer
                print("\n" + "=" * 80)
                print("GENERATED ANSWER")
                print("=" * 80)
                print(result['answer'])
                print("\n" + "=" * 80)
                
                # Display metadata
                print("\nMETADATA:")
                print(f"  Generation time: {result['metadata']['generation_time']:.2f}s")
                print(f"  Total time: {result['metadata']['total_time']:.2f}s")
                print(f"  Tokens: {result['metadata']['tokens_generated']}")
                print(f"  Speed: {result['metadata']['tokens_per_second']:.1f} tok/s")
                
                # Display citations
                print("\nCITATIONS:")
                for citation in result['citations']:
                    print(f"  [{citation['id']}] {citation['citation_text']}")
                
                # Display sources
                print("\nSOURCES:")
                for source in result['sources']:
                    print(f"  {source['id']}. {source['title']} (Score: {source['relevance_score']:.3f})")
                
                print()
                
                self.logger.success("✅ Full Generation test PASSED")
                return True
            else:
                self.logger.error("Generation failed", {
                    "error": result.get('error')
                })
                return False
            
        except Exception as e:
            self.logger.error(f"Full Generation test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_streaming_generation(self) -> bool:
        """Test streaming generation"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 5: STREAMING GENERATION")
        self.logger.info("=" * 80)
        
        # Check if model is loaded
        if not self.generation_engine.llm_engine._model:
            self.logger.warning("LLM model not loaded - skipping streaming test")
            return True
        
        try:
            test_query = "Jelaskan sanksi dalam UU ITE secara singkat"
            
            self.logger.info(f"Test query: {test_query}")
            self.logger.info("Starting streaming generation...")
            
            print("\n--- STREAMING OUTPUT ---")
            
            full_answer = ""
            tokens = 0
            
            for chunk in self.generation_engine.generate_answer(
                query=test_query,
                retrieved_results=self.mock_results[:2],  # Use fewer for speed
                stream=True
            ):
                if chunk['type'] == 'token':
                    print(chunk['token'], end='', flush=True)
                    full_answer += chunk['token']
                    tokens = chunk['tokens_generated']
                elif chunk['type'] == 'complete':
                    print("\n")
                    self.logger.success("Streaming completed", {
                        "tokens": tokens,
                        "time": f"{chunk.get('generation_time', 0):.2f}s"
                    })
                elif chunk['type'] == 'error':
                    self.logger.error(f"Streaming error: {chunk['error']}")
                    return False
            
            print("--- END STREAMING ---\n")
            
            self.logger.success("✅ Streaming Generation test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Streaming Generation test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_follow_up_suggestions(self) -> bool:
        """Test follow-up suggestion generation"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 6: FOLLOW-UP SUGGESTIONS")
        self.logger.info("=" * 80)
        
        try:
            test_query = "Apa sanksi pidana UU ITE?"
            test_answer = "Sanksi pidana dalam UU ITE mencakup..."
            
            self.logger.info("Generating follow-up suggestions...")
            
            suggestions = self.generation_engine.generate_follow_up_suggestions(
                query=test_query,
                answer=test_answer,
                retrieved_results=self.mock_results
            )
            
            self.logger.success("Suggestions generated", {
                "count": len(suggestions)
            })
            
            print("\n--- FOLLOW-UP SUGGESTIONS ---")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
            print()
            
            self.logger.success("✅ Follow-up Suggestions test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Follow-up Suggestions test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RUNNING ALL GENERATION MODULE TESTS")
        self.logger.info("=" * 80)
        
        tests = [
            ("Prompt Builder", self.test_prompt_builder),
            ("Citation Formatter", self.test_citation_formatter),
            ("Response Validator", self.test_response_validator),
            ("Full Generation", self.test_full_generation),
            ("Streaming Generation", self.test_streaming_generation),
            ("Follow-up Suggestions", self.test_follow_up_suggestions)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                self.logger.error(f"Test {test_name} crashed: {e}")
                results[test_name] = False
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        if self.generation_engine:
            self.generation_engine.shutdown()
        self.logger.success("Cleanup completed")


def main():
    """Main test execution"""
    # Initialize logging
    initialize_logging(
        enable_file_logging=True,
        log_dir="logs",
        append=False,
        log_filename="generation_test.log"
    )
    
    logger = get_logger("Main")
    logger.info("=" * 80)
    logger.info("GENERATION MODULE TEST SUITE")
    logger.info("=" * 80)
    
    # Create tester
    tester = GenerationTester()
    
    # Setup
    if not tester.setup():
        logger.error("Setup failed, aborting tests")
        return False
    
    # Run tests
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    print("=" * 80)
    
    logger.info(f"Test results: {passed} passed, {failed} failed")
    
    # Cleanup
    tester.cleanup()
    
    # End session
    log_session_end()
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)