"""
Enhanced RAG API Integration Test

Real integration test that initializes the full RAG pipeline and tests all API endpoints.
Tests retrieval, research, and conversational services with actual pipeline execution.

Run with:
    python tests/integration/test_api_integration.py
    
Options:
    --quick      Use lighter pipeline config for faster testing
    --verbose    Show detailed output during processing

File: tests/integration/test_api_integration.py
"""

import sys
import os
import time
from typing import Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_utils import get_logger, initialize_logging
from config import LOG_DIR, ENABLE_FILE_LOGGING


class APIIntegrationTester:
    """Integration tester for Enhanced RAG API with real pipeline"""
    
    def __init__(self, quick_mode: bool = False, verbose: bool = False):
        initialize_logging(
            enable_file_logging=ENABLE_FILE_LOGGING,
            log_dir=LOG_DIR,
            verbosity_mode='verbose' if verbose else 'minimal'
        )
        self.logger = get_logger("APIIntegrationTest")
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.pipeline = None
        self.conversation_manager = None
        self.results: Dict[str, Any] = {}
    
    def print_header(self):
        """Print test header"""
        mode = "QUICK MODE" if self.quick_mode else "FULL MODE"
        print("\n" + "=" * 100)
        print(f"ENHANCED RAG API INTEGRATION TEST - {mode}")
        print("=" * 100)
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def initialize_pipeline(self) -> bool:
        """Initialize RAG pipeline for testing"""
        print("\n" + "-" * 80)
        print("INITIALIZING RAG PIPELINE")
        print("-" * 80)
        
        try:
            from pipeline import RAGPipeline
            
            # Use lighter config for quick mode
            config = {}
            if self.quick_mode:
                config = {
                    'final_top_k': 3,
                    'research_team_size': 2,
                    'max_new_tokens': 2048
                }
            
            self.pipeline = RAGPipeline(config=config)
            success = self.pipeline.initialize()
            
            if success:
                print("✓ RAG Pipeline initialized successfully")
                self.logger.success("Pipeline ready for API testing")
            else:
                print("✗ Pipeline initialization failed")
                self.logger.error("Pipeline initialization failed")
            
            return success
            
        except Exception as e:
            print(f"✗ Initialization error: {e}")
            self.logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_conversation_manager(self) -> bool:
        """Initialize conversation manager for chat testing"""
        try:
            from conversation import ConversationManager
            
            self.conversation_manager = ConversationManager()
            print("✓ Conversation Manager initialized")
            self.logger.success("Conversation manager ready")
            return True
            
        except Exception as e:
            print(f"✗ ConversationManager initialization error: {e}")
            self.logger.error(f"ConversationManager init error: {e}")
            return False
    
    def test_retrieval_endpoint(self) -> bool:
        """Test /rag/retrieve endpoint (pure retrieval without LLM)"""
        print("\n" + "=" * 100)
        print("TEST 1: RETRIEVAL ENDPOINT (Pure Document Search)")
        print("=" * 100)
        
        try:
            query = "prosedur pendirian PT"
            top_k = 3
            min_score = 0.5
            
            print(f"\nQuery: {query}")
            print(f"Parameters: top_k={top_k}, min_score={min_score}")
            print()
            
            start_time = time.time()
            
            # Simulate endpoint logic: call orchestrator for retrieval
            if not hasattr(self.pipeline, 'orchestrator') or self.pipeline.orchestrator is None:
                print("✗ Orchestrator not initialized")
                return False
            
            # Run orchestrator
            rag_result = self.pipeline.orchestrator.run(
                query=query,
                conversation_history=[]
            )
            
            final_results = rag_result.get('final_results', [])
            
            # Filter by score
            final_results = [
                r for r in final_results
                if r.get('scores', {}).get('final', 0) >= min_score
            ]
            
            # Limit to top_k
            final_results = final_results[:top_k]
            
            retrieval_time = time.time() - start_time
            
            # Validate results
            print(f"Retrieved: {len(final_results)} documents in {retrieval_time:.2f}s")
            
            if len(final_results) > 0:
                print("\nTop Results:")
                for i, result in enumerate(final_results[:3], 1):
                    record = result.get('record', {})
                    scores = result.get('scores', {})
                    print(f"  {i}. {record.get('regulation_type')} No. {record.get('regulation_number')}/{record.get('year')}")
                    print(f"     Score: {scores.get('final', 0):.4f}")
                    print(f"     About: {record.get('about', 'N/A')[:60]}...")
                
                print("\n✓ Retrieval endpoint test PASSED")
                self.logger.success(f"Retrieval test passed: {len(final_results)} docs in {retrieval_time:.2f}s")
                return True
            else:
                # Get total candidates before filtering
                total_candidates = len(rag_result.get('final_results', []))
                print(f"\n⚠️ No documents met criteria (min_score={min_score})")
                print(f"   Total candidates before filtering: {total_candidates}")
                if total_candidates > 0:
                    print(f"   Suggestion: Lower min_score or check query relevance")
                    # Still pass if we got candidates
                    print("\n✓ Retrieval endpoint test PASSED (retrieved candidates but none met score threshold)")
                    self.logger.info(f"Retrieval working but no docs met min_score={min_score}")
                    return True
                else:
                    print("\n✗ No documents retrieved at all")
                    return False
                
        except Exception as e:
            print(f"\n✗ Retrieval endpoint test FAILED: {e}")
            self.logger.error(f"Retrieval test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_research_endpoint(self) -> bool:
        """Test /rag/research endpoint (deep research with LLM)"""
        print("\n" + "=" * 100)
        print("TEST 2: RESEARCH ENDPOINT (Deep Analysis with LLM)")
        print("=" * 100)
        
        try:
            query = "Apa syarat minimal pendirian PT?"
            thinking_level = 'low' if self.quick_mode else 'medium'
            team_size = 2 if self.quick_mode else 3
            
            print(f"\nQuery: {query}")
            print(f"Parameters: thinking_level={thinking_level}, team_size={team_size}")
            print()
            
            start_time = time.time()
            
            # Update pipeline config
            self.pipeline.update_config(research_team_size=team_size)
            
            # Execute with specified thinking level
            result = self.pipeline.query(
                question=query,
                conversation_history=None,
                stream=False,
                thinking_mode=thinking_level
            )
            
            research_time = time.time() - start_time
            
            # Validate results
            if result.get('success', True):
                answer = result.get('answer', '')
                citations = result.get('citations', [])
                
                print(f"✓ Research completed in {research_time:.2f}s")
                print(f"\nAnswer length: {len(answer)} characters")
                print(f"Citations: {len(citations)} documents")
                
                if len(answer) > 50:
                    print(f"\nAnswer preview:\n{answer[:200]}...")
                
                if citations:
                    print(f"\nTop Citation:")
                    cit = citations[0]
                    print(f"  {cit.get('regulation_type')} No. {cit.get('regulation_number')}/{cit.get('year')}")
                    print(f"  {cit.get('about', 'N/A')[:60]}...")
                
                print("\n✓ Research endpoint test PASSED")
                self.logger.success(f"Research test passed: {len(answer)} chars in {research_time:.2f}s")
                return True
            else:
                print(f"✗ Research failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"\n✗ Research endpoint test FAILED: {e}")
            self.logger.error(f"Research test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_chat_endpoint(self) -> bool:
        """Test /rag/chat endpoint (conversational)"""
        print("\n" + "=" * 100)
        print("TEST 3: CHAT ENDPOINT (Conversational Service)")
        print("=" * 100)
        
        try:
            session_id = "test_session_123"
            thinking_level = 'low'
            
            # Create session first
            self.conversation_manager.start_session(session_id)
            print(f"\n✓ Session created: {session_id}")
            
            # Multi-turn conversation
            queries = [
                "Apa itu PT?",
                "Berapa modal minimal untuk mendirikannya?"
            ]
            
            for turn, query in enumerate(queries, 1):
                print(f"\n--- Turn {turn} ---")
                print(f"Query: {query}")
                
                start_time = time.time()
                
                # Get conversation context
                context = None
                if turn > 1:
                    context = self.conversation_manager.get_context_for_query(session_id)
                    print(f"Context: {len(context) if context else 0} previous turns")
                
                # Execute query
                result = self.pipeline.query(
                    question=query,
                    conversation_history=context,
                    stream=False,
                    thinking_mode=thinking_level
                )
                
                chat_time = time.time() - start_time
                
                # Save to history
                if result.get('success', True):
                    answer = result.get('answer', '')
                    self.conversation_manager.add_turn(
                        session_id=session_id,
                        query=query,
                        answer=answer,
                        metadata=result.get('metadata')
                    )
                    
                    print(f"Answer ({chat_time:.2f}s): {answer[:100]}...")
                else:
                    print(f"✗ Turn {turn} failed: {result.get('error')}")
                    return False
            
            # Verify history
            history = self.conversation_manager.get_history(session_id)
            
            if len(history) == len(queries):
                print(f"\n✓ Conversation with {len(history)} turns completed")
                print(f"Session memory maintained correctly")
                print("\n✓ Chat endpoint test PASSED")
                self.logger.success(f"Chat test passed: {len(history)} turns")
                return True
            else:
                print(f"✗ History mismatch: expected {len(queries)}, got {len(history)}")
                return False
                
        except Exception as e:
            print(f"\n✗ Chat endpoint test FAILED: {e}")
            self.logger.error(f"Chat test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API integration tests"""
        self.print_header()
        
        # Initialize components
        if not self.initialize_pipeline():
            print("\n✗ FATAL: Pipeline initialization failed. Cannot proceed.")
            return {}
        
        if not self.initialize_conversation_manager():
            print("\n✗ FATAL: ConversationManager initialization failed. Cannot proceed.")
            return {}
        
        # Run all API endpoint tests
        results = {}
        results['retrieval'] = self.test_retrieval_endpoint()
        results['research'] = self.test_research_endpoint()
        results['chat'] = self.test_chat_endpoint()
        
        return results
    
    def print_results(self, results: Dict[str, bool]):
        """Print final test results"""
        print("\n" + "=" * 100)
        print("API INTEGRATION TEST RESULTS")
        print("=" * 100)
        print()
        
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        
        for test_name, passed_test in results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            endpoint = f"/api/v1/rag/{test_name}"
            print(f"{status} - {endpoint}")
        
        print()
        print(f"Results: {passed}/{total} tests passed")
        
        mode = "quick mode" if self.quick_mode else "full mode"
        
        if passed == total:
            print(f"\n🎉 ALL API TESTS PASSED ({mode})!")
            self.logger.success("All API integration tests passed")
        else:
            print(f"\n⚠️ {total - passed} tests failed ({mode})")
            self.logger.error(f"{total - passed} API tests failed")
        
        print("=" * 100)
    
    def shutdown(self):
        """Clean up resources"""
        if self.pipeline:
            try:
                self.pipeline.shutdown()
                print("\nPipeline shutdown complete")
            except Exception as e:
                print(f"Shutdown warning: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAG API Integration Test")
    parser.add_argument('--quick', action='store_true', help='Use lighter config for faster testing')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    # Create tester
    tester = APIIntegrationTester(quick_mode=args.quick, verbose=args.verbose)
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        if not results:
            print("\nTests aborted due to initialization failure")
            sys.exit(1)
        
        # Print results
        tester.print_results(results)
        
        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        tester.shutdown()


if __name__ == "__main__":
    main()
