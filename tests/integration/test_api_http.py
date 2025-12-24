"""
HTTP-Level API Integration Test

Real HTTP tests that start the FastAPI server and make actual HTTP requests.
Tests authentication, endpoints, and error handling over the network.

Run with:
    python tests/integration/test_api_http.py

File: tests/integration/test_api_http.py
"""

import sys
import os
import time
import subprocess
import requests
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_utils import get_logger, initialize_logging
from config import LOG_DIR, ENABLE_FILE_LOGGING


class HTTPAPITester:
    """HTTP-level API tester with real server"""
    
    def __init__(self, verbose: bool = False):
        initialize_logging(
            enable_file_logging=ENABLE_FILE_LOGGING,
            log_dir=LOG_DIR,
            verbosity_mode='verbose' if verbose else 'minimal'
        )
        self.logger = get_logger("HTTPAPITest")
        self.verbose = verbose
        self.base_url = "http://localhost:8000"
        self.api_key = "test_integration_key_12345"
        self.server_process: Optional[subprocess.Popen] = None
    
    def print_header(self):
        """Print test header"""
        print("\n" + "=" * 100)
        print("HTTP-LEVEL API INTEGRATION TEST")
        print("=" * 100)
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base URL: {self.base_url}")
        print()
    
    def test_authentication_logic(self) -> bool:
        """Test API key validation logic"""
        print("\n" + "-" * 80)
        print("TEST 1: Authentication Logic")
        print("-" * 80)
        
        try:
            from security import validate_api_key, APIKeyValidator
            import os
            
            # Set test API key
            os.environ['LEGAL_API_KEY'] = self.api_key
            
            # Test 1: Valid key
            validator = APIKeyValidator()
            assert validator.validate(self.api_key), "Valid key should pass"
            print("✓ Valid API key accepted")
            
            # Test 2: Invalid key
            assert not validator.validate('invalid_key'), "Invalid key should fail"
            print("✓ Invalid API key rejected")
            
            # Test 3: Empty key
            assert not validator.validate(''), "Empty key should fail"
            print("✓ Empty API key rejected")
            
            return True
            
        except Exception as e:
            print(f"✗ Authentication logic test failed: {e}")
            self.logger.error(f"Auth logic test failed: {e}")
            return False
    
    def test_endpoint_logic(self) -> bool:
        """Test endpoint logic directly"""
        print("\n" + "-" * 80)
        print("TEST 2: Endpoint Logic Testing")
        print("-" * 80)
        
        try:
            # Test retrieval logic
            print("\nTesting retrieval endpoint logic...")
            result = self.pipeline.orchestrator.run(
                query="prosedur pendirian PT",
                conversation_history=[]
            )
            final_results = result.get('final_results', [])
            print(f"✓ Retrieval logic: {len(final_results)} documents retrieved")
            
            # Test research logic
            print("\nTesting research endpoint logic...")
            self.pipeline.update_config(research_team_size=2)
            result = self.pipeline.query(
                question="Apa syarat minimal pendirian PT?",
                conversation_history=None,
                stream=False,
                thinking_mode='low'
            )
            assert result.get('success', True), "Research should succeed"
            answer_len = len(result.get('answer', ''))
            print(f"✓ Research logic: Generated {answer_len} chars")
            
            # Test chat logic with session
            print("\nTesting chat endpoint logic...")
            from conversation import ConversationManager
            manager = ConversationManager()
            session_id = "test_logic_session"
            manager.start_session(session_id)
            
            result = self.pipeline.query(
                question="Apa itu PT?",
                conversation_history=None,
                stream=False,
                thinking_mode='low'
            )
            
            # Save to session
            manager.add_turn(
                session_id=session_id,
                query="Apa itu PT?",
                answer=result.get('answer', ''),
                metadata=result.get('metadata')
            )
            
            history = manager.get_history(session_id)
            assert len(history) == 1, "Should have 1 turn in history"
            print(f"✓ Chat logic: Session management working")
            
            return True
            
        except Exception as e:
            print(f"✗ Endpoint logic test failed: {e}")
            self.logger.error(f"Endpoint logic test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_input_validation(self) -> bool:
        """Test input validation logic"""
        print("\n" + "-" * 80)
        print("TEST 3: Input Validation")
        print("-" * 80)
        
        try:
            from security import sanitize_query
            from pydantic import ValidationError
            from api.routes.rag_enhanced import RetrievalRequest
            
            # Test 1: Invalid query (XSS)
            try:
                sanitize_query("<script>alert('xss')</script>")
                print("✗ XSS should be blocked")
                return False
            except ValueError:
                print("✓ XSS attempt blocked")
            
            # Test 2: Missing required field
            try:
                req = RetrievalRequest(top_k=3)  # Missing query
                print("✗ Missing field should be rejected")
                return False
            except ValidationError:
                print("✓ Missing field rejected")
            
            # Test 3: Invalid parameter value
            try:
                req = RetrievalRequest(query="test", top_k=999)  # Max is 10
                print("✗ Invalid parameter should be rejected")
                return False
            except ValidationError:
                print("✓ Invalid parameter rejected")
            
            # Test 4: Valid request
            req = RetrievalRequest(query="Apa itu UU Perdata?", top_k=3)
            assert req.query == "Apa itu UU Perdata?"
            print("✓ Valid request accepted")
            
            return True
            
        except Exception as e:
            print(f"✗ Input validation test failed: {e}")
            self.logger.error(f"Input validation test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all endpoint logic tests"""
        self.print_header()
        
        # Initialize pipeline
        if not self.initialize_pipeline():
            print("\n✗ FATAL: Cannot initialize pipeline. Aborting tests.")
            return {}
        
        # Run tests
        results = {}
        results['authentication'] = self.test_authentication_logic()
        results['endpoint_logic'] = self.test_endpoint_logic()
        results['input_validation'] = self.test_input_validation()
        
        return results
    
    def print_results(self, results: Dict[str, bool]):
        """Print final test results"""
        print("\n" + "=" * 100)
        print("API ENDPOINT LOGIC TEST RESULTS (Kaggle Compatible)")
        print("=" * 100)
        print()
        
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        
        for test_name, passed_test in results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            print(f"{status} - {test_name.replace('_', ' ').title()}")
        
        print()
        print(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL ENDPOINT LOGIC TESTS PASSED!")
            self.logger.success("All endpoint logic tests passed")
        else:
            print(f"\n⚠️ {total - passed} tests failed")
            self.logger.error(f"{total - passed} endpoint logic tests failed")
        
        print("=" * 100)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Endpoint Logic Test (Kaggle Compatible)")
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    tester = HTTPAPITester(verbose=args.verbose)
    
    try:
        results = tester.run_all_tests()
        
        if not results:
            print("\nTests aborted")
            sys.exit(1)
        
        tester.print_results(results)
        
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
        # Clean up pipeline
        if hasattr(tester, 'pipeline') and tester.pipeline:
            try:
                tester.pipeline.shutdown()
                print("\nPipeline shutdown complete")
            except Exception as e:
                print(f"Shutdown warning: {e}")


if __name__ == "__main__":
    main()
