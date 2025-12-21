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
    
    def start_server(self) -> bool:
        """Start FastAPI server in background"""
        print("\n" + "-" * 80)
        print("Starting FastAPI Server")
        print("-" * 80)
        
        try:
            # Set API key in environment
            os.environ['LEGAL_API_KEY'] = self.api_key
            
            # Start uvicorn server
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            
            # Wait for server to start
            print("Waiting for server to start...")
            max_attempts = 30
            for i in range(max_attempts):
                try:
                    response = requests.get(f"{self.base_url}/api/v1/health", timeout=1)
                    if response.status_code == 200:
                        print(f"✓ Server started successfully (attempt {i+1}/{max_attempts})")
                        self.logger.success("FastAPI server is running")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            print("✗ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"✗ Failed to start server: {e}")
            self.logger.error(f"Server start failed: {e}")
            return False
    
    def stop_server(self):
        """Stop FastAPI server"""
        if self.server_process:
            print("\nStopping FastAPI server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("✓ Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("✓ Server killed")
    
    def test_authentication(self) -> bool:
        """Test API key authentication over HTTP"""
        print("\n" + "-" * 80)
        print("TEST 1: HTTP Authentication")
        print("-" * 80)
        
        try:
            # Test 1: No API key
            response = requests.get(f"{self.base_url}/api/v1/rag/retrieve", timeout=5)
            assert response.status_code == 401, "Should return 401 without API key"
            print("✓ Request without API key rejected (401)")
            
            # Test 2: Invalid API key
            response = requests.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                headers={"X-API-Key": "invalid_key"},
                json={"query": "test", "top_k": 3},
                timeout=5
            )
            assert response.status_code == 401, "Should return 401 with invalid key"
            print("✓ Request with invalid API key rejected (401)")
            
            # Test 3: Valid API key
            response = requests.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                headers={"X-API-Key": self.api_key},
                json={"query": "Apa itu UU Perdata?", "top_k": 3, "min_score": 0.5},
                timeout=30
            )
            assert response.status_code == 200, f"Should return 200 with valid key, got {response.status_code}"
            print("✓ Request with valid API key accepted (200)")
            
            # Validate response structure
            data = response.json()
            assert 'query' in data, "Response should contain query"
            assert 'documents' in data, "Response should contain documents"
            print(f"✓ Response structure valid (retrieved {len(data.get('documents', []))} docs)")
            
            return True
            
        except Exception as e:
            print(f"✗ Authentication test failed: {e}")
            self.logger.error(f"HTTP auth test failed: {e}")
            return False
    
    def test_endpoints(self) -> bool:
        """Test all API endpoints over HTTP"""
        print("\n" + "-" * 80)
        print("TEST 2: HTTP Endpoint Testing")
        print("-" * 80)
        
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        
        try:
            # Test 1: Retrieval endpoint
            print("\nTesting /api/v1/rag/retrieve...")
            response = requests.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                headers=headers,
                json={"query": "prosedur pendirian PT", "top_k": 3, "min_score": 0.0},
                timeout=30
            )
            assert response.status_code == 200, f"Retrieval failed: {response.status_code}"
            data = response.json()
            print(f"✓ Retrieval endpoint: {data.get('total_retrieved', 0)} docs in {data.get('search_time', 0):.2f}s")
            
            # Test 2: Research endpoint (quick)
            print("\nTesting /api/v1/rag/research...")
            response = requests.post(
                f"{self.base_url}/api/v1/rag/research",
                headers=headers,
                json={
                    "query": "Apa syarat minimal pendirian PT?",
                    "thinking_level": "low",
                    "team_size": 2
                },
                timeout=120  # Longer timeout for LLM
            )
            assert response.status_code == 200, f"Research failed: {response.status_code}"
            data = response.json()
            print(f"✓ Research endpoint: {len(data.get('answer', ''))} chars in {data.get('research_time', 0):.2f}s")
            assert len(data.get('legal_references', '')) > 0, "Should include legal references"
            print(f"✓ Legal references included")
            
            # Test 3: Chat endpoint (non-streaming)
            print("\nTesting /api/v1/rag/chat...")
            response = requests.post(
                f"{self.base_url}/api/v1/rag/chat",
                headers=headers,
                json={
                    "query": "Apa itu PT?",
                    "session_id": "http_test_session",
                    "thinking_level": "low",
                    "stream": False
                },
                timeout=120
            )
            assert response.status_code == 200, f"Chat failed: {response.status_code}"
            data = response.json()
            print(f"✓ Chat endpoint: {len(data.get('answer', ''))} chars")
            assert data.get('session_id') == "http_test_session", "Should preserve session_id"
            print(f"✓ Session management working")
            
            return True
            
        except Exception as e:
            print(f"✗ Endpoint test failed: {e}")
            self.logger.error(f"HTTP endpoint test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_error_handling(self) -> bool:
        """Test error responses over HTTP"""
        print("\n" + "-" * 80)
        print("TEST 3: HTTP Error Handling")
        print("-" * 80)
        
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        
        try:
            # Test 1: Invalid JSON
            response = requests.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                headers=headers,
                data="invalid json",
                timeout=5
            )
            assert response.status_code == 422, "Should return 422 for invalid JSON"
            print("✓ Invalid JSON handled (422)")
            
            # Test 2: Missing required field
            response = requests.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                headers=headers,
                json={"top_k": 3},  # Missing 'query'
                timeout=5
            )
            assert response.status_code == 422, "Should return 422 for missing field"
            print("✓ Missing field handled (422)")
            
            # Test 3: Invalid parameter value
            response = requests.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                headers=headers,
                json={"query": "test", "top_k": 999},  # top_k max is 10
                timeout=5
            )
            assert response.status_code == 422, "Should return 422 for invalid parameter"
            print("✓ Invalid parameter handled (422)")
            
            # Test 4: XSS attempt
            response = requests.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                headers=headers,
                json={"query": "<script>alert('xss')</script>", "top_k": 3},
                timeout=5
            )
            assert response.status_code == 422, "Should reject XSS attempts"
            print("✓ XSS attempt blocked (422)")
            
            return True
            
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            self.logger.error(f"HTTP error handling test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all HTTP-level tests"""
        self.print_header()
        
        # Start server
        if not self.start_server():
            print("\n✗ FATAL: Cannot start server. Aborting tests.")
            return {}
        
        try:
            # Run tests
            results = {}
            results['authentication'] = self.test_authentication()
            results['endpoints'] = self.test_endpoints()
            results['error_handling'] = self.test_error_handling()
            
            return results
            
        finally:
            self.stop_server()
    
    def print_results(self, results: Dict[str, bool]):
        """Print final test results"""
        print("\n" + "=" * 100)
        print("HTTP-LEVEL API TEST RESULTS")
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
            print("\n🎉 ALL HTTP TESTS PASSED!")
            self.logger.success("All HTTP-level tests passed")
        else:
            print(f"\n⚠️ {total - passed} tests failed")
            self.logger.error(f"{total - passed} HTTP tests failed")
        
        print("=" * 100)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HTTP-Level API Integration Test")
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
        tester.stop_server()
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        tester.stop_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
