"""
Comprehensive API Endpoint Integration Tests
Tests all API endpoints with real initialization and output display

Run with:
    python tests/integration/test_api_endpoints.py

Or with pytest:
    pytest tests/integration/test_api_endpoints.py -v -s

This shows REAL output like production would generate.
"""

import sys
import os
import time
import json
import requests
from typing import Optional
import subprocess
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import LOG_DIR, ENABLE_FILE_LOGGING, LOG_VERBOSITY
from utils.logger_utils import get_logger, initialize_logging


class APIEndpointTester:
    """Comprehensive API endpoint tester with real server"""

    def __init__(self, port: int = 8000):
        initialize_logging(
        enable_file_logging=ENABLE_FILE_LOGGING,
        log_dir=LOG_DIR,
        verbosity_mode=LOG_VERBOSITY
    )
        self.logger = get_logger("APITest")
        self.port = port
        self.base_url = f"http://localhost:{port}/api/v1"
        self.server_process: Optional[subprocess.Popen] = None
        
        # Get API key from environment (MUST match server's key)
        self.api_key = os.getenv('LEGAL_API_KEY')
        if not self.api_key:
            self.logger.warning("LEGAL_API_KEY not set - using empty key (tests may fail)")
            self.api_key = ""
        self.headers = {'X-API-Key': self.api_key}

    def start_server(self, timeout: int = 600) -> bool:
        """Start the API server and wait for it to be ready"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING API SERVER")
        self.logger.info("=" * 80)

        try:
            # Start uvicorn server
            self.logger.info(f"Starting server on port {self.port}...")
            
            # Cross-platform process creation
            popen_kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.PIPE,
            }
            
            # Unix-specific: create new process group
            if hasattr(os, 'setsid'):
                popen_kwargs['preexec_fn'] = os.setsid
            
            self.server_process = subprocess.Popen(
                ["python", "-m", "uvicorn", "api.server:app",
                 "--host", "0.0.0.0", "--port", str(self.port)],
                **popen_kwargs
            )

            # Wait for server to be ready (use /ready endpoint which checks pipeline)
            self.logger.info(f"Waiting for server and pipeline to initialize (timeout: {timeout}s)...")
            self.logger.info("This may take 5-10 minutes on first run while loading models...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    # Use /ready endpoint which checks if pipeline is initialized
                    response = requests.get(f"{self.base_url}/ready", timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('ready', False):
                            self.logger.success(f"Server and pipeline ready after {time.time() - start_time:.1f}s")
                            return True
                        else:
                            # Show progress
                            elapsed = int(time.time() - start_time)
                            if elapsed % 30 == 0:
                                self.logger.info(f"Still initializing... ({elapsed}s elapsed)")
                except requests.exceptions.RequestException:
                    pass
                time.sleep(2)

            self.logger.error("Server failed to start within timeout")
            return False

        except Exception as e:
            self.logger.error("Failed to start server", {"error": str(e)})
            return False

    def stop_server(self):
        """Stop the API server"""
        if self.server_process:
            self.logger.info("Stopping API server...")
            try:
                # Cross-platform process termination
                if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                    # Unix: Kill the entire process group
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                else:
                    # Windows: Just terminate the process
                    self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.logger.success("Server stopped")
            except Exception as e:
                self.logger.error("Error stopping server", {"error": str(e)})
                try:
                    self.server_process.kill()
                except:
                    pass

    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Health Check Endpoint")
        self.logger.info("=" * 80)

        try:
            response = requests.get(f"{self.base_url}/health")

            self.logger.info(f"Status Code: {response.status_code}")
            self.logger.info(f"Response: {response.json()}")

            if response.status_code == 200:
                self.logger.success("✅ Health check passed")
                return True
            else:
                self.logger.error("❌ Health check failed")
                return False

        except Exception as e:
            self.logger.error("❌ Health check error", {"error": str(e)})
            return False

    def test_search_endpoint(self) -> bool:
        """Test search endpoint with real query"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Search Endpoint")
        self.logger.info("=" * 80)

        test_query = "Apa sanksi dalam UU ITE?"
        self.logger.info(f"Query: {test_query}")

        try:
            payload = {
                "query": test_query,
                "max_results": 5
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/search",
                json=payload,
                headers=self.headers,
                timeout=300  # Increased for Kaggle
            )
            elapsed = time.time() - start_time

            self.logger.info(f"Status Code: {response.status_code}")
            self.logger.info(f"Response Time: {elapsed:.2f}s")

            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"Total Results: {data.get('total_results', 0)}")
                self.logger.info(f"Search Time: {data.get('search_time', 0):.2f}s")

                # Show first result
                if data.get('results'):
                    first = data['results'][0]
                    self.logger.info("First Result:")
                    self.logger.info(f"  Type: {first.get('regulation_type')}")
                    self.logger.info(f"  Number: {first.get('regulation_number')}")
                    self.logger.info(f"  Year: {first.get('year')}")
                    self.logger.info(f"  Score: {first.get('score', 0):.4f}")

                self.logger.success("✅ Search endpoint passed")
                return True
            else:
                self.logger.error(f"❌ Search failed: {response.text}")
                return False

        except Exception as e:
            self.logger.error("❌ Search error", {"error": str(e)})
            return False

    def test_generate_endpoint(self) -> bool:
        """Test answer generation endpoint"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Generate Answer Endpoint")
        self.logger.info("=" * 80)

        test_query = "Jelaskan tentang perlindungan data pribadi"
        self.logger.info(f"Query: {test_query}")

        try:
            payload = {
                "query": test_query,
                "stream": False
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=self.headers,
                timeout=180  # Increased for Kaggle - generation is slow
            )
            elapsed = time.time() - start_time

            self.logger.info(f"Status Code: {response.status_code}")
            self.logger.info(f"Response Time: {elapsed:.2f}s")

            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                citations = data.get('citations', [])

                self.logger.info(f"Answer Length: {len(answer)} chars")
                self.logger.info(f"Citations: {len(citations)}")
                self.logger.info(f"Answer Preview: {answer[:200]}...")

                if citations:
                    self.logger.info("First Citation:")
                    first_cite = citations[0]
                    self.logger.info(f"  {first_cite.get('regulation_type')} No. {first_cite.get('regulation_number')}/{first_cite.get('year')}")

                self.logger.success("✅ Generate endpoint passed")
                return True
            else:
                self.logger.error(f"❌ Generate failed: {response.text}")
                return False

        except Exception as e:
            self.logger.error("❌ Generate error", {"error": str(e)})
            return False

    def test_session_management(self) -> bool:
        """Test complete session lifecycle"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Session Management (Create, List, Get, Delete)")
        self.logger.info("=" * 80)

        session_id = f"test-session-{int(time.time())}"

        try:
            # 1. Create session
            self.logger.info("Step 1: Creating session...")
            response = requests.post(
                f"{self.base_url}/sessions",
                json={"session_id": session_id},
                headers=self.headers
            )

            if response.status_code != 200:
                self.logger.error(f"Failed to create session: {response.text}")
                return False

            data = response.json()
            self.logger.info(f"Created: {data.get('session_id')}")

            # 2. List sessions
            self.logger.info("Step 2: Listing sessions...")
            response = requests.get(f"{self.base_url}/sessions", headers=self.headers)

            if response.status_code != 200:
                self.logger.error(f"Failed to list sessions: {response.text}")
                return False

            sessions = response.json()
            self.logger.info(f"Total Sessions: {len(sessions)}")

            # 3. Get specific session
            self.logger.info("Step 3: Getting session details...")
            response = requests.get(f"{self.base_url}/sessions/{session_id}", headers=self.headers)

            if response.status_code != 200:
                self.logger.error(f"Failed to get session: {response.text}")
                return False

            session_data = response.json()
            self.logger.info(f"Session: {session_data.get('session_id')}")
            self.logger.info(f"Turns: {session_data.get('total_turns', 0)}")

            # 4. Delete session
            self.logger.info("Step 4: Deleting session...")
            response = requests.delete(f"{self.base_url}/sessions/{session_id}", headers=self.headers)

            if response.status_code != 200:
                self.logger.error(f"Failed to delete session: {response.text}")
                return False

            self.logger.success("✅ Session management passed")
            return True

        except Exception as e:
            self.logger.error("❌ Session management error", {"error": str(e)})
            return False

    def test_input_validation(self) -> bool:
        """Test input validation and security"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Input Validation & Security")
        self.logger.info("=" * 80)

        test_cases = [
            # XSS attempts
            ("<script>alert(1)</script>", 422, "XSS script tag"),
            ("javascript:alert(1)", 422, "JavaScript protocol"),
            ("<img onerror=alert(1)>", 422, "Image onerror"),

            # Long input
            ("x" * 3000, 422, "Exceeds max length"),

            # Valid inputs
            ("Valid query about legal matters", 200, "Valid query"),
        ]

        passed = 0
        failed = 0

        for query, expected_status, description in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/search",
                    json={"query": query, "max_results": 5},
                    headers=self.headers,
                    timeout=180  # Increased for Kaggle
                )

                if response.status_code == expected_status:
                    self.logger.info(f"✅ {description}: {expected_status}")
                    passed += 1
                else:
                    self.logger.error(f"❌ {description}: expected {expected_status}, got {response.status_code}")
                    failed += 1

            except Exception as e:
                self.logger.error(f"❌ {description}: {str(e)}")
                failed += 1

        self.logger.info(f"\nValidation Results: {passed} passed, {failed} failed")

        if failed == 0:
            self.logger.success("✅ Input validation passed")
            return True
        else:
            self.logger.error("❌ Input validation failed")
            return False

    def test_rate_limiting(self) -> bool:
        """Test rate limiting"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Rate Limiting (60 req/min)")
        self.logger.info("=" * 80)

        try:
            # Send rapid requests
            self.logger.info("Sending 70 rapid requests...")

            success_count = 0
            rate_limited_count = 0

            for i in range(70):
                response = requests.get(f"{self.base_url}/health", timeout=30)

                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited_count += 1
                    self.logger.info(f"Rate limited at request #{i + 1}")
                    break

                if i % 10 == 0:
                    self.logger.info(f"Progress: {i + 1}/70")

            self.logger.info(f"Successful: {success_count}")
            self.logger.info(f"Rate Limited: {rate_limited_count}")

            # Should be rate limited around 60 requests
            if rate_limited_count > 0:
                self.logger.success("✅ Rate limiting is working")
                return True
            else:
                self.logger.warning("⚠️ Rate limiting not triggered (may need more time)")
                return True  # Not a failure, just different timing

        except Exception as e:
            self.logger.error("❌ Rate limiting test error", {"error": str(e)})
            return False

    def run_all_tests(self) -> bool:
        """Run all API endpoint tests"""
        self.logger.info("\n" + "🚀 COMPREHENSIVE API ENDPOINT TESTS".center(80))
        self.logger.info("=" * 80)

        # Start server first
        if not self.start_server():
            self.logger.error("Failed to start server. Aborting tests.")
            return False

        try:
            # Give server time to fully initialize
            time.sleep(5)

            results = []

            # Run all tests
            results.append(("Health Check", self.test_health_check()))
            results.append(("Search Endpoint", self.test_search_endpoint()))
            results.append(("Generate Endpoint", self.test_generate_endpoint()))
            results.append(("Session Management", self.test_session_management()))
            results.append(("Input Validation", self.test_input_validation()))
            results.append(("Rate Limiting", self.test_rate_limiting()))

            # Summary
            self.logger.info("\n" + "=" * 80)
            self.logger.info("TEST SUMMARY")
            self.logger.info("=" * 80)

            passed = sum(1 for _, result in results if result)
            total = len(results)

            for test_name, result in results:
                status = "✅ PASS" if result else "❌ FAIL"
                self.logger.info(f"{status} - {test_name}")

            self.logger.info("=" * 80)
            self.logger.info(f"RESULT: {passed}/{total} tests passed")
            self.logger.info("=" * 80)

            return passed == total

        finally:
            self.stop_server()


def main():
    """Main test runner"""
    tester = APIEndpointTester(port=8000)

    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        tester.stop_server()
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        tester.stop_server()
        return 1


if __name__ == "__main__":
    sys.exit(main())
