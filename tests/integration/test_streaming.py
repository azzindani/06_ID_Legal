"""
Streaming LLM Output Test
Tests real-time streaming of LLM responses

This test demonstrates:
1. Streaming API endpoint with Server-Sent Events (SSE)
2. Direct pipeline streaming
3. Real-time output display
4. Chunk-by-chunk response

Run with:
    python tests/integration/test_streaming.py

You'll see the LLM output streaming in REAL-TIME!
"""

import sys
import os
import time
import json
import requests
import subprocess
import signal
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging
from pipeline import RAGPipeline


class StreamingTester:
    """Tests streaming LLM output with real-time display"""

    def __init__(self):
        initialize_logging()
        self.logger = get_logger("StreamingTest")
        self.pipeline: Optional[RAGPipeline] = None
        self.server_process: Optional[subprocess.Popen] = None

    def test_direct_pipeline_streaming(self) -> bool:
        """Test 1: Direct pipeline streaming (shows chunks in real-time)"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 1: Direct Pipeline Streaming")
        self.logger.info("=" * 80)

        try:
            # Initialize pipeline
            self.logger.info("Initializing RAG pipeline...")
            self.pipeline = RAGPipeline()

            if not self.pipeline.initialize():
                self.logger.error("❌ Pipeline initialization failed")
                return False

            self.logger.success("✅ Pipeline initialized")

            # Test query
            query = "Jelaskan singkat tentang UU Ketenagakerjaan"
            self.logger.info(f"\n📝 Query: {query}\n")
            self.logger.info("🔄 Streaming response (watch it appear in real-time):\n")

            print("─" * 80)
            print("STREAMING OUTPUT:")
            print("─" * 80)

            full_answer = ""
            chunk_count = 0
            start_time = time.time()

            # Stream the response
            # Pipeline yields dicts with 'type' field: 'token', 'complete', or 'error'
            for chunk in self.pipeline.query(query, stream=True):
                chunk_type = chunk.get('type', '')

                if chunk_type == 'token':
                    # Token chunk - print immediately (real-time streaming)
                    token = chunk.get('token', '')
                    print(token, end='', flush=True)
                    full_answer += token
                    chunk_count += 1

                elif chunk_type == 'complete':
                    # Final result with metadata
                    full_answer = chunk.get('answer', full_answer)
                    metadata = chunk.get('metadata', {})

                    print("\n" + "─" * 80)
                    self.logger.info(f"\n✅ Streaming complete!")
                    self.logger.info(f"Total chunks: {chunk_count}")
                    self.logger.info(f"Total time: {time.time() - start_time:.2f}s")
                    self.logger.info(f"Final length: {len(full_answer)} chars")

                    if metadata.get('results_count'):
                        self.logger.info(f"Sources used: {metadata['results_count']}")

                elif chunk_type == 'error':
                    # Error during streaming
                    error_msg = chunk.get('error', 'Unknown error')
                    self.logger.error(f"Streaming error: {error_msg}")
                    return False

            if full_answer and chunk_count > 0:
                self.logger.success("\n✅ Direct pipeline streaming passed")
                return True
            else:
                self.logger.error("❌ No streaming output received")
                return False

        except Exception as e:
            self.logger.error("❌ Direct pipeline streaming failed", {"error": str(e)})
            import traceback
            self.logger.error(traceback.format_exc())
            return False

        finally:
            if self.pipeline:
                self.logger.info("Shutting down pipeline...")
                self.pipeline.shutdown()

    def start_api_server(self, port: int = 8001, timeout: int = 30) -> bool:
        """Start API server for streaming tests"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Starting API Server")
        self.logger.info("=" * 80)

        try:
            self.logger.info(f"Starting server on port {port}...")
            self.server_process = subprocess.Popen(
                ["python", "-m", "uvicorn", "api.server:app",
                 "--host", "0.0.0.0", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            # Wait for server to be ready
            self.logger.info(f"Waiting for server (timeout: {timeout}s)...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"http://localhost:{port}/api/v1/health", timeout=1)
                    if response.status_code == 200:
                        self.logger.success(f"✅ Server ready in {time.time() - start_time:.1f}s")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)

            self.logger.error("❌ Server failed to start")
            return False

        except Exception as e:
            self.logger.error("Failed to start server", {"error": str(e)})
            return False

    def stop_api_server(self):
        """Stop API server"""
        if self.server_process:
            self.logger.info("Stopping API server...")
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                self.logger.success("✅ Server stopped")
            except Exception as e:
                self.logger.error("Error stopping server", {"error": str(e)})

    def test_api_streaming(self, port: int = 8001) -> bool:
        """Test 2: API streaming endpoint (Server-Sent Events)"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 2: API Streaming Endpoint (SSE)")
        self.logger.info("=" * 80)

        try:
            query = "Apa sanksi dalam UU Perlindungan Data Pribadi?"
            self.logger.info(f"\n📝 Query: {query}\n")
            self.logger.info("🔄 Streaming from API endpoint:\n")

            print("─" * 80)
            print("STREAMING OUTPUT (Server-Sent Events):")
            print("─" * 80)

            url = f"http://localhost:{port}/api/v1/generate/stream"
            payload = {
                "query": query,
                "stream": True
            }

            full_answer = ""
            chunk_count = 0
            start_time = time.time()

            # Make streaming request
            response = requests.post(url, json=payload, stream=True, timeout=60)

            if response.status_code != 200:
                self.logger.error(f"❌ API returned status {response.status_code}")
                return False

            # Process SSE stream
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])  # Remove 'data: ' prefix

                        if data['type'] == 'chunk':
                            # Print chunk in real-time
                            print(data['content'], end='', flush=True)
                            full_answer += data['content']
                            chunk_count += 1

                        elif data['type'] == 'done':
                            # Final result
                            full_answer = data.get('answer', full_answer)
                            print("\n" + "─" * 80)
                            self.logger.info(f"\n✅ Streaming complete!")
                            self.logger.info(f"Total chunks: {chunk_count}")
                            self.logger.info(f"Total time: {time.time() - start_time:.2f}s")
                            self.logger.info(f"Final length: {len(full_answer)} chars")

                        elif data['type'] == 'error':
                            self.logger.error(f"❌ Stream error: {data['message']}")
                            return False

            if full_answer and chunk_count > 0:
                self.logger.success("\n✅ API streaming passed")
                return True
            else:
                self.logger.error("❌ No streaming output received")
                return False

        except Exception as e:
            self.logger.error("❌ API streaming failed", {"error": str(e)})
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def test_streaming_with_session(self, port: int = 8001) -> bool:
        """Test 3: Streaming with session context"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 3: Streaming with Session Context")
        self.logger.info("=" * 80)

        try:
            # Create session
            session_id = f"stream-test-{int(time.time())}"
            self.logger.info(f"Creating session: {session_id}")

            session_response = requests.post(
                f"http://localhost:{port}/api/v1/sessions",
                json={"session_id": session_id}
            )

            if session_response.status_code != 200:
                self.logger.error("❌ Failed to create session")
                return False

            self.logger.success("✅ Session created")

            # First query
            self.logger.info("\n🗣️  Turn 1: Initial question")
            query1 = "Apa itu perlindungan konsumen?"

            response1 = requests.post(
                f"http://localhost:{port}/api/v1/generate/stream",
                json={"query": query1, "session_id": session_id},
                stream=True
            )

            print("\n" + "─" * 40)
            print(f"User: {query1}")
            print("Assistant: ", end='', flush=True)

            for line in response1.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if data['type'] == 'chunk':
                            print(data['content'], end='', flush=True)
                        elif data['type'] == 'done':
                            print("\n" + "─" * 40)
                            break

            # Follow-up query (uses context)
            time.sleep(2)
            self.logger.info("\n🗣️  Turn 2: Follow-up question")
            query2 = "Apa sanksinya?"

            response2 = requests.post(
                f"http://localhost:{port}/api/v1/generate/stream",
                json={"query": query2, "session_id": session_id},
                stream=True
            )

            print(f"User: {query2}")
            print("Assistant: ", end='', flush=True)

            for line in response2.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if data['type'] == 'chunk':
                            print(data['content'], end='', flush=True)
                        elif data['type'] == 'done':
                            print("\n" + "─" * 40)
                            break

            # Clean up session
            requests.delete(f"http://localhost:{port}/api/v1/sessions/{session_id}")

            self.logger.success("\n✅ Session-based streaming passed")
            return True

        except Exception as e:
            self.logger.error("❌ Session streaming failed", {"error": str(e)})
            return False

    def run_all_tests(self, test_api: bool = False) -> bool:
        """Run all streaming tests"""
        self.logger.info("\n" + "🚀 STREAMING LLM OUTPUT TESTS".center(80))
        self.logger.info("=" * 80)

        results = []

        # Test 1: Direct pipeline streaming (always run)
        results.append(("Direct Pipeline Streaming", self.test_direct_pipeline_streaming()))

        # Test 2 & 3: API streaming (optional, requires server)
        if test_api:
            port = 8001  # Use different port to avoid conflicts

            if self.start_api_server(port=port):
                try:
                    # Give server time to initialize fully
                    time.sleep(5)

                    results.append(("API Streaming Endpoint", self.test_api_streaming(port=port)))
                    results.append(("Streaming with Session", self.test_streaming_with_session(port=port)))

                finally:
                    self.stop_api_server()
            else:
                self.logger.warning("⚠️  Skipping API tests (server failed to start)")
        else:
            self.logger.info("\nℹ️  Skipping API streaming tests (use --api flag to enable)")

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

        if passed == total:
            self.logger.success("\n🎉 ALL STREAMING TESTS PASSED!")

        return passed == total


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Test streaming LLM output")
    parser.add_argument('--api', action='store_true', help='Also test API streaming endpoints (slower)')
    args = parser.parse_args()

    tester = StreamingTester()

    try:
        success = tester.run_all_tests(test_api=args.api)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        tester.stop_api_server()
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        tester.stop_api_server()
        return 1


if __name__ == "__main__":
    sys.exit(main())
