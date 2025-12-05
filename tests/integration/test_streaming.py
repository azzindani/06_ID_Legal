#!/usr/bin/env python3
"""
Streaming Test - Real-time LLM Output Testing using TextIteratorStreamer

Tests the streaming capabilities of the RAG system using TextIteratorStreamer:
1. Direct LLM streaming (llm_engine.generate_stream with TextIteratorStreamer)
2. Pipeline streaming (RAGPipeline.query with stream=True)
3. API SSE (Server-Sent Events) streaming endpoint
4. Session-based streaming with conversation context

IMPORTANT: This system uses HuggingFace's TextIteratorStreamer for REAL
token-by-token streaming, exactly like ChatGPT. Tokens are yielded as the
model generates them in a background thread.

The streaming implementation is in:
- core/generation/llm_engine.py:303-460 (TextIteratorStreamer)
- core/generation/generation_engine.py:217-280 (_generate_streaming_answer)

Run with:
    python tests/integration/test_streaming.py
    python tests/integration/test_streaming.py --api  # Test API endpoint
    python tests/integration/test_streaming.py --query "Your custom question"
    python tests/integration/test_streaming.py --llm   # Test LLM directly (skip RAG)
"""

import sys
import os
import time
import argparse
import json
from typing import Dict, Any, Optional, Generator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging


class StreamingTester:
    """
    Tests streaming capabilities with visual real-time output
    """

    def __init__(self):
        initialize_logging(level="INFO")
        self.logger = get_logger("StreamingTest")
        self.pipeline = None
        self.api_url = "http://localhost:8000"

    def initialize_pipeline(self) -> bool:
        """Initialize the RAG pipeline for streaming tests"""
        self.logger.info("=" * 80)
        self.logger.info("STREAMING TEST INITIALIZATION")
        self.logger.info("=" * 80)

        try:
            from pipeline import RAGPipeline

            self.logger.info("Creating RAG Pipeline...")
            self.pipeline = RAGPipeline()

            self.logger.info("Initializing all components (this may take a while)...")
            start_time = time.time()

            if not self.pipeline.initialize():
                self.logger.error("Pipeline initialization failed")
                return False

            elapsed = time.time() - start_time
            self.logger.success(f"Pipeline initialized in {elapsed:.1f}s")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_llm_direct_streaming(self, prompt: str = None) -> bool:
        """
        Test 0: Direct LLM TextIteratorStreamer Test

        Tests the LLM engine directly using TextIteratorStreamer.
        This bypasses RAG and tests pure LLM streaming.

        The LLM uses:
        - TextIteratorStreamer from transformers
        - Background thread for generation
        - Real token-by-token yielding
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 0: Direct LLM TextIteratorStreamer Streaming")
        self.logger.info("=" * 80)

        prompt = prompt or "Jelaskan secara singkat apa itu hukum perdata di Indonesia."
        self.logger.info(f"Prompt: {prompt}")
        self.logger.info("Using: TextIteratorStreamer + Background Thread")
        self.logger.info("\n" + "-" * 40)
        self.logger.info("RAW LLM STREAMING (token-by-token):")
        self.logger.info("-" * 40 + "\n")

        try:
            # Access LLM engine directly
            llm_engine = self.pipeline.generation_engine.llm_engine

            start_time = time.time()
            first_token_time = None
            total_tokens = 0
            full_response = ""

            print(">>> ", end="", flush=True)

            # Use generate_stream which uses TextIteratorStreamer
            for chunk in llm_engine.generate_stream(prompt, max_new_tokens=256):
                if chunk.get('success'):
                    if not chunk.get('done'):
                        token = chunk.get('token', '')
                        if token:
                            if first_token_time is None:
                                first_token_time = time.time() - start_time

                            print(token, end="", flush=True)
                            full_response += token
                            total_tokens = chunk.get('tokens_generated', total_tokens + 1)
                    else:
                        # Final chunk with stats
                        total_tokens = chunk.get('tokens_generated', total_tokens)

            print("\n")

            elapsed = time.time() - start_time

            self.logger.info("-" * 40)
            self.logger.info("TextIteratorStreamer STATISTICS:")
            self.logger.info(f"  Time to First Token: {first_token_time:.3f}s" if first_token_time else "  Time to First Token: N/A")
            self.logger.info(f"  Total Time: {elapsed:.2f}s")
            self.logger.info(f"  Tokens Generated: {total_tokens}")
            self.logger.info(f"  Response Length: {len(full_response)} characters")

            if total_tokens > 0 and elapsed > 0:
                tokens_per_sec = total_tokens / elapsed
                self.logger.info(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

            self.logger.success("Direct LLM streaming test passed")
            return True

        except Exception as e:
            self.logger.error(f"Direct LLM streaming test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_direct_streaming(self, query: str = None) -> bool:
        """
        Test 1: Pipeline Streaming (RAG + TextIteratorStreamer)

        Shows real-time token generation through the full RAG pipeline.
        Uses TextIteratorStreamer under the hood via:
        - RAGPipeline.query(stream=True)
        - GenerationEngine._generate_streaming_answer()
        - LLMEngine.generate_stream() with TextIteratorStreamer
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 1: Pipeline Streaming (RAG + TextIteratorStreamer)")
        self.logger.info("=" * 80)

        query = query or "Apa sanksi pelanggaran UU ITE?"
        self.logger.info(f"Query: {query}")
        self.logger.info("Pipeline: RAGPipeline -> GenerationEngine -> LLMEngine.generate_stream()")
        self.logger.info("\n" + "-" * 40)
        self.logger.info("STREAMING RESPONSE (real-time):")
        self.logger.info("-" * 40 + "\n")

        try:
            start_time = time.time()
            first_token_time = None
            total_tokens = 0
            full_response = ""

            # Use streaming mode
            print("\n>>> ", end="", flush=True)

            for chunk in self.pipeline.query(query, stream=True):
                if isinstance(chunk, dict):
                    # Check for streaming token
                    if chunk.get('type') == 'token':
                        token = chunk.get('token', '')
                        if token:
                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                            print(token, end="", flush=True)
                            full_response += token
                            total_tokens = chunk.get('tokens_generated', total_tokens + 1)
                    # Final result with metadata
                    elif 'answer' in chunk:
                        full_response = chunk['answer']
                        total_tokens = chunk.get('metadata', {}).get('tokens_generated', 0)
                else:
                    # Text chunk - print in real-time
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    print(chunk, end="", flush=True)
                    full_response += str(chunk)
                    total_tokens += 1

            print("\n")  # End the streaming line

            elapsed = time.time() - start_time

            self.logger.info("-" * 40)
            self.logger.info("STREAMING STATISTICS:")
            self.logger.info(f"  Time to First Token: {first_token_time:.3f}s" if first_token_time else "  Time to First Token: N/A")
            self.logger.info(f"  Total Time: {elapsed:.2f}s")
            self.logger.info(f"  Response Length: {len(full_response)} characters")
            self.logger.info(f"  Tokens Generated: {total_tokens}")

            if total_tokens > 0 and elapsed > 0:
                tokens_per_sec = total_tokens / elapsed
                self.logger.info(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

            self.logger.success("Pipeline streaming test passed")
            return True

        except Exception as e:
            self.logger.error(f"Streaming test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_streaming_with_sources(self, query: str = None) -> bool:
        """
        Test 2: Streaming with Interleaved Sources

        Shows sources BEFORE the answer, then streams the response.
        Useful for UIs that want to show citations first.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 2: Streaming with Sources First")
        self.logger.info("=" * 80)

        query = query or "Jelaskan prosedur pelaporan pelanggaran data pribadi"
        self.logger.info(f"Query: {query}")

        try:
            from pipeline.streaming_pipeline import StreamingPipeline

            # Create streaming pipeline
            streaming_pipe = StreamingPipeline()
            streaming_pipe.embedding_model = self.pipeline.embedding_model
            streaming_pipe.reranker_model = self.pipeline.reranker_model
            streaming_pipe.data_loader = self.pipeline.data_loader
            streaming_pipe.orchestrator = self.pipeline.orchestrator
            streaming_pipe.generation_engine = self.pipeline.generation_engine
            streaming_pipe._initialized = True

            start_time = time.time()
            sources_shown = False
            full_response = ""

            self.logger.info("\n" + "-" * 40)

            for chunk in streaming_pipe.stream_with_sources(query):
                chunk_type = chunk.get('type', 'unknown')

                if chunk_type == 'phase':
                    phase = chunk.get('content', '')
                    if phase == 'retrieval':
                        self.logger.info("Searching documents...")
                    elif phase == 'generation':
                        self.logger.info("\nGenerating response:")
                        print("\n>>> ", end="", flush=True)

                elif chunk_type == 'sources':
                    sources = chunk.get('content', [])
                    if sources:
                        self.logger.info(f"\nFound {len(sources)} relevant sources:")
                        for i, source in enumerate(sources[:3], 1):
                            reg = source.get('record', source)
                            reg_type = reg.get('regulation_type', 'N/A')
                            reg_num = reg.get('regulation_number', 'N/A')
                            year = reg.get('year', 'N/A')
                            about = reg.get('about', '')[:50]
                            self.logger.info(f"  {i}. {reg_type} No. {reg_num}/{year}")
                            self.logger.info(f"     {about}...")
                        sources_shown = True

                elif chunk_type == 'token':
                    content = chunk.get('content', '')
                    print(content, end="", flush=True)
                    full_response += content

                elif chunk_type == 'complete':
                    print("\n")
                    metadata = chunk.get('metadata', {})
                    self.logger.info("-" * 40)
                    self.logger.info("STREAMING COMPLETE")

            elapsed = time.time() - start_time
            self.logger.info(f"  Total Time: {elapsed:.2f}s")
            self.logger.info(f"  Sources Displayed: {'Yes' if sources_shown else 'No'}")
            self.logger.info(f"  Response Length: {len(full_response)} characters")

            self.logger.success("Streaming with sources test passed")
            return True

        except Exception as e:
            self.logger.error(f"Streaming with sources test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_session_streaming(self, query: str = None) -> bool:
        """
        Test 3: Session-Based Streaming

        Tests streaming with conversation context -
        follow-up questions use previous context.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 3: Session-Based Streaming (Multi-turn)")
        self.logger.info("=" * 80)

        try:
            # Turn 1: Initial question
            query1 = "Apa itu perlindungan konsumen?"
            self.logger.info(f"\nTurn 1 - Query: {query1}")
            print("\n>>> ", end="", flush=True)

            result1 = self.pipeline.query(query1, stream=False)
            answer1 = result1.get('answer', '')
            print(answer1[:300] + "..." if len(answer1) > 300 else answer1)
            print()

            # Build conversation history
            conversation_history = [
                {'role': 'user', 'content': query1},
                {'role': 'assistant', 'content': answer1}
            ]

            # Turn 2: Follow-up with streaming (uses context)
            query2 = query or "Apa sanksinya?"
            self.logger.info(f"\nTurn 2 - Query: {query2}")
            self.logger.info("(Streaming with conversation context)")
            print("\n>>> ", end="", flush=True)

            full_response = ""
            for chunk in self.pipeline.query(query2, conversation_history=conversation_history, stream=True):
                if isinstance(chunk, dict):
                    if 'answer' in chunk:
                        full_response = chunk['answer']
                else:
                    print(chunk, end="", flush=True)
                    full_response += str(chunk)

            print("\n")

            self.logger.info(f"Response length: {len(full_response)} characters")
            self.logger.success("Session-based streaming test passed")
            return True

        except Exception as e:
            self.logger.error(f"Session streaming test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_api_streaming(self, query: str = None) -> bool:
        """
        Test 4: API SSE Streaming Endpoint

        Tests the /generate/stream endpoint with Server-Sent Events.
        Requires the API server to be running.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 4: API SSE Streaming Endpoint")
        self.logger.info("=" * 80)

        try:
            import requests
        except ImportError:
            self.logger.warning("requests library not installed, skipping API test")
            return True

        query = query or "Apa itu hukum perdata?"
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Endpoint: {self.api_url}/api/v1/generate/stream")

        try:
            # Test if server is running
            try:
                health_response = requests.get(f"{self.api_url}/api/v1/health", timeout=5)
                if health_response.status_code != 200:
                    self.logger.warning("API server not healthy, skipping test")
                    return True
            except requests.exceptions.ConnectionError:
                self.logger.warning("API server not running. Start with: uvicorn api.server:app --port 8000")
                self.logger.info("Skipping API streaming test")
                return True

            # Make streaming request
            self.logger.info("\nStreaming from API...")
            print("\n>>> ", end="", flush=True)

            start_time = time.time()
            full_response = ""
            chunk_count = 0

            with requests.post(
                f"{self.api_url}/api/v1/generate/stream",
                json={"query": query, "stream": True},
                stream=True,
                timeout=120
            ) as response:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])
                                if data.get('type') == 'chunk':
                                    content = data.get('content', '')
                                    print(content, end="", flush=True)
                                    full_response += content
                                    chunk_count += 1
                                elif data.get('type') == 'done':
                                    full_response = data.get('answer', full_response)
                                elif data.get('type') == 'error':
                                    self.logger.error(f"API error: {data.get('message')}")
                            except json.JSONDecodeError:
                                pass

            print("\n")
            elapsed = time.time() - start_time

            self.logger.info("-" * 40)
            self.logger.info("API STREAMING STATISTICS:")
            self.logger.info(f"  Total Time: {elapsed:.2f}s")
            self.logger.info(f"  Chunks Received: {chunk_count}")
            self.logger.info(f"  Response Length: {len(full_response)} characters")

            self.logger.success("API streaming test passed")
            return True

        except Exception as e:
            self.logger.error(f"API streaming test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_streaming_comparison(self, query: str = None) -> bool:
        """
        Test 5: Streaming vs Non-Streaming Comparison

        Compares timing and output between streaming and non-streaming modes.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST 5: Streaming vs Non-Streaming Comparison")
        self.logger.info("=" * 80)

        query = query or "Bagaimana prosedur perizinan usaha?"
        self.logger.info(f"Query: {query}")

        try:
            # Non-streaming test
            self.logger.info("\n1. Non-Streaming Mode:")
            start_ns = time.time()
            result_ns = self.pipeline.query(query, stream=False)
            time_ns = time.time() - start_ns
            answer_ns = result_ns.get('answer', '')
            self.logger.info(f"   Time: {time_ns:.2f}s")
            self.logger.info(f"   Length: {len(answer_ns)} characters")
            self.logger.info(f"   First 100 chars: {answer_ns[:100]}...")

            # Streaming test
            self.logger.info("\n2. Streaming Mode:")
            start_s = time.time()
            first_chunk_time = None
            full_answer = ""
            chunk_count = 0

            for chunk in self.pipeline.query(query, stream=True):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_s

                if isinstance(chunk, dict):
                    if 'answer' in chunk:
                        full_answer = chunk['answer']
                else:
                    full_answer += str(chunk)
                    chunk_count += 1

            time_s = time.time() - start_s

            self.logger.info(f"   Time to first chunk: {first_chunk_time:.2f}s")
            self.logger.info(f"   Total time: {time_s:.2f}s")
            self.logger.info(f"   Chunks: {chunk_count}")
            self.logger.info(f"   Length: {len(full_answer)} characters")

            # Comparison
            self.logger.info("\n3. COMPARISON:")
            self.logger.info(f"   Non-streaming wait time: {time_ns:.2f}s")
            self.logger.info(f"   Streaming first response: {first_chunk_time:.2f}s")

            if first_chunk_time and time_ns > first_chunk_time:
                improvement = ((time_ns - first_chunk_time) / time_ns) * 100
                self.logger.info(f"   Time to first response: {improvement:.0f}% faster with streaming")

            self.logger.success("Comparison test passed")
            return True

        except Exception as e:
            self.logger.error(f"Comparison test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def shutdown(self):
        """Clean shutdown"""
        if self.pipeline:
            self.logger.info("Shutting down pipeline...")
            self.pipeline.shutdown()
            self.logger.success("Shutdown complete")

    def run_all_tests(self, query: str = None, include_llm_direct: bool = True) -> bool:
        """Run all streaming tests"""
        self.logger.info("\n" + "STREAMING TEST SUITE (TextIteratorStreamer)".center(80))
        self.logger.info("=" * 80)
        self.logger.info("Testing real-time streaming using TextIteratorStreamer")
        self.logger.info("=" * 80)

        # Initialize
        if not self.initialize_pipeline():
            self.logger.error("Cannot proceed without pipeline")
            return False

        results = []

        try:
            # Run tests
            if include_llm_direct:
                results.append(("Direct LLM (TextIteratorStreamer)", self.test_llm_direct_streaming()))
            results.append(("Pipeline Streaming", self.test_direct_streaming(query)))
            results.append(("Streaming with Sources", self.test_streaming_with_sources(query)))
            results.append(("Session Streaming", self.test_session_streaming(query)))
            results.append(("Streaming Comparison", self.test_streaming_comparison(query)))

        finally:
            self.shutdown()

        # Summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STREAMING TEST SUMMARY")
        self.logger.info("=" * 80)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            self.logger.info(f"  [{status}] {test_name}")

        self.logger.info("=" * 80)
        self.logger.info(f"RESULT: {passed}/{total} tests passed")
        self.logger.info("=" * 80)

        if passed == total:
            self.logger.success("\nAll streaming tests passed!")

        return passed == total

    def run_llm_only_test(self, prompt: str = None) -> bool:
        """Run only the direct LLM TextIteratorStreamer test"""
        self.logger.info("\n" + "DIRECT LLM STREAMING TEST".center(80))
        self.logger.info("=" * 80)
        self.logger.info("Testing TextIteratorStreamer directly (bypassing RAG)")
        self.logger.info("=" * 80)

        # Initialize
        if not self.initialize_pipeline():
            self.logger.error("Cannot proceed without pipeline")
            return False

        try:
            return self.test_llm_direct_streaming(prompt)
        finally:
            self.shutdown()

    def run_api_test(self, query: str = None) -> bool:
        """Run only API streaming test (no pipeline init needed)"""
        self.logger.info("\n" + "API STREAMING TEST".center(80))
        self.logger.info("=" * 80)

        return self.test_api_streaming(query)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test streaming capabilities using TextIteratorStreamer"
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        default=None,
        help='Custom query to test (default: uses built-in queries)'
    )
    parser.add_argument(
        '--api',
        action='store_true',
        help='Only test API streaming endpoint (requires running server)'
    )
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Only test direct LLM TextIteratorStreamer (bypasses RAG pipeline)'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Only run streaming vs non-streaming comparison'
    )

    args = parser.parse_args()

    tester = StreamingTester()

    try:
        if args.api:
            success = tester.run_api_test(args.query)
        elif args.llm:
            success = tester.run_llm_only_test(args.query)
        elif args.comparison:
            if tester.initialize_pipeline():
                success = tester.test_streaming_comparison(args.query)
                tester.shutdown()
            else:
                success = False
        else:
            success = tester.run_all_tests(args.query)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        tester.shutdown()
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        tester.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())
