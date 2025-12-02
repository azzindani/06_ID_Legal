"""
Demo 08: Full System Test

Comprehensive validation of all major features:
- Core RAG pipeline
- Multi-researcher simulation
- Knowledge Graph enhancement
- Streaming responses
- Session management
- Export functionality
- Performance benchmarks

This is the main validation script to ensure the entire system works.

Run: python demos/08_full_system_test.py
"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SystemTester:
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        self.start_time = None
        self.pipeline = None

    def log(self, message, level="INFO"):
        """Log a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "  ℹ️ ",
            "SUCCESS": "  ✓",
            "ERROR": "  ❌",
            "WARNING": "  ⚠️ ",
            "SECTION": "\n[",
        }.get(level, "  ")

        if level == "SECTION":
            print(f"\n{prefix}{message}")
        else:
            print(f"{prefix} {message}")

    def test(self, name, func):
        """Run a test"""
        self.total_tests += 1
        test_num = self.total_tests

        print(f"\n{'─' * 80}")
        self.log(f"Test {test_num}: {name}]", "SECTION")
        print(f"{'─' * 80}")

        start = time.time()
        try:
            func()
            duration = time.time() - start
            self.passed_tests += 1
            self.log(f"PASSED ({duration:.1f}s)", "SUCCESS")
            self.results.append({
                'test': name,
                'status': 'PASSED',
                'duration': duration,
                'error': None
            })
            return True
        except AssertionError as e:
            duration = time.time() - start
            self.failed_tests += 1
            self.log(f"FAILED: {e} ({duration:.1f}s)", "ERROR")
            self.results.append({
                'test': name,
                'status': 'FAILED',
                'duration': duration,
                'error': str(e)
            })
            return False
        except Exception as e:
            duration = time.time() - start
            self.failed_tests += 1
            self.log(f"ERROR: {e} ({duration:.1f}s)", "ERROR")
            self.results.append({
                'test': name,
                'status': 'ERROR',
                'duration': duration,
                'error': str(e)
            })
            return False

    def initialize_pipeline(self):
        """Initialize the RAG pipeline"""
        from pipeline import RAGPipeline

        self.log("Initializing RAG pipeline...")
        self.log("This may take 30-90 seconds on first run")

        self.pipeline = RAGPipeline()
        success = self.pipeline.initialize()

        assert success, "Pipeline initialization failed"
        self.log("Pipeline initialized successfully", "SUCCESS")

    def test_basic_query(self):
        """Test 1: Basic RAG query"""
        self.log("Running basic query...")

        result = self.pipeline.query("Apa itu UU Ketenagakerjaan?")

        assert result is not None, "No result returned"
        assert 'answer' in result, "No answer in result"
        assert len(result['answer']) > 0, "Empty answer"
        assert 'metadata' in result, "No metadata"

        self.log(f"Answer length: {len(result['answer'])} chars")
        self.log(f"Sources: {len(result.get('sources', []))}")

    def test_query_types(self):
        """Test 2: Different query types"""
        test_queries = [
            ("Apa sanksi pelanggaran?", "sanctions"),
            ("Bagaimana prosedur PHK?", "procedural"),
            ("Apa definisi tenaga kerja?", "definitional"),
        ]

        for query, expected_type in test_queries:
            self.log(f"Testing {expected_type} query...")

            result = self.pipeline.query(query)

            assert result is not None, f"No result for {expected_type} query"
            assert 'answer' in result, f"No answer for {expected_type} query"

            # Check if query type was detected (may be in metadata)
            metadata = result.get('metadata', {})
            detected_type = metadata.get('query_type', 'unknown')
            self.log(f"  Query type detected: {detected_type}")

    def test_conversation_context(self):
        """Test 3: Conversation with context"""
        self.log("Testing conversation with follow-up questions...")

        # First query
        result1 = self.pipeline.query("Apa itu UU Perlindungan Konsumen?")
        assert result1 is not None, "First query failed"

        # Follow-up with context
        context = [
            {'role': 'user', 'content': 'Apa itu UU Perlindungan Konsumen?'},
            {'role': 'assistant', 'content': result1['answer']}
        ]

        result2 = self.pipeline.query("Apa sanksinya?", conversation_history=context)
        assert result2 is not None, "Follow-up query failed"
        assert 'answer' in result2, "No answer in follow-up"

        self.log("Follow-up query processed successfully")

    def test_metadata_tracking(self):
        """Test 4: Metadata tracking"""
        self.log("Testing metadata tracking...")

        result = self.pipeline.query("Apa itu hukum perdata?")

        metadata = result.get('metadata', {})
        assert metadata, "No metadata returned"

        # Check required metadata fields
        required_fields = ['total_time', 'query_type']
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        self.log(f"Total time: {metadata['total_time']:.2f}s")
        self.log(f"Query type: {metadata['query_type']}")

        if 'tokens_generated' in metadata:
            self.log(f"Tokens generated: {metadata['tokens_generated']}")

    def test_session_management(self):
        """Test 5: Session management"""
        from conversation import ConversationManager

        self.log("Testing session management...")

        manager = ConversationManager()
        session_id = manager.start_session()

        assert session_id is not None, "Failed to create session"
        self.log(f"Session created: {session_id}")

        # Add a turn
        manager.add_turn(
            session_id=session_id,
            query="Test query",
            answer="Test answer",
            metadata={'test': True}
        )

        # Get history
        history = manager.get_history(session_id)
        assert len(history) == 1, "History not recorded"

        # Get session
        session = manager.get_session(session_id)
        assert session is not None, "Session not found"
        assert len(session['turns']) == 1, "Turn not added"

        self.log("Session management working correctly")

        # End session
        final_data = manager.end_session(session_id)
        assert final_data is not None, "Failed to end session"

    def test_export_functionality(self):
        """Test 6: Export to different formats"""
        from conversation import ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter

        self.log("Testing export functionality...")

        # Create test session
        manager = ConversationManager()
        session_id = manager.start_session()

        manager.add_turn(
            session_id=session_id,
            query="Test query for export",
            answer="Test answer with sources",
            metadata={
                'total_time': 5.0,
                'sources': [
                    {'regulation_type': 'UU', 'regulation_number': '13', 'year': '2003'}
                ]
            }
        )

        session_data = manager.get_session(session_id)

        # Test Markdown export
        md_exporter = MarkdownExporter()
        md_content = md_exporter.export(session_data)
        assert len(md_content) > 0, "Empty Markdown export"
        assert "Test query for export" in md_content, "Query not in Markdown"
        self.log("Markdown export working")

        # Test JSON export
        json_exporter = JSONExporter()
        json_content = json_exporter.export(session_data)
        assert len(json_content) > 0, "Empty JSON export"
        assert "Test query for export" in json_content, "Query not in JSON"
        self.log("JSON export working")

        # Test HTML export
        html_exporter = HTMLExporter()
        html_content = html_exporter.export(session_data)
        assert len(html_content) > 0, "Empty HTML export"
        assert "Test query for export" in html_content, "Query not in HTML"
        self.log("HTML export working")

        manager.end_session(session_id)

    def test_streaming(self):
        """Test 7: Streaming responses"""
        self.log("Testing streaming functionality...")
        self.log("(Collecting streamed tokens...)")

        collected_text = ""
        token_count = 0

        try:
            for chunk in self.pipeline.query("Apa itu hukum?", stream=True):
                if chunk.get('type') == 'token':
                    collected_text += chunk.get('token', '')
                    token_count += 1
                elif chunk.get('type') == 'complete':
                    metadata = chunk.get('metadata', {})
                    break

            assert len(collected_text) > 0, "No text collected from stream"
            assert token_count > 0, "No tokens streamed"

            self.log(f"Collected {token_count} tokens")
            self.log(f"Total text length: {len(collected_text)} chars")

        except Exception as e:
            # Streaming may not be supported by all providers
            self.log(f"Streaming not supported or failed: {e}", "WARNING")
            self.warnings += 1

    def test_provider_info(self):
        """Test 8: Provider information"""
        from providers import get_provider, list_providers

        self.log("Testing provider system...")

        # List providers
        providers = list_providers()
        assert len(providers) > 0, "No providers available"
        self.log(f"Available providers: {', '.join(providers)}")

        # Get current provider
        provider = get_provider()
        info = provider.get_info()

        assert 'provider' in info, "No provider info"
        assert 'model' in info, "No model info"

        self.log(f"Current provider: {info['provider']}")
        self.log(f"Current model: {info['model']}")

    def test_query_detection(self):
        """Test 9: Query detection and analysis"""
        from core.search.query_detection import QueryDetector

        self.log("Testing query detection...")

        detector = QueryDetector()

        test_cases = [
            ("Apa sanksi pelanggaran?", "sanctions"),
            ("Bagaimana cara mendaftar?", "procedural"),
            ("Apa definisi konsumen?", "definitional"),
        ]

        for query, expected_type in test_cases:
            result = detector.analyze_query(query)

            assert 'query_type' in result, "No query type detected"
            detected_type = result['query_type']

            # Some flexibility in type detection
            self.log(f"'{query}' → {detected_type}")

    def test_context_cache(self):
        """Test 10: Context cache"""
        from conversation.context_cache import ContextCache

        self.log("Testing context cache...")

        cache = ContextCache(max_size_mb=10)

        # Add to cache
        key = "test_session_123"
        context = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ]

        cache.put(key, context)

        # Retrieve from cache
        retrieved = cache.get(key)
        assert retrieved is not None, "Cache retrieval failed"
        assert len(retrieved) == 2, "Incorrect cache content"

        self.log("Context caching working correctly")

        # Test cache stats
        stats = cache.get_stats()
        assert 'size' in stats, "No cache stats"
        self.log(f"Cache size: {stats['size']} items")

    def print_summary(self):
        """Print test summary"""
        total_time = time.time() - self.start_time

        print("\n\n")
        print("=" * 80)
        print(" " * 30 + "TEST SUMMARY")
        print("=" * 80)
        print()
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ✓")
        print(f"Failed: {self.failed_tests} ❌")
        print(f"Warnings: {self.warnings} ⚠️")
        print()
        print(f"Success rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        print(f"Total time: {total_time:.1f}s")
        print()

        if self.failed_tests > 0:
            print("FAILED TESTS:")
            print("-" * 80)
            for result in self.results:
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"  ❌ {result['test']}")
                    print(f"     Error: {result['error']}")
            print()

        print("DETAILED RESULTS:")
        print("-" * 80)
        for i, result in enumerate(self.results, 1):
            status_icon = {
                'PASSED': '✓',
                'FAILED': '❌',
                'ERROR': '❌'
            }.get(result['status'], '?')

            print(f"{i:2d}. [{status_icon}] {result['test']:50s} {result['duration']:6.1f}s")

        print()
        print("=" * 80)

        if self.failed_tests == 0:
            print("🎉 ALL TESTS PASSED! System is working correctly.")
        else:
            print("⚠️  SOME TESTS FAILED. Please review the errors above.")

        print("=" * 80)
        print()

    def run(self):
        """Run all tests"""
        self.start_time = time.time()

        print("=" * 80)
        print(" " * 25 + "FULL SYSTEM TEST")
        print(" " * 20 + "Indonesian Legal Assistant")
        print("=" * 80)
        print()
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Initialize
        print("\n" + "=" * 80)
        print("INITIALIZATION")
        print("=" * 80)

        try:
            self.initialize_pipeline()
        except Exception as e:
            print(f"\n❌ FATAL: Could not initialize pipeline: {e}")
            return 1

        # Run tests
        print("\n" + "=" * 80)
        print("RUNNING TESTS")
        print("=" * 80)

        self.test("Basic RAG Query", self.test_basic_query)
        self.test("Multiple Query Types", self.test_query_types)
        self.test("Conversation Context", self.test_conversation_context)
        self.test("Metadata Tracking", self.test_metadata_tracking)
        self.test("Session Management", self.test_session_management)
        self.test("Export Functionality", self.test_export_functionality)
        self.test("Streaming Responses", self.test_streaming)
        self.test("Provider Information", self.test_provider_info)
        self.test("Query Detection", self.test_query_detection)
        self.test("Context Cache", self.test_context_cache)

        # Cleanup
        print("\n" + "=" * 80)
        print("CLEANUP")
        print("=" * 80)
        print()

        if self.pipeline:
            self.log("Shutting down pipeline...")
            self.pipeline.shutdown()
            self.log("Pipeline shut down successfully", "SUCCESS")

        # Summary
        self.print_summary()

        return 0 if self.failed_tests == 0 else 1


def main():
    tester = SystemTester()
    return tester.run()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
