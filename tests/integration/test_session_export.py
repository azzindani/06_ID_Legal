"""
Comprehensive Session & Export Integration Tests
Tests session management, conversation history, and export functionality

Run with:
    python tests/integration/test_session_export.py

This initializes the full system and shows real output.
"""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging
from conversation import ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter


class SessionExportTester:
    """Comprehensive session and export tester"""

    def __init__(self):
        initialize_logging()
        self.logger = get_logger("SessionExportTest")
        self.manager: ConversationManager = None

    def setup(self) -> bool:
        """Initialize conversation manager"""
        self.logger.info("=" * 80)
        self.logger.info("SETTING UP SESSION MANAGER")
        self.logger.info("=" * 80)

        try:
            self.manager = ConversationManager()
            self.logger.success("ConversationManager initialized")
            return True
        except Exception as e:
            self.logger.error("Failed to initialize", {"error": str(e)})
            return False

    def test_session_creation(self) -> bool:
        """Test session creation"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Session Creation")
        self.logger.info("=" * 80)

        try:
            # Create session with custom ID
            session_id = f"test-session-{int(time.time())}"
            self.logger.info(f"Creating session: {session_id}")

            created_id = self.manager.start_session(session_id)

            if created_id == session_id:
                self.logger.success(f"✅ Session created: {created_id}")
                return True
            else:
                self.logger.error(f"❌ Session ID mismatch: {created_id} != {session_id}")
                return False

        except Exception as e:
            self.logger.error("❌ Session creation failed", {"error": str(e)})
            return False

    def test_conversation_flow(self) -> bool:
        """Test complete conversation flow"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Conversation Flow (Multi-turn)")
        self.logger.info("=" * 80)

        try:
            # Create new session
            session_id = self.manager.start_session()
            self.logger.info(f"Session ID: {session_id}")

            # Simulate conversation
            conversations = [
                ("Apa itu UU Ketenagakerjaan?", "UU Ketenagakerjaan adalah..."),
                ("Apa sanksinya?", "Sanksi dalam UU Ketenagakerjaan meliputi..."),
                ("Bagaimana prosedurnya?", "Prosedur yang berlaku adalah...")
            ]

            for turn, (query, answer) in enumerate(conversations, 1):
                self.logger.info(f"\nTurn {turn}:")
                self.logger.info(f"  Query: {query}")

                # Add turn
                self.manager.add_turn(
                    session_id=session_id,
                    query=query,
                    answer=answer,
                    metadata={
                        'turn_number': turn,
                        'timestamp': time.time()
                    }
                )

                self.logger.info(f"  Answer: {answer[:50]}...")

            # Get history
            history = self.manager.get_history(session_id)
            self.logger.info(f"\nTotal turns in history: {len(history)}")

            if len(history) == len(conversations):
                self.logger.success("✅ Conversation flow passed")
                return True
            else:
                self.logger.error(f"❌ History mismatch: {len(history)} != {len(conversations)}")
                return False

        except Exception as e:
            self.logger.error("❌ Conversation flow failed", {"error": str(e)})
            return False

    def test_session_summary(self) -> bool:
        """Test session summary"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Session Summary")
        self.logger.info("=" * 80)

        try:
            # Create session with data
            session_id = self.manager.start_session()

            for i in range(5):
                self.manager.add_turn(
                    session_id=session_id,
                    query=f"Question {i + 1}",
                    answer=f"Answer {i + 1}",
                    metadata={'tokens_generated': 100}
                )

            # Get summary
            summary = self.manager.get_session_summary(session_id)

            self.logger.info("Session Summary:")
            self.logger.info(f"  Session ID: {summary.get('session_id')}")
            self.logger.info(f"  Total Turns: {summary.get('total_turns')}")
            self.logger.info(f"  Total Queries: {summary.get('total_queries')}")
            self.logger.info(f"  Total Tokens: {summary.get('total_tokens')}")
            self.logger.info(f"  Total Time: {summary.get('total_time', 0):.2f}s")

            if summary.get('total_turns') == 5:
                self.logger.success("✅ Session summary passed")
                return True
            else:
                self.logger.error("❌ Summary data incorrect")
                return False

        except Exception as e:
            self.logger.error("❌ Session summary failed", {"error": str(e)})
            return False

    def test_markdown_export(self) -> bool:
        """Test Markdown export"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Markdown Export")
        self.logger.info("=" * 80)

        try:
            # Create session with data
            session_id = self.manager.start_session()

            self.manager.add_turn(
                session_id=session_id,
                query="Apa itu perlindungan konsumen?",
                answer="Perlindungan konsumen adalah hak konsumen...",
                metadata={'citations': [
                    {'regulation_type': 'UU', 'regulation_number': '8', 'year': '1999'}
                ]}
            )

            # Get session data
            session_data = self.manager.get_session(session_id)

            # Export to Markdown
            exporter = MarkdownExporter()
            md_content = exporter.export(session_data)

            self.logger.info("Markdown Export Preview:")
            lines = md_content.split('\n')
            for line in lines[:20]:  # Show first 20 lines
                self.logger.info(f"  {line}")

            if len(md_content) > 0 and '# Conversation' in md_content:
                self.logger.success("✅ Markdown export passed")
                return True
            else:
                self.logger.error("❌ Markdown export empty or invalid")
                return False

        except Exception as e:
            self.logger.error("❌ Markdown export failed", {"error": str(e)})
            return False

    def test_json_export(self) -> bool:
        """Test JSON export"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: JSON Export")
        self.logger.info("=" * 80)

        try:
            # Create session with data
            session_id = self.manager.start_session()

            self.manager.add_turn(
                session_id=session_id,
                query="Test query for JSON",
                answer="Test answer for JSON",
                metadata={'test_key': 'test_value'}
            )

            # Get session data
            session_data = self.manager.get_session(session_id)

            # Export to JSON
            exporter = JSONExporter()
            json_content = exporter.export(session_data)

            # Parse to verify
            parsed = json.loads(json_content)

            self.logger.info("JSON Export Structure:")
            self.logger.info(f"  Session ID: {parsed.get('session_id')}")
            self.logger.info(f"  Turns: {len(parsed.get('turns', []))}")
            self.logger.info(f"  Created: {parsed.get('created_at')}")

            if parsed.get('session_id') == session_id and len(parsed.get('turns', [])) == 1:
                self.logger.success("✅ JSON export passed")
                return True
            else:
                self.logger.error("❌ JSON export data incorrect")
                return False

        except Exception as e:
            self.logger.error("❌ JSON export failed", {"error": str(e)})
            return False

    def test_html_export(self) -> bool:
        """Test HTML export"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: HTML Export")
        self.logger.info("=" * 80)

        try:
            # Create session with data
            session_id = self.manager.start_session()

            self.manager.add_turn(
                session_id=session_id,
                query="HTML export test query",
                answer="HTML export test answer with **formatting**",
                metadata={}
            )

            # Get session data
            session_data = self.manager.get_session(session_id)

            # Export to HTML
            exporter = HTMLExporter()
            html_content = exporter.export(session_data)

            self.logger.info("HTML Export Preview:")
            lines = html_content.split('\n')
            for line in lines[:15]:  # Show first 15 lines
                self.logger.info(f"  {line[:80]}")

            if len(html_content) > 0 and '<html>' in html_content.lower():
                self.logger.success("✅ HTML export passed")
                return True
            else:
                self.logger.error("❌ HTML export empty or invalid")
                return False

        except Exception as e:
            self.logger.error("❌ HTML export failed", {"error": str(e)})
            return False

    def test_session_cleanup(self) -> bool:
        """Test session deletion and cleanup"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Session Cleanup")
        self.logger.info("=" * 80)

        try:
            # Create session
            session_id = self.manager.start_session()
            self.manager.add_turn(session_id, "Test", "Answer")

            # Verify exists
            session_before = self.manager.get_session(session_id)
            self.logger.info(f"Session exists: {session_before is not None}")

            # Delete session
            result = self.manager.end_session(session_id)
            self.logger.info(f"Deletion result: {result is not None}")

            # Verify deleted
            session_after = self.manager.get_session(session_id)
            self.logger.info(f"Session after deletion: {session_after}")

            if session_before and not session_after:
                self.logger.success("✅ Session cleanup passed")
                return True
            else:
                self.logger.error("❌ Session cleanup failed")
                return False

        except Exception as e:
            self.logger.error("❌ Session cleanup error", {"error": str(e)})
            return False

    def run_all_tests(self) -> bool:
        """Run all session and export tests"""
        self.logger.info("\n" + "🚀 COMPREHENSIVE SESSION & EXPORT TESTS".center(80))
        self.logger.info("=" * 80)

        # Setup
        if not self.setup():
            self.logger.error("Setup failed. Aborting tests.")
            return False

        results = []

        # Run all tests
        results.append(("Session Creation", self.test_session_creation()))
        results.append(("Conversation Flow", self.test_conversation_flow()))
        results.append(("Session Summary", self.test_session_summary()))
        results.append(("Markdown Export", self.test_markdown_export()))
        results.append(("JSON Export", self.test_json_export()))
        results.append(("HTML Export", self.test_html_export()))
        results.append(("Session Cleanup", self.test_session_cleanup()))

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


def main():
    """Main test runner"""
    tester = SessionExportTester()

    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
