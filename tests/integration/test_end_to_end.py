"""
End-to-end integration tests

Run with: pytest tests/integration/test_end_to_end.py -v -m integration

Note: These tests require GPU and loaded models.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEnd:
    """End-to-end system tests"""

    @pytest.fixture
    def pipeline(self):
        """Create and initialize pipeline"""
        from pipeline import RAGPipeline

        pipeline = RAGPipeline()
        if not pipeline.initialize():
            pytest.skip("Failed to initialize pipeline")

        yield pipeline
        pipeline.shutdown()

    def test_simple_query(self, pipeline):
        """Test simple question answering"""
        result = pipeline.query("Apa itu UU Ketenagakerjaan?")

        assert result is not None
        assert 'answer' in result
        assert len(result['answer']) > 0
        assert 'metadata' in result

    def test_sanctions_query(self, pipeline):
        """Test sanctions query"""
        result = pipeline.query("Apa sanksi pelanggaran UU Perlindungan Konsumen?")

        assert result is not None
        assert 'answer' in result
        # Should mention sanctions/penalties
        answer_lower = result['answer'].lower()
        assert any(word in answer_lower for word in ['sanksi', 'denda', 'pidana', 'hukuman'])

    def test_follow_up_query(self, pipeline):
        """Test follow-up question with context"""
        # First query
        result1 = pipeline.query("Apa itu UU Ketenagakerjaan?")

        # Follow-up with context
        context = [
            {'role': 'user', 'content': 'Apa itu UU Ketenagakerjaan?'},
            {'role': 'assistant', 'content': result1['answer']}
        ]

        result2 = pipeline.query("Apa sanksinya?", conversation_history=context)

        assert result2 is not None
        assert 'answer' in result2

    def test_query_metadata(self, pipeline):
        """Test that metadata is returned"""
        result = pipeline.query("Apa itu tenaga kerja?")

        metadata = result.get('metadata', {})
        assert 'total_time' in metadata
        assert metadata['total_time'] > 0

    def test_multiple_queries(self, pipeline):
        """Test multiple consecutive queries"""
        queries = [
            "Apa definisi konsumen?",
            "Apa hak konsumen?",
            "Apa sanksi pelanggaran?"
        ]

        for query in queries:
            result = pipeline.query(query)
            assert result is not None
            assert 'answer' in result


@pytest.mark.integration
@pytest.mark.slow
class TestConversationIntegration:
    """Test conversation management integration"""

    def test_session_with_pipeline(self):
        """Test full conversation session"""
        from pipeline import RAGPipeline
        from conversation import ConversationManager

        pipeline = RAGPipeline()
        if not pipeline.initialize():
            pytest.skip("Failed to initialize pipeline")

        try:
            manager = ConversationManager()
            session_id = manager.start_session()

            # First query
            result = pipeline.query("Apa itu UU Ketenagakerjaan?")
            manager.add_turn(
                session_id=session_id,
                query="Apa itu UU Ketenagakerjaan?",
                answer=result['answer'],
                metadata=result.get('metadata')
            )

            # Verify session
            session = manager.get_session(session_id)
            assert session is not None
            assert len(session['turns']) == 1

            # Follow-up
            context = manager.get_context_for_query(session_id)
            result2 = pipeline.query("Apa sanksinya?", conversation_history=context)
            manager.add_turn(
                session_id=session_id,
                query="Apa sanksinya?",
                answer=result2['answer'],
                metadata=result2.get('metadata')
            )

            # Verify updated session
            session = manager.get_session(session_id)
            assert len(session['turns']) == 2

        finally:
            pipeline.shutdown()


@pytest.mark.integration
@pytest.mark.slow
class TestExportIntegration:
    """Test export with real session data"""

    def test_export_real_session(self, temp_dir):
        """Test exporting a real conversation session"""
        from pipeline import RAGPipeline
        from conversation import ConversationManager, MarkdownExporter

        pipeline = RAGPipeline()
        if not pipeline.initialize():
            pytest.skip("Failed to initialize pipeline")

        try:
            manager = ConversationManager()
            session_id = manager.start_session()

            # Create conversation
            result = pipeline.query("Apa itu hukum pidana?")
            manager.add_turn(
                session_id=session_id,
                query="Apa itu hukum pidana?",
                answer=result['answer'],
                metadata=result.get('metadata')
            )

            # Export
            session_data = manager.get_session(session_id)
            exporter = MarkdownExporter()
            path = exporter.export_and_save(session_data, directory=str(temp_dir))

            assert path.exists()
            content = path.read_text()
            assert 'hukum pidana' in content.lower()

        finally:
            pipeline.shutdown()
