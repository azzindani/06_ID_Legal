"""
Conversation Manager Tests

Unit tests for ConversationManager class (in-memory mode).
Persistent storage is tested in test_session_storage.py.

Run with: pytest conversation/tests/test_manager.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conversation.manager import ConversationManager

# Use in-memory mode for these tests (persistent mode tested in test_session_storage.py)
IN_MEMORY_CONFIG = {'persist': False}


class TestSessionManagement:
    """Test session creation and management"""

    def test_start_session(self):
        """Test starting a new session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in manager.sessions

    def test_start_session_custom_id(self):
        """Test starting session with custom ID"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        custom_id = "my-custom-session"
        session_id = manager.start_session(custom_id)

        assert session_id == custom_id
        assert custom_id in manager.sessions

    def test_start_session_duplicate(self):
        """Test starting session with duplicate ID returns existing"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session("test-session")
        session_id_2 = manager.start_session("test-session")

        assert session_id == session_id_2
        assert len(manager.sessions) == 1

    def test_end_session(self):
        """Test ending a session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        result = manager.end_session(session_id)

        assert result is not None
        assert session_id not in manager.sessions

    def test_end_session_not_found(self):
        """Test ending non-existent session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        result = manager.end_session("non-existent")

        assert result is None

    def test_list_sessions(self):
        """Test listing all sessions"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        manager.start_session("session-1")
        manager.start_session("session-2")

        sessions = manager.list_sessions()

        assert len(sessions) == 2

    def test_clear_all_sessions(self):
        """Test clearing all sessions"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        manager.start_session()
        manager.start_session()

        manager.clear_all_sessions()

        assert len(manager.sessions) == 0


class TestConversationTurns:
    """Test conversation turn management"""

    @pytest.fixture
    def manager_with_session(self):
        """Create manager with active session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()
        return manager, session_id

    def test_add_turn(self, manager_with_session):
        """Test adding a conversation turn"""
        manager, session_id = manager_with_session

        turn_num = manager.add_turn(
            session_id=session_id,
            query="Test question?",
            answer="Test answer."
        )

        assert turn_num == 1
        assert len(manager.sessions[session_id]['turns']) == 1

    def test_add_turn_with_metadata(self, manager_with_session):
        """Test adding turn with metadata"""
        manager, session_id = manager_with_session

        metadata = {
            'total_time': 5.2,
            'tokens_generated': 150,
            'query_type': 'general'
        }

        manager.add_turn(
            session_id=session_id,
            query="Test question?",
            answer="Test answer.",
            metadata=metadata
        )

        turn = manager.sessions[session_id]['turns'][0]
        assert turn['metadata']['total_time'] == 5.2
        assert turn['metadata']['tokens_generated'] == 150

    def test_add_multiple_turns(self, manager_with_session):
        """Test adding multiple turns"""
        manager, session_id = manager_with_session

        for i in range(3):
            turn_num = manager.add_turn(
                session_id=session_id,
                query=f"Question {i+1}",
                answer=f"Answer {i+1}"
            )
            assert turn_num == i + 1

        assert len(manager.sessions[session_id]['turns']) == 3

    def test_add_turn_invalid_session(self):
        """Test adding turn to non-existent session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)

        with pytest.raises(ValueError):
            manager.add_turn(
                session_id="non-existent",
                query="Test",
                answer="Test"
            )

    def test_session_metadata_updates(self, manager_with_session):
        """Test that session metadata updates correctly"""
        manager, session_id = manager_with_session

        manager.add_turn(
            session_id=session_id,
            query="Question 1",
            answer="Answer 1",
            metadata={'total_time': 5.0, 'tokens_generated': 100}
        )

        manager.add_turn(
            session_id=session_id,
            query="Question 2",
            answer="Answer 2",
            metadata={'total_time': 3.0, 'tokens_generated': 50}
        )

        meta = manager.sessions[session_id]['metadata']
        assert meta['total_queries'] == 2
        assert meta['total_tokens'] == 150
        assert meta['total_time'] == 8.0


class TestHistoryRetrieval:
    """Test history retrieval methods"""

    @pytest.fixture
    def manager_with_turns(self):
        """Create manager with multiple turns"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        for i in range(5):
            manager.add_turn(
                session_id=session_id,
                query=f"Question {i+1}",
                answer=f"Answer {i+1}"
            )

        return manager, session_id

    def test_get_history(self, manager_with_turns):
        """Test getting full history"""
        manager, session_id = manager_with_turns

        history = manager.get_history(session_id)

        assert len(history) == 5
        assert history[0]['turn_number'] == 1
        assert history[-1]['turn_number'] == 5

    def test_get_history_limited(self, manager_with_turns):
        """Test getting limited history"""
        manager, session_id = manager_with_turns

        history = manager.get_history(session_id, max_turns=3)

        assert len(history) == 3
        assert history[0]['turn_number'] == 3  # Last 3 turns

    def test_get_history_invalid_session(self):
        """Test getting history for invalid session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        history = manager.get_history("non-existent")

        assert history == []

    def test_get_context_for_query(self, manager_with_turns):
        """Test getting context for RAG pipeline"""
        manager, session_id = manager_with_turns

        context = manager.get_context_for_query(session_id, max_turns=2)

        assert len(context) == 4  # 2 turns * 2 (user + assistant)
        assert context[0]['role'] == 'user'
        assert context[1]['role'] == 'assistant'

    def test_get_last_turn(self, manager_with_turns):
        """Test getting last turn"""
        manager, session_id = manager_with_turns

        last = manager.get_last_turn(session_id)

        assert last['turn_number'] == 5
        assert last['query'] == "Question 5"

    def test_get_last_turn_empty_session(self):
        """Test getting last turn from empty session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        last = manager.get_last_turn(session_id)

        assert last is None


class TestSessionInfo:
    """Test session information methods"""

    def test_get_session(self):
        """Test getting full session data"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        manager.add_turn(
            session_id=session_id,
            query="Test",
            answer="Test"
        )

        session = manager.get_session(session_id)

        assert session is not None
        assert session['id'] == session_id
        assert len(session['turns']) == 1
        assert 'created_at' in session

    def test_get_session_not_found(self):
        """Test getting non-existent session"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session = manager.get_session("non-existent")

        assert session is None

    def test_get_session_summary(self):
        """Test getting session summary"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        manager.add_turn(
            session_id=session_id,
            query="Test",
            answer="Test",
            metadata={'total_time': 5.0, 'tokens_generated': 100}
        )

        summary = manager.get_session_summary(session_id)

        assert summary['session_id'] == session_id
        assert summary['total_turns'] == 1
        assert summary['total_queries'] == 1
        assert summary['total_tokens'] == 100
        assert summary['total_time'] == 5.0


class TestHistorySearch:
    """Test history search functionality"""

    def test_search_history(self):
        """Test searching conversation history"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        manager.add_turn(session_id, "What is labor law?", "Labor law is...")
        manager.add_turn(session_id, "What are sanctions?", "Sanctions include...")
        manager.add_turn(session_id, "Tell me about labor rights", "Labor rights are...")

        results = manager.search_history(session_id, "labor")

        assert len(results) == 2

    def test_search_history_no_results(self):
        """Test search with no results"""
        manager = ConversationManager(IN_MEMORY_CONFIG)
        session_id = manager.start_session()

        manager.add_turn(session_id, "Question", "Answer")

        results = manager.search_history(session_id, "nonexistent")

        assert len(results) == 0


class TestConfiguration:
    """Test configuration options"""

    def test_custom_max_history(self):
        """Test custom max history turns"""
        manager = ConversationManager({'persist': False, 'max_history_turns': 3})
        session_id = manager.start_session()

        for i in range(5):
            manager.add_turn(session_id, f"Q{i}", f"A{i}")

        # Should only keep last 3
        assert len(manager.sessions[session_id]['turns']) == 3
        assert manager.sessions[session_id]['turns'][0]['query'] == "Q2"

    def test_custom_max_context(self):
        """Test custom max context turns"""
        manager = ConversationManager({'persist': False, 'max_context_turns': 2})
        session_id = manager.start_session()

        for i in range(5):
            manager.add_turn(session_id, f"Q{i}", f"A{i}")

        context = manager.get_context_for_query(session_id)

        assert len(context) == 4  # 2 turns * 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
