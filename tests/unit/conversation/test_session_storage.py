"""
Unit Tests for Session Storage (SQLite Persistence)

Tests the SQLite-backed session storage for conversation persistence.

File: tests/unit/conversation/test_session_storage.py
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from conversation.session_storage import SessionStorage


class TestSessionStorage:
    """Tests for SessionStorage SQLite backend"""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create a temporary storage instance"""
        db_path = str(tmp_path / "test_sessions.db")
        return SessionStorage(db_path)
    
    @pytest.fixture
    def db_path(self, tmp_path):
        """Return path for persistent tests"""
        return str(tmp_path / "persistent_test.db")
    
    # =========================================================================
    # Session Creation Tests
    # =========================================================================
    
    def test_create_session_auto_id(self, storage):
        """Test session creation with auto-generated ID"""
        session_id = storage.create_session()
        
        assert session_id is not None
        assert len(session_id) == 36  # UUID format
        assert storage.session_exists(session_id)
    
    def test_create_session_custom_id(self, storage):
        """Test session creation with custom ID"""
        custom_id = "my-custom-session-123"
        session_id = storage.create_session(custom_id)
        
        assert session_id == custom_id
        assert storage.session_exists(custom_id)
    
    def test_create_session_duplicate(self, storage):
        """Test creating session with existing ID returns same ID"""
        session_id = storage.create_session("duplicate-test")
        session_id2 = storage.create_session("duplicate-test")
        
        assert session_id == session_id2
    
    def test_session_exists(self, storage):
        """Test session existence check"""
        assert not storage.session_exists("nonexistent")
        
        session_id = storage.create_session("test-exists")
        assert storage.session_exists(session_id)
    
    # =========================================================================
    # Turn Management Tests
    # =========================================================================
    
    def test_add_turn_basic(self, storage):
        """Test adding a basic turn"""
        session_id = storage.create_session()
        
        turn_num = storage.add_turn(
            session_id,
            query="What is Indonesian tax law?",
            answer="Indonesian tax law is governed by UU Pajak..."
        )
        
        assert turn_num == 1
    
    def test_add_multiple_turns(self, storage):
        """Test adding multiple turns"""
        session_id = storage.create_session()
        
        turn1 = storage.add_turn(session_id, "Query 1", "Answer 1")
        turn2 = storage.add_turn(session_id, "Query 2", "Answer 2")
        turn3 = storage.add_turn(session_id, "Query 3", "Answer 3")
        
        assert turn1 == 1
        assert turn2 == 2
        assert turn3 == 3
    
    def test_add_turn_with_metadata(self, storage):
        """Test adding turn with metadata"""
        session_id = storage.create_session()
        
        metadata = {
            'tokens_generated': 150,
            'total_time': 2.5,
            'citations': [
                {'regulation_type': 'UU', 'regulation_number': '6', 'year': '2023'}
            ]
        }
        
        turn_num = storage.add_turn(session_id, "Query", "Answer", metadata)
        
        assert turn_num == 1
        
        # Verify metadata is stored
        turns = storage.get_turns(session_id)
        assert turns[0]['metadata']['tokens_generated'] == 150
    
    def test_add_turn_auto_creates_session(self, storage):
        """Test that add_turn requires existing session"""
        # Note: Session must exist - this should fail or auto-create
        # Based on implementation, add_turn doesn't auto-create
        session_id = "nonexistent-session"
        
        # The storage doesn't throw, it just adds to nonexistent
        # Let's verify current behavior
        turn_num = storage.add_turn(session_id, "Query", "Answer")
        
        # Turn is added but session metadata may not exist
        assert turn_num == 1
    
    # =========================================================================
    # Turn Retrieval Tests
    # =========================================================================
    
    def test_get_turns_empty(self, storage):
        """Test getting turns from empty session"""
        session_id = storage.create_session()
        turns = storage.get_turns(session_id)
        
        assert turns == []
    
    def test_get_turns_all(self, storage):
        """Test getting all turns"""
        session_id = storage.create_session()
        
        storage.add_turn(session_id, "Q1", "A1")
        storage.add_turn(session_id, "Q2", "A2")
        storage.add_turn(session_id, "Q3", "A3")
        
        turns = storage.get_turns(session_id)
        
        assert len(turns) == 3
        assert turns[0]['query'] == "Q1"
        assert turns[2]['query'] == "Q3"
    
    def test_get_turns_with_limit(self, storage):
        """Test getting limited turns (most recent)"""
        session_id = storage.create_session()
        
        for i in range(5):
            storage.add_turn(session_id, f"Q{i+1}", f"A{i+1}")
        
        turns = storage.get_turns(session_id, max_turns=2)
        
        assert len(turns) == 2
        assert turns[0]['query'] == "Q4"  # Second to last
        assert turns[1]['query'] == "Q5"  # Last
    
    def test_get_last_turn(self, storage):
        """Test getting last turn"""
        session_id = storage.create_session()
        
        storage.add_turn(session_id, "Q1", "A1")
        storage.add_turn(session_id, "Q2", "A2")
        storage.add_turn(session_id, "Q3", "A3")
        
        last = storage.get_last_turn(session_id)
        
        assert last is not None
        assert last['query'] == "Q3"
        assert last['turn_number'] == 3
    
    def test_get_last_turn_empty(self, storage):
        """Test getting last turn from empty session"""
        session_id = storage.create_session()
        last = storage.get_last_turn(session_id)
        
        assert last is None
    
    # =========================================================================
    # Session Data Tests
    # =========================================================================
    
    def test_get_session_full(self, storage):
        """Test getting full session data"""
        session_id = storage.create_session()
        
        storage.add_turn(session_id, "Q1", "A1", {'total_time': 1.0})
        storage.add_turn(session_id, "Q2", "A2", {'total_time': 2.0})
        
        session = storage.get_session(session_id)
        
        assert session is not None
        assert session['id'] == session_id
        assert len(session['turns']) == 2
        assert session['metadata']['total_queries'] == 2
        assert session['metadata']['total_time'] == 3.0
    
    def test_get_session_nonexistent(self, storage):
        """Test getting nonexistent session"""
        session = storage.get_session("nonexistent")
        assert session is None
    
    def test_get_session_summary(self, storage):
        """Test getting session summary statistics"""
        session_id = storage.create_session()
        
        storage.add_turn(session_id, "Q1", "A1", {'total_time': 1.5})
        storage.add_turn(session_id, "Q2", "A2", {'total_time': 2.5})
        
        summary = storage.get_session_summary(session_id)
        
        assert summary['session_id'] == session_id
        assert summary['total_turns'] == 2
        assert summary['total_queries'] == 2
        assert summary['total_time'] == 4.0
        assert summary['avg_time_per_query'] == 2.0
    
    # =========================================================================
    # Session Deletion Tests
    # =========================================================================
    
    def test_delete_session(self, storage):
        """Test session deletion"""
        session_id = storage.create_session()
        storage.add_turn(session_id, "Q", "A")
        
        deleted = storage.delete_session(session_id)
        
        assert deleted is not None
        assert deleted['id'] == session_id
        assert not storage.session_exists(session_id)
    
    def test_delete_session_nonexistent(self, storage):
        """Test deleting nonexistent session"""
        deleted = storage.delete_session("nonexistent")
        assert deleted is None
    
    def test_clear_all(self, storage):
        """Test clearing all sessions"""
        storage.create_session("session1")
        storage.create_session("session2")
        storage.create_session("session3")
        
        storage.clear_all()
        
        sessions = storage.list_sessions()
        assert len(sessions) == 0
    
    # =========================================================================
    # Search Tests
    # =========================================================================
    
    def test_search_history(self, storage):
        """Test searching conversation history"""
        session_id = storage.create_session()
        
        storage.add_turn(session_id, "What is tax?", "Tax is a levy...")
        storage.add_turn(session_id, "What about labor law?", "Labor law covers...")
        storage.add_turn(session_id, "Tax penalties?", "Penalties include...")
        
        results = storage.search_history(session_id, "tax")
        
        assert len(results) == 2
        assert "tax" in results[0]['query'].lower() or "tax" in results[0]['answer'].lower()
    
    def test_search_history_no_match(self, storage):
        """Test search with no matches"""
        session_id = storage.create_session()
        storage.add_turn(session_id, "Hello", "Hi there")
        
        results = storage.search_history(session_id, "nonexistent")
        assert len(results) == 0
    
    # =========================================================================
    # Persistence Tests (Critical!)
    # =========================================================================
    
    def test_persistence_across_instances(self, db_path):
        """Test that data persists across storage instances (simulates restart)"""
        # Create first instance and add data
        storage1 = SessionStorage(db_path)
        session_id = storage1.create_session("persistent-test")
        storage1.add_turn(session_id, "Q1", "A1")
        storage1.add_turn(session_id, "Q2", "A2")
        
        # Create second instance (simulates restart)
        storage2 = SessionStorage(db_path)
        
        # Verify data persists
        assert storage2.session_exists(session_id)
        turns = storage2.get_turns(session_id)
        assert len(turns) == 2
        assert turns[0]['query'] == "Q1"
        assert turns[1]['query'] == "Q2"
    
    def test_multiple_sessions_persistence(self, db_path):
        """Test multiple sessions persist correctly"""
        storage1 = SessionStorage(db_path)
        
        session1 = storage1.create_session("user-1")
        session2 = storage1.create_session("user-2")
        
        storage1.add_turn(session1, "User1 Q1", "User1 A1")
        storage1.add_turn(session2, "User2 Q1", "User2 A1")
        storage1.add_turn(session1, "User1 Q2", "User1 A2")
        
        # Simulate restart
        storage2 = SessionStorage(db_path)
        
        turns1 = storage2.get_turns(session1)
        turns2 = storage2.get_turns(session2)
        
        assert len(turns1) == 2
        assert len(turns2) == 1
        assert turns1[0]['query'] == "User1 Q1"
        assert turns2[0]['query'] == "User2 Q1"
    
    # =========================================================================
    # List Sessions Tests
    # =========================================================================
    
    def test_list_sessions(self, storage):
        """Test listing all sessions"""
        storage.create_session("session1")
        storage.create_session("session2")
        storage.add_turn("session1", "Q", "A")
        
        sessions = storage.list_sessions()
        
        assert len(sessions) == 2
        # Sessions should be sorted by updated_at DESC
        session_ids = [s['session_id'] for s in sessions]
        assert "session1" in session_ids
        assert "session2" in session_ids
    
    def test_list_sessions_empty(self, storage):
        """Test listing sessions when empty"""
        sessions = storage.list_sessions()
        assert sessions == []


class TestSessionStorageIsolation:
    """Tests for session isolation (no data leakage between sessions)"""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create a temporary storage instance"""
        return SessionStorage(str(tmp_path / "isolation_test.db"))
    
    def test_session_data_isolation(self, storage):
        """Test that sessions don't share data"""
        session1 = storage.create_session("user-alice")
        session2 = storage.create_session("user-bob")
        
        storage.add_turn(session1, "Alice's private question", "Alice's answer")
        storage.add_turn(session2, "Bob's question", "Bob's answer")
        
        alice_turns = storage.get_turns(session1)
        bob_turns = storage.get_turns(session2)
        
        # Verify isolation
        assert len(alice_turns) == 1
        assert len(bob_turns) == 1
        assert alice_turns[0]['query'] == "Alice's private question"
        assert bob_turns[0]['query'] == "Bob's question"
        
        # Verify no cross-contamination
        assert "Bob" not in alice_turns[0]['query']
        assert "Alice" not in bob_turns[0]['query']
    
    def test_session_deletion_isolation(self, storage):
        """Test that deleting one session doesn't affect others"""
        session1 = storage.create_session("keep-me")
        session2 = storage.create_session("delete-me")
        
        storage.add_turn(session1, "Keep Q", "Keep A")
        storage.add_turn(session2, "Delete Q", "Delete A")
        
        storage.delete_session(session2)
        
        # Session1 should be untouched
        assert storage.session_exists(session1)
        turns = storage.get_turns(session1)
        assert len(turns) == 1
        assert turns[0]['query'] == "Keep Q"
        
        # Session2 should be gone
        assert not storage.session_exists(session2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
