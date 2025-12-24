"""
Integration Tests for Multi-User Session Management

Tests concurrent session handling as would occur with the API serving
multiple users simultaneously. Validates session isolation, parallel
operations, and data persistence.

File: tests/integration/test_multi_user_sessions.py
"""

import os
import sys
import pytest
import asyncio
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conversation.manager import ConversationManager
from conversation.session_storage import SessionStorage


class TestMultiUserSessions:
    """Tests for multi-user concurrent session scenarios (API-like usage)"""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a persistent conversation manager"""
        db_path = str(tmp_path / "multi_user_test.db")
        return ConversationManager({'persist': True, 'db_path': db_path})
    
    @pytest.fixture
    def db_path(self, tmp_path):
        """Return path for concurrent tests"""
        return str(tmp_path / "concurrent_test.db")
    
    # =========================================================================
    # Concurrent Session Creation Tests
    # =========================================================================
    
    def test_concurrent_session_creation(self, db_path):
        """Test creating sessions from multiple threads (simulates API requests)"""
        num_users = 10
        created_sessions = []
        
        def create_user_session(user_id):
            manager = ConversationManager({'persist': True, 'db_path': db_path})
            session_id = manager.start_session(f"user-{user_id}")
            manager.add_turn(session_id, f"User {user_id} query", f"Response to user {user_id}")
            return session_id
        
        # Simulate concurrent API requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_user_session, i) for i in range(num_users)]
            for future in as_completed(futures):
                created_sessions.append(future.result())
        
        # Verify all sessions were created
        assert len(created_sessions) == num_users
        assert len(set(created_sessions)) == num_users  # All unique
        
        # Verify persistence
        verify_manager = ConversationManager({'persist': True, 'db_path': db_path})
        sessions = verify_manager.list_sessions()
        assert len(sessions) == num_users
    
    def test_concurrent_turn_additions(self, db_path):
        """Test adding turns concurrently to different sessions"""
        # Setup: Create sessions first
        setup_manager = ConversationManager({'persist': True, 'db_path': db_path})
        session_ids = [setup_manager.start_session(f"concurrent-{i}") for i in range(5)]
        
        def add_turns_to_session(session_id, turn_count):
            manager = ConversationManager({'persist': True, 'db_path': db_path})
            for i in range(turn_count):
                manager.add_turn(session_id, f"Query {i}", f"Answer {i}")
            return session_id, turn_count
        
        # Add turns concurrently
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(add_turns_to_session, sid, 3)
                for sid in session_ids
            ]
            for future in as_completed(futures):
                results.append(future.result())
        
        # Verify all turns were added
        verify_manager = ConversationManager({'persist': True, 'db_path': db_path})
        for session_id in session_ids:
            turns = verify_manager.get_history(session_id)
            assert len(turns) == 3, f"Session {session_id} should have 3 turns"
    
    def test_interleaved_operations(self, db_path):
        """Test interleaved read/write operations (realistic API pattern)"""
        setup_manager = ConversationManager({'persist': True, 'db_path': db_path})
        session_id = setup_manager.start_session("interleaved-test")
        
        errors = []
        
        def writer():
            """Simulates API writes"""
            try:
                manager = ConversationManager({'persist': True, 'db_path': db_path})
                for i in range(10):
                    manager.add_turn(session_id, f"Write query {i}", f"Write answer {i}")
            except Exception as e:
                errors.append(f"Writer error: {e}")
        
        def reader():
            """Simulates API reads"""
            try:
                manager = ConversationManager({'persist': True, 'db_path': db_path})
                for _ in range(20):
                    manager.get_history(session_id)
                    manager.get_session_summary(session_id)
            except Exception as e:
                errors.append(f"Reader error: {e}")
        
        # Run writer and readers concurrently
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        
        # Check no errors
        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"
        
        # Verify final state
        verify_manager = ConversationManager({'persist': True, 'db_path': db_path})
        turns = verify_manager.get_history(session_id)
        assert len(turns) == 10
    
    # =========================================================================
    # Session Isolation Tests
    # =========================================================================
    
    def test_session_isolation_no_data_leakage(self, manager):
        """Test that sessions don't leak data between users"""
        # Create sessions for different "users"
        alice_session = manager.start_session("alice-secret")
        bob_session = manager.start_session("bob-private")
        
        # Alice's private data
        manager.add_turn(alice_session, 
            "My secret password is 12345",
            "I'll keep that safe for you, Alice")
        
        # Bob's data
        manager.add_turn(bob_session,
            "What is the weather?",
            "It's sunny today")
        
        # Verify isolation
        alice_history = manager.get_history(alice_session)
        bob_history = manager.get_history(bob_session)
        
        # No cross-contamination
        assert len(alice_history) == 1
        assert len(bob_history) == 1
        assert "secret" not in bob_history[0]['query'].lower()
        assert "weather" not in alice_history[0]['query'].lower()
    
    def test_session_context_isolation(self, manager):
        """Test that conversation context is properly isolated"""
        session1 = manager.start_session()
        session2 = manager.start_session()
        
        # Build different contexts
        manager.add_turn(session1, "Topic: Tax Law", "Let's discuss tax law")
        manager.add_turn(session1, "What about VAT?", "VAT is 11% in Indonesia")
        
        manager.add_turn(session2, "Topic: Labor Law", "Let's discuss labor law")
        manager.add_turn(session2, "Minimum wage?", "Varies by region")
        
        # Get context for each
        context1 = manager.get_context_for_query(session1)
        context2 = manager.get_context_for_query(session2)
        
        # Verify contexts are separate
        context1_text = " ".join([c['content'] for c in context1])
        context2_text = " ".join([c['content'] for c in context2])
        
        assert "Tax" in context1_text
        assert "Tax" not in context2_text
        assert "Labor" in context2_text
        assert "Labor" not in context1_text
    
    # =========================================================================
    # API Simulation Tests
    # =========================================================================
    
    def test_api_like_session_flow(self, db_path):
        """Simulate typical API session flow"""
        # Request 1: Start session
        manager1 = ConversationManager({'persist': True, 'db_path': db_path})
        session_id = manager1.start_session()
        
        # Request 2: First query (new manager instance = new API request)
        manager2 = ConversationManager({'persist': True, 'db_path': db_path})
        manager2.add_turn(session_id, "What is UU Ketenagakerjaan?", "UU No 13/2003...")
        
        # Request 3: Follow-up query
        manager3 = ConversationManager({'persist': True, 'db_path': db_path})
        context = manager3.get_context_for_query(session_id)
        manager3.add_turn(session_id, "What about article 156?", "Article 156 covers...")
        
        # Request 4: Get history
        manager4 = ConversationManager({'persist': True, 'db_path': db_path})
        history = manager4.get_history(session_id)
        
        # Verify state persisted across "requests"
        assert len(history) == 2
        assert "UU Ketenagakerjaan" in history[0]['query']
        assert "article 156" in history[1]['query']
    
    def test_multi_worker_simulation(self, db_path):
        """Simulate multiple API workers accessing same database"""
        num_workers = 4
        requests_per_worker = 5
        
        def worker_task(worker_id):
            """Simulate a worker handling requests"""
            results = []
            for req in range(requests_per_worker):
                manager = ConversationManager({'persist': True, 'db_path': db_path})
                session_id = f"worker-{worker_id}-req-{req}"
                manager.start_session(session_id)
                manager.add_turn(session_id, f"Query from worker {worker_id}", f"Response {req}")
                results.append(session_id)
            return results
        
        # Run workers in parallel
        all_sessions = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, w) for w in range(num_workers)]
            for future in as_completed(futures):
                all_sessions.extend(future.result())
        
        # Verify all sessions exist
        verify_manager = ConversationManager({'persist': True, 'db_path': db_path})
        sessions = verify_manager.list_sessions()
        assert len(sessions) == num_workers * requests_per_worker
    
    # =========================================================================
    # Edge Cases
    # =========================================================================
    
    def test_empty_sessions_across_workers(self, db_path):
        """Test handling of empty sessions created by different workers"""
        # Worker 1 creates session
        manager1 = ConversationManager({'persist': True, 'db_path': db_path})
        session_id = manager1.start_session("empty-session")
        
        # Worker 2 tries to get history (should be empty, not error)
        manager2 = ConversationManager({'persist': True, 'db_path': db_path})
        history = manager2.get_history(session_id)
        
        assert history == []
    
    def test_session_summary_concurrent_updates(self, db_path):
        """Test session summary reflects concurrent updates"""
        manager = ConversationManager({'persist': True, 'db_path': db_path})
        session_id = manager.start_session("summary-test")
        
        def add_turns():
            m = ConversationManager({'persist': True, 'db_path': db_path})
            for _ in range(5):
                m.add_turn(session_id, "Q", "A", {'total_time': 1.0})
        
        threads = [threading.Thread(target=add_turns) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify summary is accurate
        verify = ConversationManager({'persist': True, 'db_path': db_path})
        summary = verify.get_session_summary(session_id)
        
        assert summary['total_turns'] == 15  # 3 threads × 5 turns
        assert summary['total_time'] == 15.0  # 15 × 1.0s


class TestBackwardCompatibility:
    """Tests for backward compatibility with in-memory mode"""
    
    def test_in_memory_mode_still_works(self):
        """Test that persist=False still uses in-memory storage"""
        manager = ConversationManager({'persist': False})
        
        session_id = manager.start_session()
        manager.add_turn(session_id, "Test", "Response")
        
        history = manager.get_history(session_id)
        assert len(history) == 1
        
        # Verify it's using in-memory (sessions dict should exist)
        assert manager.sessions is not None
    
    def test_default_is_persistent(self):
        """Test that default behavior is now persistent"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "default_test.db")
            manager = ConversationManager({'db_path': db_path})
            
            # Should be persistent by default
            assert manager.persist == True
            assert manager.storage is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
