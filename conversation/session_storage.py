"""
Session Storage - SQLite-backed persistent storage for conversation sessions

Provides persistent storage for conversation sessions using SQLite.
Sessions survive server restarts and can be accessed across processes.

File: conversation/session_storage.py
"""

import sqlite3
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class SessionStorage:
    """
    SQLite-backed session storage for conversation persistence
    
    Features:
    - Persistent storage across server restarts
    - Thread-safe database access
    - Automatic schema migration
    - JSON serialization for complex metadata
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize session storage
        
        Args:
            db_path: Path to SQLite database file (default: .data/sessions.db)
        """
        if db_path is None:
            data_dir = Path(".data")
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "sessions.db")
        
        self.db_path = db_path
        self._init_database()
        
        logger.info(f"SessionStorage initialized", {"db_path": db_path})
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with automatic cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_queries INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0.0,
                    regulations_cited TEXT DEFAULT '[]'
                )
            ''')
            
            # Turns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            ''')
            
            # Index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_turns_session 
                ON turns(session_id, turn_number)
            ''')
            
            logger.debug("Database schema initialized")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
            if cursor.fetchone():
                logger.debug(f"Session already exists: {session_id}")
                return session_id
            
            cursor.execute('''
                INSERT INTO sessions (id, created_at, updated_at, total_queries, total_tokens, total_time, regulations_cited)
                VALUES (?, ?, ?, 0, 0, 0.0, '[]')
            ''', (session_id, now, now))
        
        logger.info(f"Session created: {session_id}")
        return session_id
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
            return cursor.fetchone() is not None
    
    def add_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a conversation turn
        
        Args:
            session_id: Session ID
            query: User query
            answer: Assistant answer
            metadata: Optional turn metadata
            
        Returns:
            Turn number
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current turn count
            cursor.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ?",
                (session_id,)
            )
            turn_number = cursor.fetchone()[0] + 1
            
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})
            
            # Insert turn
            cursor.execute('''
                INSERT INTO turns (session_id, turn_number, timestamp, query, answer, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, turn_number, now, query, answer, metadata_json))
            
            # Update session metadata
            cursor.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id)
            )
            
            # Update counters
            tokens = (metadata or {}).get('tokens_generated', 0)
            time_taken = (metadata or {}).get('total_time', 0.0)
            
            cursor.execute('''
                UPDATE sessions 
                SET total_queries = total_queries + 1,
                    total_tokens = total_tokens + ?,
                    total_time = total_time + ?
                WHERE id = ?
            ''', (tokens, time_taken, session_id))
            
            # Update regulations cited
            if metadata and 'citations' in metadata:
                cursor.execute(
                    "SELECT regulations_cited FROM sessions WHERE id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                existing = set(json.loads(row[0])) if row else set()
                
                for citation in metadata['citations']:
                    reg_ref = f"{citation.get('regulation_type', '')} {citation.get('regulation_number', '')}/{citation.get('year', '')}"
                    existing.add(reg_ref)
                
                cursor.execute(
                    "UPDATE sessions SET regulations_cited = ? WHERE id = ?",
                    (json.dumps(list(existing)), session_id)
                )
        
        logger.debug(f"Turn {turn_number} added to session {session_id}")
        return turn_number
    
    def get_turns(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation turns
        
        Args:
            session_id: Session ID
            max_turns: Maximum turns to return (None = all)
            
        Returns:
            List of turns
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if max_turns:
                cursor.execute('''
                    SELECT turn_number, timestamp, query, answer, metadata
                    FROM turns
                    WHERE session_id = ?
                    ORDER BY turn_number DESC
                    LIMIT ?
                ''', (session_id, max_turns))
                rows = list(reversed(cursor.fetchall()))
            else:
                cursor.execute('''
                    SELECT turn_number, timestamp, query, answer, metadata
                    FROM turns
                    WHERE session_id = ?
                    ORDER BY turn_number ASC
                ''', (session_id,))
                rows = cursor.fetchall()
            
            return [
                {
                    'turn_number': row['turn_number'],
                    'timestamp': row['timestamp'],
                    'query': row['query'],
                    'answer': row['answer'],
                    'metadata': json.loads(row['metadata'])
                }
                for row in rows
            ]
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full session data
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, created_at, updated_at, total_queries, 
                       total_tokens, total_time, regulations_cited
                FROM sessions WHERE id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'id': row['id'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'turns': self.get_turns(session_id),
                'metadata': {
                    'total_queries': row['total_queries'],
                    'total_tokens': row['total_tokens'],
                    'total_time': row['total_time'],
                    'regulations_cited': json.loads(row['regulations_cited'])
                }
            }
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, created_at, updated_at, total_queries,
                       total_tokens, total_time, regulations_cited
                FROM sessions WHERE id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            # Get turn count
            cursor.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ?",
                (session_id,)
            )
            turn_count = cursor.fetchone()[0]
            
            return {
                'session_id': session_id,
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'total_turns': turn_count,
                'total_queries': row['total_queries'],
                'total_tokens': row['total_tokens'],
                'total_time': round(row['total_time'], 2),
                'avg_time_per_query': round(
                    row['total_time'] / max(row['total_queries'], 1), 2
                ),
                'regulations_cited': len(json.loads(row['regulations_cited']))
            }
    
    def delete_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Delete a session and return its data
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data before deletion or None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        logger.info(f"Session deleted: {session_id}")
        return session
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with summaries"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC")
            
            return [
                self.get_session_summary(row['id'])
                for row in cursor.fetchall()
            ]
    
    def clear_all(self):
        """Clear all sessions"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM turns")
            cursor.execute("DELETE FROM sessions")
        
        logger.info("All sessions cleared")
    
    def search_history(
        self,
        session_id: str,
        keyword: str
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history for keyword
        
        Args:
            session_id: Session ID
            keyword: Search keyword
            
        Returns:
            Matching turns
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            keyword_pattern = f"%{keyword}%"
            cursor.execute('''
                SELECT turn_number, timestamp, query, answer, metadata
                FROM turns
                WHERE session_id = ? 
                AND (query LIKE ? OR answer LIKE ?)
                ORDER BY turn_number ASC
            ''', (session_id, keyword_pattern, keyword_pattern))
            
            return [
                {
                    'turn_number': row['turn_number'],
                    'timestamp': row['timestamp'],
                    'query': row['query'],
                    'answer': row['answer'],
                    'metadata': json.loads(row['metadata'])
                }
                for row in cursor.fetchall()
            ]
    
    def get_last_turn(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the last turn in a session"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT turn_number, timestamp, query, answer, metadata
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_number DESC
                LIMIT 1
            ''', (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'turn_number': row['turn_number'],
                'timestamp': row['timestamp'],
                'query': row['query'],
                'answer': row['answer'],
                'metadata': json.loads(row['metadata'])
            }
