"""
Conversation Manager - Session State and History Tracking

Manages conversation sessions with full metadata for the RAG pipeline.
Supports both persistent (SQLite) and in-memory storage.

File: conversation/manager.py
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from utils.logger_utils import get_logger


class ConversationManager:
    """
    Manages conversation sessions and history

    Features:
    - Session creation and management
    - Turn-by-turn history tracking
    - Metadata storage (sources, scores, timing)
    - Context retrieval for follow-up questions
    - Persistent storage (default) or in-memory option
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Conversation Manager

        Args:
            config: Optional configuration
                - persist: Use SQLite storage (default: True)
                - db_path: Custom database path
                - max_history_turns: Max turns to keep
                - max_context_turns: Max turns for context
        """
        self.logger = get_logger("ConversationManager")
        self.config = config or {}

        # Configuration
        self.max_history_turns = self.config.get('max_history_turns', 50)
        self.max_context_turns = self.config.get('max_context_turns', 5)
        
        # Persistence setting (default: enabled)
        self.persist = self.config.get('persist', True)
        
        if self.persist:
            # Use SQLite storage
            from .session_storage import SessionStorage
            db_path = self.config.get('db_path')
            self.storage = SessionStorage(db_path)
            self.sessions = None  # Not used with persistence
            self.logger.info("ConversationManager initialized with persistent storage", {
                "max_history_turns": self.max_history_turns,
                "max_context_turns": self.max_context_turns,
                "db_path": self.storage.db_path
            })
        else:
            # Use in-memory storage
            self.storage = None
            self.sessions: Dict[str, Dict[str, Any]] = {}
            self.logger.info("ConversationManager initialized with in-memory storage", {
                "max_history_turns": self.max_history_turns,
                "max_context_turns": self.max_context_turns
            })

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if self.persist:
            # Use persistent storage
            session_id = self.storage.create_session(session_id)
        else:
            # In-memory storage
            if session_id in self.sessions:
                self.logger.warning(f"Session {session_id} already exists, returning existing")
                return session_id

            self.sessions[session_id] = {
                'id': session_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'turns': [],
                'metadata': {
                    'total_queries': 0,
                    'total_tokens': 0,
                    'total_time': 0.0,
                    'regulations_cited': set()
                }
            }

        self.logger.info(f"Session started: {session_id}")
        return session_id

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
            metadata: Optional turn metadata (sources, scores, timing)

        Returns:
            Turn number
        """
        if self.persist:
            # Use persistent storage
            if not self.storage.session_exists(session_id):
                self.storage.create_session(session_id)
            turn_number = self.storage.add_turn(session_id, query, answer, metadata)
        else:
            # In-memory storage
            if session_id not in self.sessions:
                self.logger.error(f"Session not found: {session_id}")
                raise ValueError(f"Session not found: {session_id}")

            session = self.sessions[session_id]
            turn_number = len(session['turns']) + 1

            turn = {
                'turn_number': turn_number,
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'answer': answer,
                'metadata': metadata or {}
            }

            session['turns'].append(turn)
            session['updated_at'] = datetime.now().isoformat()
            session['metadata']['total_queries'] += 1

            if metadata:
                if 'tokens_generated' in metadata:
                    session['metadata']['total_tokens'] += metadata['tokens_generated']
                if 'total_time' in metadata:
                    session['metadata']['total_time'] += metadata['total_time']
                if 'citations' in metadata:
                    for citation in metadata['citations']:
                        reg_ref = f"{citation.get('regulation_type', '')} {citation.get('regulation_number', '')}/{citation.get('year', '')}"
                        session['metadata']['regulations_cited'].add(reg_ref)

            if len(session['turns']) > self.max_history_turns:
                session['turns'] = session['turns'][-self.max_history_turns:]

        self.logger.debug(f"Turn {turn_number} added to session {session_id}")
        return turn_number

    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history

        Args:
            session_id: Session ID
            max_turns: Maximum turns to return (None = all)

        Returns:
            List of conversation turns
        """
        if self.persist:
            return self.storage.get_turns(session_id, max_turns)
        else:
            if session_id not in self.sessions:
                return []
            turns = self.sessions[session_id]['turns']
            if max_turns:
                turns = turns[-max_turns:]
            return turns

    def get_context_for_query(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get formatted context for RAG pipeline

        Args:
            session_id: Session ID
            max_turns: Maximum turns to include

        Returns:
            List of {'role': 'user'/'assistant', 'content': str}
        """
        max_turns = max_turns or self.max_context_turns
        turns = self.get_history(session_id, max_turns)

        context = []
        for turn in turns:
            context.append({'role': 'user', 'content': turn['query']})
            context.append({'role': 'assistant', 'content': turn['answer']})

        return context

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full session data

        Args:
            session_id: Session ID

        Returns:
            Session data or None
        """
        if self.persist:
            return self.storage.get_session(session_id)
        else:
            if session_id not in self.sessions:
                return None
            session = self.sessions[session_id].copy()
            session['metadata']['regulations_cited'] = list(
                session['metadata']['regulations_cited']
            )
            return session

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get session summary statistics

        Args:
            session_id: Session ID

        Returns:
            Summary statistics
        """
        if self.persist:
            return self.storage.get_session_summary(session_id)
        else:
            if session_id not in self.sessions:
                return {}
            session = self.sessions[session_id]
            return {
                'session_id': session_id,
                'created_at': session['created_at'],
                'updated_at': session['updated_at'],
                'total_turns': len(session['turns']),
                'total_queries': session['metadata']['total_queries'],
                'total_tokens': session['metadata']['total_tokens'],
                'total_time': round(session['metadata']['total_time'], 2),
                'avg_time_per_query': round(
                    session['metadata']['total_time'] / max(session['metadata']['total_queries'], 1), 2
                ),
                'regulations_cited': len(session['metadata']['regulations_cited'])
            }

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End and remove a session

        Args:
            session_id: Session ID

        Returns:
            Final session data or None
        """
        if self.persist:
            session = self.storage.delete_session(session_id)
        else:
            if session_id not in self.sessions:
                self.logger.warning(f"Session not found: {session_id}")
                return None
            session = self.get_session(session_id)
            del self.sessions[session_id]

        if session:
            self.logger.info(f"Session ended: {session_id}", {
                "total_turns": len(session.get('turns', []))
            })
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions

        Returns:
            List of session summaries
        """
        if self.persist:
            return self.storage.list_sessions()
        else:
            return [self.get_session_summary(sid) for sid in self.sessions]

    def clear_all_sessions(self):
        """Clear all sessions"""
        if self.persist:
            self.storage.clear_all()
        else:
            count = len(self.sessions)
            self.sessions.clear()
            self.logger.info(f"Cleared {count} sessions")

    def get_last_turn(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the last turn in a session

        Args:
            session_id: Session ID

        Returns:
            Last turn or None
        """
        if self.persist:
            return self.storage.get_last_turn(session_id)
        else:
            if session_id not in self.sessions:
                return None
            turns = self.sessions[session_id]['turns']
            return turns[-1] if turns else None

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
        if self.persist:
            return self.storage.search_history(session_id, keyword)
        else:
            if session_id not in self.sessions:
                return []
            keyword_lower = keyword.lower()
            matches = []
            for turn in self.sessions[session_id]['turns']:
                if (keyword_lower in turn['query'].lower() or
                    keyword_lower in turn['answer'].lower()):
                    matches.append(turn)
            return matches
