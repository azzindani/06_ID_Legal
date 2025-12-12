"""
Unified Memory Manager - Integrated conversation and context management

This module provides a unified interface for managing conversation memory,
combining ConversationManager (persistent history) and ContextCache (efficient caching).

Features:
- Single interface for all memory operations
- Automatic caching with LRU eviction
- Session management and history tracking
- Context-aware query retrieval
- Token limit enforcement
- Memory statistics and monitoring

File: conversation/memory_manager.py
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from logger_utils import get_logger

from .manager import ConversationManager
from .context_cache import ContextCache

logger = get_logger(__name__)


class MemoryManager:
    """
    Unified memory manager for conversational RAG

    Combines:
    - ConversationManager: Persistent conversation history and metadata
    - ContextCache: Efficient LRU caching with compression

    This provides a single, consistent interface for all memory operations
    across the entire application (Gradio, API, tests, CLI).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize memory manager

        Args:
            config: Optional configuration dictionary
                - max_history_turns: Maximum turns to keep in history (default: 50)
                - max_context_turns: Maximum turns to include in context (default: 10)
                - enable_cache: Enable context caching (default: True)
                - cache_size: LRU cache size (default: 100)
                - max_tokens: Max tokens per context (default: 8000)
        """
        self.config = config or {}
        self.logger = logger

        # Initialize conversation manager
        manager_config = {
            'max_history_turns': self.config.get('max_history_turns', 50),
            'max_context_turns': self.config.get('max_context_turns', 10)
        }
        self.conversation_manager = ConversationManager(manager_config)

        # Initialize context cache
        self.enable_cache = self.config.get('enable_cache', True)
        if self.enable_cache:
            cache_config = {
                'cache_size': self.config.get('cache_size', 100),
                'max_tokens': self.config.get('max_tokens', 8000),
                'compression': self.config.get('compression', True),
                'summary_threshold': self.config.get('summary_threshold', 10)
            }
            self.context_cache = ContextCache(cache_config)
        else:
            self.context_cache = None

        # Statistics
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'contexts_retrieved': 0,
            'turns_saved': 0
        }

        self.logger.info("MemoryManager initialized", {
            "caching_enabled": self.enable_cache,
            "max_history": manager_config['max_history_turns'],
            "max_context": manager_config['max_context_turns']
        })

    # ===== Session Management =====

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        session_id = self.conversation_manager.start_session(session_id)
        self.logger.info(f"Session started: {session_id}")
        return session_id

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End a conversation session and clean up

        Args:
            session_id: Session ID

        Returns:
            Final session data or None
        """
        # Clear cache
        if self.context_cache:
            self.context_cache.clear(session_id)

        # End session in manager
        session_data = self.conversation_manager.end_session(session_id)

        self.logger.info(f"Session ended: {session_id}")
        return session_data

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full session data

        Args:
            session_id: Session ID

        Returns:
            Session data or None
        """
        return self.conversation_manager.get_session(session_id)

    # ===== Memory Operations =====

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save a conversation turn with automatic cache update

        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Optional turn metadata

        Returns:
            Turn number
        """
        # Save to conversation manager
        turn_num = self.conversation_manager.add_turn(
            session_id,
            user_message,
            assistant_message,
            metadata
        )

        # Update cache if enabled
        if self.context_cache:
            # Add user turn
            self.context_cache.update(session_id, {
                'role': 'user',
                'content': user_message
            })

            # Add assistant turn
            self.context_cache.update(session_id, {
                'role': 'assistant',
                'content': assistant_message[:500]  # Truncate long responses
            })

        self._stats['turns_saved'] += 1

        self.logger.debug(f"Turn {turn_num} saved for session {session_id}")
        return turn_num

    def get_context(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get conversation context with intelligent caching

        This method:
        1. Checks cache first (if enabled and not forced refresh)
        2. Falls back to conversation manager
        3. Caches the result for future use

        Args:
            session_id: Session ID
            max_turns: Maximum turns to include (None = use config default)
            force_refresh: Force bypass cache and reload from manager

        Returns:
            List of context messages [{'role': 'user'/'assistant', 'content': str}]
        """
        self._stats['contexts_retrieved'] += 1

        # Try cache first (if enabled and not forced refresh)
        if self.context_cache and not force_refresh:
            cached_context = self.context_cache.get(session_id)
            if cached_context is not None:
                self._stats['cache_hits'] += 1
                self.logger.debug(f"Cache HIT for session {session_id}")

                # Apply max_turns if specified
                if max_turns and len(cached_context) > max_turns * 2:
                    # Each turn has 2 messages (user + assistant)
                    cached_context = cached_context[-(max_turns * 2):]

                return cached_context

            self._stats['cache_misses'] += 1
            self.logger.debug(f"Cache MISS for session {session_id}")

        # Get from conversation manager
        context = self.conversation_manager.get_context_for_query(
            session_id,
            max_turns
        )

        # Cache it for next time
        if self.context_cache and context:
            self.context_cache.put(session_id, context, {
                'session_id': session_id,
                'retrieved_at': datetime.now().isoformat()
            })

        return context or []

    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get raw conversation history (with full metadata)

        Unlike get_context(), this returns full turn data including metadata.
        Use this for exports, analysis, and auditing.

        Args:
            session_id: Session ID
            max_turns: Maximum turns to return (None = all)

        Returns:
            List of conversation turns with metadata
        """
        return self.conversation_manager.get_history(session_id, max_turns)

    # ===== Search and Query =====

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
        return self.conversation_manager.search_history(session_id, keyword)

    def get_last_turn(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the last turn in a session

        Args:
            session_id: Session ID

        Returns:
            Last turn or None
        """
        return self.conversation_manager.get_last_turn(session_id)

    # ===== Statistics and Monitoring =====

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get session summary statistics

        Args:
            session_id: Session ID

        Returns:
            Summary statistics
        """
        return self.conversation_manager.get_session_summary(session_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics

        Returns:
            Statistics including cache hits/misses, sessions, etc.
        """
        stats = {
            'manager_stats': self._stats.copy(),
            'sessions': len(self.conversation_manager.sessions),
            'active_sessions': list(self.conversation_manager.sessions.keys())
        }

        if self.context_cache:
            cache_stats = self.context_cache.get_stats()
            stats['cache_stats'] = cache_stats

            # Calculate cache hit rate
            total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
            if total_requests > 0:
                stats['cache_hit_rate'] = self._stats['cache_hits'] / total_requests
            else:
                stats['cache_hit_rate'] = 0.0
        else:
            stats['cache_stats'] = None
            stats['cache_hit_rate'] = None

        return stats

    def clear_cache(self, session_id: Optional[str] = None):
        """
        Clear context cache

        Args:
            session_id: Optional session ID to clear (None = clear all)
        """
        if self.context_cache:
            self.context_cache.clear(session_id)
            self.logger.info(f"Cache cleared: {session_id or 'ALL'}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions

        Returns:
            List of session summaries
        """
        return self.conversation_manager.list_sessions()

    def clear_all_sessions(self):
        """Clear all sessions and cache"""
        # Clear cache
        if self.context_cache:
            self.context_cache.clear()

        # Clear sessions
        self.conversation_manager.clear_all_sessions()

        # Reset stats
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'contexts_retrieved': 0,
            'turns_saved': 0
        }

        self.logger.info("All sessions and cache cleared")


# Factory function for easy creation
def create_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """
    Create a memory manager instance

    Args:
        config: Optional configuration

    Returns:
        MemoryManager instance
    """
    return MemoryManager(config)


# Global instance (optional)
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """
    Get or create global memory manager instance

    Args:
        config: Optional configuration (only used if creating new instance)

    Returns:
        Global MemoryManager instance
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = create_memory_manager(config)
    return _global_memory_manager


if __name__ == "__main__":
    print("=" * 80)
    print("MEMORY MANAGER TEST")
    print("=" * 80)

    # Create memory manager
    mm = create_memory_manager({
        'max_history_turns': 10,
        'max_context_turns': 5,
        'enable_cache': True,
        'cache_size': 10
    })

    # Test 1: Session management
    print("\n[Test 1: Session Management]")
    session_id = mm.start_session()
    print(f"✓ Session created: {session_id}")

    # Test 2: Save turns
    print("\n[Test 2: Save Turns]")
    mm.save_turn(session_id, "Apa itu UU Ketenagakerjaan?", "UU Ketenagakerjaan adalah...")
    mm.save_turn(session_id, "Jelaskan tentang pesangon", "Pesangon diatur dalam...")
    mm.save_turn(session_id, "Berapa perhitungannya?", "Perhitungan pesangon...")
    print(f"✓ Saved 3 turns")

    # Test 3: Get context (should hit cache on 2nd call)
    print("\n[Test 3: Context Retrieval with Caching]")
    context1 = mm.get_context(session_id)
    print(f"  First call: {len(context1)} messages")

    context2 = mm.get_context(session_id)
    print(f"  Second call: {len(context2)} messages (should be cached)")

    # Test 4: Get history
    print("\n[Test 4: Get History]")
    history = mm.get_history(session_id)
    print(f"  History: {len(history)} turns")
    for idx, turn in enumerate(history, 1):
        print(f"    Turn {idx}: {turn['query'][:30]}...")

    # Test 5: Statistics
    print("\n[Test 5: Statistics]")
    stats = mm.get_stats()
    print(f"  Sessions: {stats['sessions']}")
    print(f"  Turns saved: {stats['manager_stats']['turns_saved']}")
    print(f"  Contexts retrieved: {stats['manager_stats']['contexts_retrieved']}")
    print(f"  Cache hits: {stats['manager_stats']['cache_hits']}")
    print(f"  Cache misses: {stats['manager_stats']['cache_misses']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")

    # Test 6: Search
    print("\n[Test 6: Search History]")
    results = mm.search_history(session_id, "pesangon")
    print(f"  Found {len(results)} turns mentioning 'pesangon'")

    # Test 7: Session summary
    print("\n[Test 7: Session Summary]")
    summary = mm.get_session_summary(session_id)
    print(f"  Total turns: {summary['total_turns']}")
    print(f"  Created: {summary['created_at']}")

    # Test 8: Clear and end
    print("\n[Test 8: Cleanup]")
    mm.end_session(session_id)
    print(f"✓ Session ended")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
