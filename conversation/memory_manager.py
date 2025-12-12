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
        Initialize memory manager with enhanced legal consultation support

        Args:
            config: Optional configuration dictionary
                - max_history_turns: Maximum turns to keep in history (default: 100 for legal)
                - max_context_turns: Maximum turns to include in context (default: 30 for legal)
                - enable_cache: Enable context caching (default: True)
                - cache_size: LRU cache size (default: 100)
                - max_tokens: Max tokens per context (default: 16000 for legal)
                - enable_summarization: Auto-summarize old turns (default: True)
                - summarization_threshold: Start summarizing after N turns (default: 20)
                - enable_key_facts: Track important case facts (default: True)
                - min_context_turns: Minimum recent turns to keep detailed (default: 10)
        """
        self.config = config or {}
        self.logger = logger

        # Enhanced defaults for legal consultations
        self.max_history_turns = self.config.get('max_history_turns', 100)
        self.max_context_turns = self.config.get('max_context_turns', 30)
        self.min_context_turns = self.config.get('min_context_turns', 10)

        # Summarization settings
        self.enable_summarization = self.config.get('enable_summarization', True)
        self.summarization_threshold = self.config.get('summarization_threshold', 20)

        # Key facts tracking
        self.enable_key_facts = self.config.get('enable_key_facts', True)
        self.key_facts_storage = {}  # session_id -> list of key facts
        self.session_summaries = {}  # session_id -> consultation summary

        # Initialize conversation manager
        manager_config = {
            'max_history_turns': self.max_history_turns,
            'max_context_turns': self.max_context_turns
        }
        self.conversation_manager = ConversationManager(manager_config)

        # Initialize context cache
        self.enable_cache = self.config.get('enable_cache', True)
        if self.enable_cache:
            cache_config = {
                'cache_size': self.config.get('cache_size', 100),
                'max_tokens': self.config.get('max_tokens', 16000),  # Increased for legal
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
            'turns_saved': 0,
            'summaries_created': 0,
            'key_facts_extracted': 0
        }

        self.logger.info("MemoryManager initialized (Enhanced for Legal)", {
            "caching_enabled": self.enable_cache,
            "max_history": self.max_history_turns,
            "max_context": self.max_context_turns,
            "min_context": self.min_context_turns,
            "summarization": self.enable_summarization,
            "key_facts_tracking": self.enable_key_facts
        })

    # ===== Session Management =====

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session with enhanced tracking

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        session_id = self.conversation_manager.start_session(session_id)

        # Initialize key facts and summary storage for this session
        if self.enable_key_facts:
            self.key_facts_storage[session_id] = []
            self.session_summaries[session_id] = {
                'created_at': datetime.now().isoformat(),
                'topics_discussed': [],
                'regulations_mentioned': [],
                'key_points': [],
                'case_type': None
            }

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

        # Clean up key facts and summaries
        if session_id in self.key_facts_storage:
            del self.key_facts_storage[session_id]
        if session_id in self.session_summaries:
            del self.session_summaries[session_id]

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

    # ===== Enhanced Memory Operations =====

    def _extract_key_facts(
        self,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Extract key legal facts from a conversation turn

        Looks for:
        - Regulation references (UU, PP, Perpu, etc.)
        - Important dates, amounts, names
        - Legal actions or disputes mentioned
        - Critical case details

        Args:
            user_message: User's question
            assistant_message: Assistant's answer
            metadata: Optional metadata with sources

        Returns:
            List of key facts
        """
        key_facts = []

        # Extract regulations from metadata sources
        if metadata and 'sources' in metadata:
            for source in metadata['sources'][:3]:  # Top 3 sources
                reg_type = source.get('regulation_type', '')
                reg_num = source.get('regulation_number', '')
                year = source.get('year', '')
                if reg_type and reg_num:
                    key_facts.append(f"{reg_type} No. {reg_num} Tahun {year}")

        # Extract from user question - keywords that indicate important facts
        import re
        # Look for specific regulations mentioned
        uu_pattern = r'(UU|Undang-Undang|PP|Perpu|Perpres)\s+(?:No\.?|Nomor)\s*(\d+)\s+Tahun\s+(\d+)'
        matches = re.findall(uu_pattern, user_message, re.IGNORECASE)
        for match in matches:
            key_facts.append(f"{match[0]} No. {match[1]} Tahun {match[2]}")

        # Look for monetary amounts (important in legal cases)
        amount_pattern = r'(Rp\.?\s*[\d.,]+(?:\s*(?:juta|miliar|ribu))?)'
        amounts = re.findall(amount_pattern, user_message, re.IGNORECASE)
        for amount in amounts[:2]:  # Max 2 amounts
            key_facts.append(f"Amount: {amount}")

        # Look for dates
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        dates = re.findall(date_pattern, user_message)
        for date in dates[:2]:
            key_facts.append(f"Date: {date}")

        return key_facts

    def _update_session_summary(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        key_facts: List[str]
    ):
        """
        Update the ongoing session summary with new information

        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's answer
            key_facts: Extracted key facts
        """
        if session_id not in self.session_summaries:
            return

        summary = self.session_summaries[session_id]

        # Extract topic from user message (simplified)
        if any(word in user_message.lower() for word in ['kerja', 'pekerja', 'karyawan', 'pesangon']):
            topic = 'Ketenagakerjaan'
        elif any(word in user_message.lower() for word in ['lingkungan', 'amdal', 'izin lingkungan']):
            topic = 'Lingkungan Hidup'
        elif any(word in user_message.lower() for word in ['pajak', 'pph', 'ppn', 'keberatan']):
            topic = 'Perpajakan'
        elif any(word in user_message.lower() for word in ['tanah', 'sertifikat', 'hak guna']):
            topic = 'Pertanahan'
        else:
            topic = 'Hukum Umum'

        if topic not in summary['topics_discussed']:
            summary['topics_discussed'].append(topic)

        # Add regulations
        for fact in key_facts:
            if 'UU' in fact or 'PP' in fact or 'Perpu' in fact:
                if fact not in summary['regulations_mentioned']:
                    summary['regulations_mentioned'].append(fact)

        # Add key points (summary of question)
        point = user_message[:100] + "..." if len(user_message) > 100 else user_message
        if point not in summary['key_points']:
            summary['key_points'].append(point)

    def _create_intelligent_context(
        self,
        session_id: str,
        all_turns: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Create intelligent context that includes:
        1. Session summary (if available)
        2. Key facts (never forgotten)
        3. Recent turns (detailed, last N turns)
        4. Older turns summary (if past threshold)

        This ensures long conversations don't lose important early context.

        Args:
            session_id: Session ID
            all_turns: All conversation turns

        Returns:
            Intelligent context list
        """
        context = []
        turn_count = len(all_turns)

        # 1. Add session summary as system message (if exists and significant conversation)
        if turn_count > 5 and session_id in self.session_summaries:
            summary = self.session_summaries[session_id]
            if summary['topics_discussed'] or summary['regulations_mentioned']:
                summary_text = "Consultation Summary:\n"
                if summary['topics_discussed']:
                    summary_text += f"Topics: {', '.join(summary['topics_discussed'])}\n"
                if summary['regulations_mentioned']:
                    summary_text += f"Regulations: {', '.join(summary['regulations_mentioned'][:5])}\n"

                context.append({
                    'role': 'system',
                    'content': summary_text
                })

        # 2. Add key facts as system message (if exists)
        if session_id in self.key_facts_storage and self.key_facts_storage[session_id]:
            key_facts = self.key_facts_storage[session_id][:10]  # Max 10 key facts
            facts_text = "Key Facts:\n" + "\n".join(f"• {fact}" for fact in key_facts)
            context.append({
                'role': 'system',
                'content': facts_text
            })

        # 3. Decide on strategy based on conversation length
        if turn_count <= self.max_context_turns:
            # Short conversation - include everything
            for turn in all_turns:
                context.append({'role': 'user', 'content': turn['query']})
                context.append({'role': 'assistant', 'content': turn['answer'][:1000]})

        else:
            # Long conversation - summarize old, keep recent detailed
            older_turns = all_turns[:-self.min_context_turns]
            recent_turns = all_turns[-self.min_context_turns:]

            # Summarize older turns if enabled
            if self.enable_summarization and older_turns:
                summary_text = self._summarize_turns(older_turns)
                context.append({
                    'role': 'system',
                    'content': f"Previous discussion (turns 1-{len(older_turns)}):\n{summary_text}"
                })
                self._stats['summaries_created'] += 1

            # Add recent turns in detail
            for turn in recent_turns:
                context.append({'role': 'user', 'content': turn['query']})
                context.append({'role': 'assistant', 'content': turn['answer'][:1000]})

        return context

    def _summarize_turns(self, turns: List[Dict[str, Any]]) -> str:
        """
        Create a concise summary of multiple turns

        Args:
            turns: List of conversation turns

        Returns:
            Summary text
        """
        if not turns:
            return ""

        summary_parts = []
        for i, turn in enumerate(turns, 1):
            # Extract key points from each turn
            user_q = turn['query'][:150]
            # Just list the questions asked
            summary_parts.append(f"{i}. {user_q}...")

        return "\n".join(summary_parts)

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save a conversation turn with intelligent tracking

        Features:
        - Saves to conversation manager
        - Extracts and stores key legal facts
        - Updates session summary
        - Updates cache

        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Optional turn metadata

        Returns:
            Turn number
        """
        # Extract key facts if enabled
        if self.enable_key_facts:
            key_facts = self._extract_key_facts(user_message, assistant_message, metadata)
            if key_facts:
                if session_id not in self.key_facts_storage:
                    self.key_facts_storage[session_id] = []
                self.key_facts_storage[session_id].extend(key_facts)
                self._stats['key_facts_extracted'] += len(key_facts)

                # Update session summary
                self._update_session_summary(session_id, user_message, assistant_message, key_facts)

        # Save to conversation manager
        turn_num = self.conversation_manager.add_turn(
            session_id,
            user_message,
            assistant_message,
            metadata
        )

        # Clear cache to force rebuild with new intelligent context
        # This ensures the next get_context() uses updated key facts and summary
        if self.context_cache:
            self.context_cache.clear(session_id)

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
        Get INTELLIGENT conversation context with:
        - Session summary (topics, regulations)
        - Key facts (never forgotten)
        - Recent turns (detailed)
        - Older turns (summarized)

        This ensures long legal consultations maintain full context.

        Args:
            session_id: Session ID
            max_turns: Maximum turns to include (None = use config default)
            force_refresh: Force bypass cache and reload from manager

        Returns:
            List of context messages [{'role': 'user'/'assistant'/'system', 'content': str}]
        """
        self._stats['contexts_retrieved'] += 1

        # Try cache first (if enabled and not forced refresh)
        if self.context_cache and not force_refresh:
            cached_context = self.context_cache.get(session_id)
            if cached_context is not None:
                self._stats['cache_hits'] += 1
                self.logger.debug(f"Cache HIT for session {session_id}")
                return cached_context

            self._stats['cache_misses'] += 1
            self.logger.debug(f"Cache MISS for session {session_id}")

        # Get all turns from conversation manager
        all_turns = self.conversation_manager.get_history(session_id, max_turns=None)

        if not all_turns:
            return []

        # Build INTELLIGENT context with summaries and key facts
        context = self._create_intelligent_context(session_id, all_turns)

        # Cache the intelligent context
        if self.context_cache and context:
            self.context_cache.put(session_id, context, {
                'session_id': session_id,
                'retrieved_at': datetime.now().isoformat(),
                'intelligent_context': True
            })

        self.logger.debug(f"Built intelligent context: {len(context)} messages for {len(all_turns)} turns")

        return context

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
        Get enhanced memory manager statistics

        Returns:
            Statistics including cache, key facts, summaries, etc.
        """
        stats = {
            'manager_stats': self._stats.copy(),
            'sessions': len(self.conversation_manager.sessions),
            'active_sessions': list(self.conversation_manager.sessions.keys()),
            'total_key_facts': sum(len(facts) for facts in self.key_facts_storage.values()),
            'sessions_with_summaries': len(self.session_summaries)
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

        # Clear key facts and summaries
        self.key_facts_storage.clear()
        self.session_summaries.clear()

        # Clear sessions
        self.conversation_manager.clear_all_sessions()

        # Reset stats
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'contexts_retrieved': 0,
            'turns_saved': 0,
            'summaries_created': 0,
            'key_facts_extracted': 0
        }

        self.logger.info("All sessions and cache cleared")

    # ===== Legal-Specific Features =====

    def get_key_facts(self, session_id: str) -> List[str]:
        """
        Get extracted key facts for a session

        Args:
            session_id: Session ID

        Returns:
            List of key legal facts
        """
        return self.key_facts_storage.get(session_id, [])

    def get_session_summary_dict(self, session_id: str) -> Dict[str, Any]:
        """
        Get consultation summary for a session

        Args:
            session_id: Session ID

        Returns:
            Summary dictionary with topics, regulations, key points
        """
        return self.session_summaries.get(session_id, {})


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
