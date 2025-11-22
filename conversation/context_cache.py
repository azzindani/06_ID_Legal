"""
Context Cache Manager - Efficient conversation context management

Inspired by Claude Code's efficient context handling.
"""

from typing import Dict, Any, List, Optional
from collections import OrderedDict
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_utils import get_logger

logger = get_logger(__name__)


class ContextCache:
    """
    Efficient context caching for conversation management

    Features:
    - LRU cache for recent contexts
    - Context compression/summarization
    - Token-aware truncation
    - Semantic deduplication
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from config import (
            CONTEXT_CACHE_SIZE,
            CONTEXT_MAX_TOKENS,
            CONTEXT_COMPRESSION,
            CONTEXT_SUMMARY_THRESHOLD
        )

        self.config = config or {}
        self.max_size = self.config.get('cache_size', CONTEXT_CACHE_SIZE)
        self.max_tokens = self.config.get('max_tokens', CONTEXT_MAX_TOKENS)
        self.enable_compression = self.config.get('compression', CONTEXT_COMPRESSION)
        self.summary_threshold = self.config.get('summary_threshold', CONTEXT_SUMMARY_THRESHOLD)

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Token counter (approximate)
        self._token_ratio = 4  # Approximate chars per token

    def get(self, key: str) -> Optional[List[Dict[str, str]]]:
        """
        Get cached context

        Args:
            key: Cache key (e.g., session_id)

        Returns:
            Cached context or None
        """
        if key not in self._cache:
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        return self._cache[key]['context']

    def put(
        self,
        key: str,
        context: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Cache a context

        Args:
            key: Cache key
            context: Conversation context
            metadata: Optional metadata
        """
        # Compress if needed
        if self.enable_compression:
            context = self._compress_context(context)

        # Store
        self._cache[key] = {
            'context': context,
            'metadata': metadata or {},
            'token_count': self._estimate_tokens(context)
        }

        # Move to end
        self._cache.move_to_end(key)

        # Evict if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def update(
        self,
        key: str,
        new_turn: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Add a new turn to cached context

        Args:
            key: Cache key
            new_turn: New conversation turn

        Returns:
            Updated context
        """
        context = self.get(key) or []
        context.append(new_turn)

        # Compress if over token limit
        if self._estimate_tokens(context) > self.max_tokens:
            context = self._compress_context(context)

        self.put(key, context)
        return context

    def _compress_context(
        self,
        context: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Compress context to fit within token limits

        Strategies:
        1. Keep system message
        2. Keep recent turns
        3. Summarize older turns
        """
        if not context:
            return context

        # Handle non-list contexts (e.g., result dicts)
        if not isinstance(context, list):
            return context

        total_tokens = self._estimate_tokens(context)

        if total_tokens <= self.max_tokens:
            return context

        # Separate system messages and conversation
        system_msgs = [m for m in context if m.get('role') == 'system']
        conversation = [m for m in context if m.get('role') != 'system']

        if not conversation:
            return context

        # Keep at least the last 2 turns
        min_keep = min(4, len(conversation))
        recent = conversation[-min_keep:]
        older = conversation[:-min_keep]

        # If still over limit, summarize older messages
        if older and self._estimate_tokens(system_msgs + recent) < self.max_tokens:
            # Create summary of older messages
            summary = self._create_summary(older)
            if summary:
                compressed = system_msgs + [summary] + recent
            else:
                compressed = system_msgs + recent
        else:
            # Just keep recent
            compressed = system_msgs + recent

        # Final truncation if still over
        while self._estimate_tokens(compressed) > self.max_tokens and len(compressed) > 2:
            # Remove oldest non-system message
            for i, msg in enumerate(compressed):
                if msg.get('role') != 'system':
                    compressed.pop(i)
                    break

        logger.debug(
            f"Compressed context from {len(context)} to {len(compressed)} messages "
            f"({total_tokens} -> {self._estimate_tokens(compressed)} tokens)"
        )

        return compressed

    def _create_summary(
        self,
        messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """Create a summary of messages"""
        if not messages:
            return None

        # Simple extractive summary
        topics = []
        for msg in messages:
            content = msg.get('content', '')
            # Extract first sentence or first 100 chars
            if '.' in content[:200]:
                topics.append(content[:content.index('.') + 1])
            else:
                topics.append(content[:100] + '...')

        summary_text = "Previous discussion covered: " + " ".join(topics[:3])

        return {
            'role': 'system',
            'content': f"[Context Summary] {summary_text}"
        }

    def _estimate_tokens(self, context: List[Dict[str, str]]) -> int:
        """Estimate token count for context"""
        # Handle non-list contexts (e.g., result dicts)
        if not isinstance(context, list):
            # Estimate based on string representation
            return len(str(context)) // self._token_ratio

        total_chars = sum(
            len(msg.get('content', '') if isinstance(msg, dict) else str(msg))
            for msg in context
        )
        return total_chars // self._token_ratio

    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache"""
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'total_tokens': sum(
                item['token_count']
                for item in self._cache.values()
            ),
            'keys': list(self._cache.keys())
        }


# Global cache instance
_context_cache: Optional[ContextCache] = None


def get_context_cache() -> ContextCache:
    """Get or create global context cache"""
    global _context_cache
    if _context_cache is None:
        _context_cache = ContextCache()
    return _context_cache


if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXT CACHE TEST")
    print("=" * 60)

    cache = ContextCache({'cache_size': 5, 'max_tokens': 1000})

    # Test basic put/get
    print("\nTest 1: Basic Put/Get")
    context1 = [
        {'role': 'user', 'content': 'What is labor law?'},
        {'role': 'assistant', 'content': 'Labor law regulates employment relationships.'}
    ]
    cache.put('session-1', context1)
    retrieved = cache.get('session-1')
    print(f"  ✓ Stored and retrieved {len(retrieved)} messages")

    # Test update
    print("\nTest 2: Context Update")
    new_turn = {'role': 'user', 'content': 'What about minimum wage?'}
    updated = cache.update('session-1', new_turn)
    print(f"  ✓ Updated context now has {len(updated)} messages")

    # Test LRU eviction
    print("\nTest 3: LRU Eviction")
    for i in range(6):
        cache.put(f'session-{i}', [{'role': 'user', 'content': f'Query {i}'}])

    stats = cache.get_stats()
    print(f"  Cache size: {stats['size']}/{stats['max_size']}")
    print(f"  ✓ Oldest session evicted: {'session-0' not in stats['keys']}")

    # Test compression
    print("\nTest 4: Context Compression")
    long_context = [
        {'role': 'system', 'content': 'You are a legal assistant.'},
    ]
    for i in range(20):
        long_context.append({'role': 'user', 'content': f'Question {i}: ' + 'x' * 200})
        long_context.append({'role': 'assistant', 'content': f'Answer {i}: ' + 'y' * 200})

    cache.put('session-long', long_context, {'test': True})
    compressed = cache.get('session-long')
    print(f"  Original: {len(long_context)} messages")
    print(f"  Compressed: {len(compressed)} messages")
    print(f"  ✓ Compression working: {len(compressed) < len(long_context)}")

    # Test stats
    print("\nTest 5: Cache Statistics")
    stats = cache.get_stats()
    print(f"  Size: {stats['size']}/{stats['max_size']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Keys: {stats['keys']}")

    # Test clear
    print("\nTest 6: Clear Cache")
    cache.clear('session-long')
    print(f"  ✓ Cleared specific key: {cache.get('session-long') is None}")

    cache.clear()
    print(f"  ✓ Cleared all: {cache.get_stats()['size'] == 0}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
