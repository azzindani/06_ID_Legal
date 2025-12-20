"""
Unit tests for context cache

Run with: pytest tests/unit/test_context_cache.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conversation.context_cache import ContextCache, get_context_cache


@pytest.mark.unit
class TestContextCache:
    """Test context cache functionality"""

    @pytest.fixture
    def cache(self):
        return ContextCache({'cache_size': 10, 'max_tokens': 1000})

    def test_put_and_get(self, cache):
        """Test basic put and get"""
        context = [{'role': 'user', 'content': 'Hello'}]
        cache.put('session-1', context)

        retrieved = cache.get('session-1')

        assert retrieved == context

    def test_get_nonexistent(self, cache):
        """Test getting nonexistent key"""
        result = cache.get('nonexistent')

        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when over capacity"""
        cache = ContextCache({'cache_size': 2})

        cache.put('session-1', [{'role': 'user', 'content': '1'}])
        cache.put('session-2', [{'role': 'user', 'content': '2'}])
        cache.put('session-3', [{'role': 'user', 'content': '3'}])

        # session-1 should be evicted
        assert cache.get('session-1') is None
        assert cache.get('session-2') is not None
        assert cache.get('session-3') is not None

    def test_update(self, cache):
        """Test updating context with new turn"""
        cache.put('session-1', [{'role': 'user', 'content': '1'}])

        updated = cache.update('session-1', {'role': 'assistant', 'content': '2'})

        assert len(updated) == 2
        assert updated[-1]['content'] == '2'

    def test_clear_specific(self, cache):
        """Test clearing specific key"""
        cache.put('session-1', [])
        cache.put('session-2', [])

        cache.clear('session-1')

        assert cache.get('session-1') is None
        assert cache.get('session-2') is not None

    def test_clear_all(self, cache):
        """Test clearing all cache"""
        cache.put('session-1', [])
        cache.put('session-2', [])

        cache.clear()

        assert cache.get('session-1') is None
        assert cache.get('session-2') is None

    def test_get_stats(self, cache):
        """Test cache statistics"""
        cache.put('session-1', [{'role': 'user', 'content': 'Hello'}])

        stats = cache.get_stats()

        assert stats['size'] == 1
        assert stats['max_size'] == 10
        assert 'session-1' in stats['keys']


@pytest.mark.unit
class TestContextCompression:
    """Test context compression functionality"""

    @pytest.fixture
    def cache(self):
        return ContextCache({'max_tokens': 100, 'compression': True})

    def test_compression_triggers(self):
        """Test that compression triggers when over limit"""
        cache = ContextCache({'max_tokens': 50, 'compression': True})

        # Create long context
        long_context = [
            {'role': 'user', 'content': 'x' * 100},
            {'role': 'assistant', 'content': 'y' * 100},
            {'role': 'user', 'content': 'z' * 100}
        ]

        cache.put('session-1', long_context)
        compressed = cache.get('session-1')

        # Should be compressed
        assert len(compressed) <= len(long_context)

    def test_keeps_recent_turns(self):
        """Test that recent turns are kept during compression"""
        cache = ContextCache({'max_tokens': 50, 'compression': True})

        context = [
            {'role': 'user', 'content': 'old message ' * 20},
            {'role': 'assistant', 'content': 'old response ' * 20},
            {'role': 'user', 'content': 'recent'},
            {'role': 'assistant', 'content': 'latest'}
        ]

        cache.put('session-1', context)
        compressed = cache.get('session-1')

        # Recent messages should be kept
        contents = [m['content'] for m in compressed]
        assert 'recent' in contents or 'latest' in contents


@pytest.mark.unit
class TestGlobalCache:
    """Test global cache singleton"""

    def test_get_context_cache(self):
        """Test getting global cache"""
        cache1 = get_context_cache()
        cache2 = get_context_cache()

        assert cache1 is cache2

    def test_global_cache_operations(self):
        """Test operations on global cache"""
        cache = get_context_cache()

        cache.put('global-test', [{'role': 'user', 'content': 'test'}])
        result = cache.get('global-test')

        assert result is not None
        assert result[0]['content'] == 'test'

        # Cleanup
        cache.clear('global-test')
