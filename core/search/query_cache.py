"""
Query Result Cache - LRU Cache for Search Results

Provides in-memory caching of search results to avoid redundant computations.

Features:
- LRU (Least Recently Used) eviction policy
- Configurable cache size
- Thread-safe operations
- Cache statistics (hits, misses, hit rate)
- TTL (Time To Live) support for cache entries

Performance impact:
- Cache hit: ~0.1ms (instant return)
- Cache miss: Full search pipeline (~10-1000ms depending on dataset size)
- Memory usage: ~1KB per cached query (with 50 results)
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from threading import Lock
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class QueryResultCache:
    """
    LRU cache for search query results

    Uses OrderedDict for O(1) access and O(1) eviction of least recently used items.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = 3600,
        enabled: bool = True
    ):
        """
        Initialize query result cache

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries in seconds (None = no expiration)
            enabled: Whether caching is enabled
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

        # Cache storage: {query_hash: (result, timestamp)}
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

        # Thread lock for concurrent access
        self._lock = Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(f"QueryResultCache initialized", {
            "max_size": max_size,
            "ttl_seconds": ttl_seconds,
            "enabled": enabled
        })

    def _hash_query(self, query: str, **kwargs) -> str:
        """
        Create a hash key for the query and its parameters

        Args:
            query: Search query string
            **kwargs: Additional search parameters (top_k, persona, etc.)

        Returns:
            SHA256 hash of the query and parameters
        """
        # Combine query with sorted kwargs for consistent hashing
        key_parts = [query]
        for k in sorted(kwargs.keys()):
            key_parts.append(f"{k}={kwargs[k]}")

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[Any]:
        """
        Get cached result for query

        Args:
            query: Search query
            **kwargs: Search parameters

        Returns:
            Cached result if found and not expired, None otherwise
        """
        if not self.enabled:
            return None

        query_hash = self._hash_query(query, **kwargs)

        with self._lock:
            if query_hash not in self._cache:
                self._misses += 1
                return None

            result, timestamp = self._cache[query_hash]

            # Check if expired
            if self.ttl_seconds is not None:
                age = time.time() - timestamp
                if age > self.ttl_seconds:
                    # Expired - remove from cache
                    del self._cache[query_hash]
                    self._misses += 1
                    logger.debug(f"Cache entry expired", {"query_hash": query_hash[:8], "age_sec": age})
                    return None

            # Move to end (most recently used)
            self._cache.move_to_end(query_hash)

            self._hits += 1
            logger.debug(f"Cache hit", {
                "query_hash": query_hash[:8],
                "hit_rate": f"{self.get_hit_rate():.1%}"
            })

            return result

    def put(self, query: str, result: Any, **kwargs):
        """
        Store result in cache

        Args:
            query: Search query
            result: Search result to cache
            **kwargs: Search parameters
        """
        if not self.enabled:
            return

        query_hash = self._hash_query(query, **kwargs)

        with self._lock:
            # Remove oldest entry if at capacity
            if len(self._cache) >= self.max_size and query_hash not in self._cache:
                # Remove least recently used (first item)
                evicted_hash, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"Cache eviction", {"evicted_hash": evicted_hash[:8], "cache_size": len(self._cache)})

            # Store result with timestamp
            self._cache[query_hash] = (result, time.time())

            # Move to end (most recently used)
            if query_hash in self._cache:
                self._cache.move_to_end(query_hash)

            logger.debug(f"Cache put", {
                "query_hash": query_hash[:8],
                "cache_size": len(self._cache)
            })

    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "enabled": self.enabled,
                "max_size": self.max_size,
                "current_size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds
            }

    def get_hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def reset_stats(self):
        """Reset cache statistics"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.info("Cache statistics reset")

    def set_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self.enabled = enabled
        logger.info(f"Cache {'enabled' if enabled else 'disabled'}")

    def __len__(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"QueryResultCache(size={stats['current_size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"hits={stats['hits']}, misses={stats['misses']})"
        )
