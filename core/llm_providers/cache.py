"""
Response Cache - LRU Cache for LLM Responses

Caches LLM responses to avoid redundant API calls,
reducing costs and latency for repeated queries.

File: core/llm_providers/cache.py
"""

import hashlib
import time
from typing import Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, field

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("ResponseCache")
except ImportError:
    import logging
    logger = logging.getLogger("ResponseCache")


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    response: Dict[str, Any]
    created_at: float
    hits: int = 0
    provider: str = ""
    model: str = ""
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired"""
        return time.time() - self.created_at > ttl_seconds


class ResponseCache:
    """
    LRU cache for LLM responses.
    
    Features:
    - Hash-based key derivation from prompt
    - Configurable TTL (time-to-live)
    - LRU eviction when capacity exceeded
    - Hit/miss statistics
    - Provider/model-specific caching
    
    Usage:
        cache = ResponseCache(max_size=100, ttl_seconds=3600)
        
        # Check cache
        cached = cache.get(prompt, model="gpt-4")
        if cached:
            return cached
        
        # Generate and cache
        response = llm.generate(prompt)
        cache.set(prompt, response, model="gpt-4")
    """
    
    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,  # 1 hour default
        enabled: bool = True
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
            enabled: Whether caching is enabled
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        
        # OrderedDict for LRU ordering
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"Response cache initialized", {
            "max_size": max_size,
            "ttl_seconds": ttl_seconds,
            "enabled": enabled
        })
    
    def _generate_key(
        self,
        prompt: str,
        model: str = "",
        **kwargs
    ) -> str:
        """
        Generate cache key from prompt and parameters.
        
        Uses SHA256 hash for consistent key length.
        """
        # Include model and key parameters in hash
        key_parts = [prompt, model]
        
        # Add relevant generation parameters
        for param in ['max_new_tokens', 'temperature', 'top_p']:
            if param in kwargs:
                key_parts.append(f"{param}={kwargs[param]}")
        
        key_string = "||".join(str(p) for p in key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def get(
        self,
        prompt: str,
        model: str = "",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response.
        
        Args:
            prompt: The prompt to look up
            model: Model name for model-specific caching
            **kwargs: Additional parameters that affect the response
            
        Returns:
            Cached response dict or None
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(prompt, model, **kwargs)
        
        if key not in self._cache:
            self.misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if entry.is_expired(self.ttl_seconds):
            del self._cache[key]
            self.misses += 1
            return None
        
        # Update LRU order (move to end)
        self._cache.move_to_end(key)
        
        # Update stats
        entry.hits += 1
        self.hits += 1
        
        logger.debug(f"Cache hit", {"key": key[:8], "hits": entry.hits})
        
        # Return copy to prevent mutation
        return entry.response.copy()
    
    def set(
        self,
        prompt: str,
        response: Dict[str, Any],
        model: str = "",
        provider: str = "",
        **kwargs
    ):
        """
        Cache a response.
        
        Args:
            prompt: The prompt
            response: Response to cache
            model: Model name
            provider: Provider name
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return
        
        key = self._generate_key(prompt, model, **kwargs)
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            # Remove oldest (first) entry
            evicted_key, _ = self._cache.popitem(last=False)
            self.evictions += 1
            logger.debug(f"Evicted cache entry", {"key": evicted_key[:8]})
        
        # Store entry
        self._cache[key] = CacheEntry(
            response=response.copy(),
            created_at=time.time(),
            provider=provider,
            model=model
        )
        
        logger.debug(f"Cached response", {"key": key[:8], "size": len(self._cache)})
    
    def invalidate(
        self,
        prompt: str = None,
        model: str = None,
        **kwargs
    ):
        """
        Invalidate cache entries.
        
        Args:
            prompt: Specific prompt to invalidate (or all if None)
            model: Filter by model (invalidate all for this model)
        """
        if prompt:
            key = self._generate_key(prompt, model or "", **kwargs)
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Invalidated specific entry", {"key": key[:8]})
        elif model:
            # Invalidate all entries for this model
            keys_to_remove = [
                k for k, v in self._cache.items()
                if v.model == model
            ]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Invalidated {len(keys_to_remove)} entries for model {model}")
        else:
            # Clear all
            self.clear()
    
    def clear(self):
        """Clear all cache entries"""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared", {"entries_removed": count})
    
    def cleanup_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired = [
            k for k, v in self._cache.items()
            if v.is_expired(self.ttl_seconds)
        ]
        
        for key in expired:
            del self._cache[key]
        
        if expired:
            logger.debug(f"Cleaned up expired entries", {"count": len(expired)})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{hit_rate:.2%}",
            "ttl_seconds": self.ttl_seconds,
        }
    
    def set_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self.enabled = enabled
        logger.info(f"Cache {'enabled' if enabled else 'disabled'}")


# Global cache instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache(
    max_size: int = 100,
    ttl_seconds: int = 3600
) -> ResponseCache:
    """Get global response cache instance"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds
        )
    return _response_cache
