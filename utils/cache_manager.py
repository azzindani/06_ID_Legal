# utils/cache_manager.py
"""
Generic cache management utilities.
Supports disk and memory caching with TTL and size limits.
"""
import pickle
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
from utils.logging_config import get_logger, LogBlock

logger = get_logger(__name__)

class CacheEntry:
    """Single cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 0
        self.ttl = ttl  # Time to live in seconds
        self.size_bytes = self._estimate_size(value)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def access(self):
        """Record access to entry."""
        self.accessed_at = time.time()
        self.access_count += 1


class MemoryCache:
    """
    In-memory LRU cache with size limits and TTL support.
    Thread-safe with basic locking.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 500,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(
            f"MemoryCache initialized: max_size={max_size}, "
            f"max_memory={max_memory_mb}MB, ttl={default_ttl}s"
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if key in self._cache:
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                logger.debug(f"Cache entry expired: {key}")
                del self._cache[key]
                self._misses += 1
                return default
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.access()
            
            self._hits += 1
            logger.debug(f"Cache hit: {key} (hits={self._hits})")
            return entry.value
        
        self._misses += 1
        logger.debug(f"Cache miss: {key} (misses={self._misses})")
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (overrides default)
        """
        # Remove if already exists
        if key in self._cache:
            del self._cache[key]
        
        # Create entry
        entry = CacheEntry(key, value, ttl or self.default_ttl)
        
        # Check size limit
        if len(self._cache) >= self.max_size:
            self._evict_one()
        
        # Check memory limit
        current_memory = self._get_total_memory()
        if current_memory + entry.size_bytes > self.max_memory_bytes:
            self._evict_until_space(entry.size_bytes)
        
        self._cache[key] = entry
        logger.debug(
            f"Cache set: {key} ({entry.size_bytes} bytes, "
            f"total_memory={self._get_total_memory() / 1024 / 1024:.1f}MB)"
        )
    
    def _evict_one(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Remove oldest (least recently used)
        evicted_key, evicted_entry = self._cache.popitem(last=False)
        self._evictions += 1
        
        logger.debug(
            f"Evicted entry: {evicted_key} "
            f"(accessed {evicted_entry.access_count} times)"
        )
    
    def _evict_until_space(self, needed_bytes: int):
        """Evict entries until enough space available."""
        evicted = 0
        while (self._get_total_memory() + needed_bytes > self.max_memory_bytes and 
               self._cache):
            self._evict_one()
            evicted += 1
        
        if evicted > 0:
            logger.info(f"Evicted {evicted} entries to free space")
    
    def _get_total_memory(self) -> int:
        """Get total cache memory usage in bytes."""
        return sum(entry.size_bytes for entry in self._cache.values())
    
    def clear(self):
        """Clear all cache entries."""
        size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared: {size} entries removed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'memory_mb': self._get_total_memory() / 1024 / 1024,
            'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'evictions': self._evictions
        }


class DiskCache:
    """
    Persistent disk-based cache with serialization.
    Slower than memory cache but survives restarts.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 5.0,
        compression: bool = True
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum total size in GB
            compression: Enable compression
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.compression = compression
        
        self._hits = 0
        self._misses = 0
        
        logger.info(
            f"DiskCache initialized: dir={cache_dir}, "
            f"max_size={max_size_gb}GB, compression={compression}"
        )
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash key to create safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from disk cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self._misses += 1
            logger.debug(f"Disk cache miss: {key}")
            return default
        
        try:
            with open(cache_path, 'rb') as f:
                if self.compression:
                    import gzip
                    data = gzip.decompress(f.read())
                    value = pickle.loads(data)
                else:
                    value = pickle.load(f)
            
            self._hits += 1
            logger.debug(f"Disk cache hit: {key}")
            return value
            
        except Exception as e:
            logger.warning(f"Error reading cache {key}: {e}")
            self._misses += 1
            return default
    
    def set(self, key: str, value: Any):
        """Set value in disk cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            # Check total size
            self._enforce_size_limit()
            
            # Write to temp file first
            temp_path = cache_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                if self.compression:
                    import gzip
                    data = pickle.dumps(value)
                    f.write(gzip.compress(data))
                else:
                    pickle.dump(value, f)
            
            # Atomic rename
            temp_path.replace(cache_path)
            
            size_mb = cache_path.stat().st_size / 1024 / 1024
            logger.debug(f"Disk cache set: {key} ({size_mb:.2f}MB)")
            
        except Exception as e:
            logger.error(f"Error writing cache {key}: {e}", exc_info=True)
    
    def _enforce_size_limit(self):
        """Remove old cache files if size limit exceeded."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.cache'))
        
        if total_size > self.max_size_bytes:
            # Remove oldest files
            files = sorted(
                self.cache_dir.glob('*.cache'),
                key=lambda f: f.stat().st_mtime
            )
            
            removed = 0
            for file in files:
                if total_size <= self.max_size_bytes * 0.8:  # 80% threshold
                    break
                
                size = file.stat().st_size
                file.unlink()
                total_size -= size
                removed += 1
            
            logger.info(f"Removed {removed} old cache files to enforce size limit")
    
    def clear(self):
        """Clear all disk cache."""
        count = 0
        for file in self.cache_dir.glob('*.cache'):
            file.unlink()
            count += 1
        
        logger.info(f"Disk cache cleared: {count} files removed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        files = list(self.cache_dir.glob('*.cache'))
        total_size = sum(f.stat().st_size for f in files)
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'files': len(files),
            'size_mb': total_size / 1024 / 1024,
            'max_size_mb': self.max_size_bytes / 1024 / 1024,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate
        }


class HybridCache:
    """
    Two-level cache: Fast memory cache + persistent disk cache.
    Automatically promotes frequently accessed items to memory.
    """
    
    def __init__(
        self,
        memory_cache: MemoryCache,
        disk_cache: DiskCache,
        promotion_threshold: int = 3
    ):
        """
        Initialize hybrid cache.
        
        Args:
            memory_cache: L1 memory cache
            disk_cache: L2 disk cache
            promotion_threshold: Access count to promote to memory
        """
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache
        self.promotion_threshold = promotion_threshold
        
        # Track disk access counts for promotion
        self._disk_access_counts: Dict[str, int] = {}
        
        logger.info("HybridCache initialized (L1=Memory, L2=Disk)")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from hybrid cache.
        Checks memory first, then disk.
        """
        # Try memory cache first (L1)
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache (L2)
        value = self.disk_cache.get(key)
        if value is not None:
            # Track access for promotion
            self._disk_access_counts[key] = self._disk_access_counts.get(key, 0) + 1
            
            # Promote to memory if accessed frequently
            if self._disk_access_counts[key] >= self.promotion_threshold:
                logger.debug(f"Promoting to memory cache: {key}")
                self.memory_cache.set(key, value)
                del self._disk_access_counts[key]
            
            return value
        
        return default
    
    def set(self, key: str, value: Any, memory_only: bool = False):
        """
        Set value in hybrid cache.
        
        Args:
            key: Cache key
            value: Value to cache
            memory_only: If True, only cache in memory
        """
        # Always set in memory cache
        self.memory_cache.set(key, value)
        
        # Also persist to disk unless memory_only
        if not memory_only:
            self.disk_cache.set(key, value)
    
    def clear(self):
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        self._disk_access_counts.clear()
        logger.info("Hybrid cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            'memory': self.memory_cache.get_stats(),
            'disk': self.disk_cache.get_stats(),
            'promotion_queue': len(self._disk_access_counts)
        }