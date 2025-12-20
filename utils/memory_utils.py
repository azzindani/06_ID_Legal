"""
Memory Management Utilities for Indonesian Legal RAG System

Provides standardized memory cleanup functions for:
- CPU memory (Python garbage collection)
- GPU memory (CUDA cache)
- Cache clearing
- Memory profiling

Usage:
    from utils.memory_utils import cleanup_memory, aggressive_cleanup

    # Standard cleanup (after search, before generation)
    cleanup_memory(aggressive=False)

    # Aggressive cleanup (before large operations)
    aggressive_cleanup()
"""

import gc
from typing import Dict, Any, Optional
from utils.logger_utils import get_logger

# Optional torch support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger("MemoryUtils")


def cleanup_memory(aggressive: bool = False, reason: str = "memory cleanup") -> Dict[str, Any]:
    """
    Standard memory cleanup for CPU and GPU

    Args:
        aggressive: If True, performs more thorough cleanup (slower but more effective)
        reason: Reason for cleanup (for logging)

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        'cpu_collected': 0,
        'gpu_freed_mb': 0.0,
        'gpu_cached_freed_mb': 0.0
    }

    # CPU cleanup
    stats['cpu_collected'] = gc.collect()

    if aggressive:
        # Additional CPU cleanup passes
        stats['cpu_collected'] += gc.collect()
        stats['cpu_collected'] += gc.collect()

    # GPU cleanup
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            # Get memory before cleanup
            allocated_before = torch.cuda.memory_allocated() / 1024**2  # MB
            cached_before = torch.cuda.memory_reserved() / 1024**2  # MB

            # Clear CUDA cache
            torch.cuda.empty_cache()

            if aggressive:
                # Synchronize all CUDA operations
                torch.cuda.synchronize()
                # Additional cache clear
                torch.cuda.empty_cache()

            # Get memory after cleanup
            allocated_after = torch.cuda.memory_allocated() / 1024**2  # MB
            cached_after = torch.cuda.memory_reserved() / 1024**2  # MB

            stats['gpu_freed_mb'] = allocated_before - allocated_after
            stats['gpu_cached_freed_mb'] = cached_before - cached_after

            if stats['gpu_freed_mb'] > 0 or stats['gpu_cached_freed_mb'] > 0:
                logger.debug(f"GPU cleanup ({reason}): Freed {stats['gpu_freed_mb']:.1f}MB allocated, "
                           f"{stats['gpu_cached_freed_mb']:.1f}MB cached")

        except Exception as e:
            logger.debug(f"GPU cleanup warning: {e}")

    logger.debug(f"Memory cleanup completed ({reason}): CPU collected {stats['cpu_collected']} objects, "
               f"GPU freed {stats['gpu_freed_mb']:.1f}MB")

    return stats


def aggressive_cleanup(reason: str = "aggressive cleanup") -> Dict[str, Any]:
    """
    Aggressive memory cleanup before large operations

    Performs multiple cleanup passes and CUDA synchronization.
    Use before:
    - LLM generation
    - Large batch processing
    - Memory-intensive operations

    Args:
        reason: Reason for cleanup (for logging)

    Returns:
        Dictionary with cleanup statistics
    """
    logger.info(f"Performing aggressive memory cleanup: {reason}")
    return cleanup_memory(aggressive=True, reason=reason)


def cleanup_before_generation(context: str = "generation") -> None:
    """
    Standard cleanup before LLM generation

    Ensures maximum available memory for model inference.
    Call this right before model.generate() to prevent OOM.

    Args:
        context: Context string for logging
    """
    stats = aggressive_cleanup(reason=f"before {context}")

    # Log memory state
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2  # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB

            logger.info(f"Pre-generation GPU state: {allocated:.1f}MB allocated, "
                       f"{cached:.1f}MB cached, {total:.1f}MB total")
        except Exception as e:
            logger.debug(f"GPU memory check warning: {e}")


def cleanup_after_search(num_results: int = 0) -> None:
    """
    Standard cleanup after search/retrieval operations

    Call this after expansion and before generation to free
    memory from embeddings, similarity calculations, etc.

    Args:
        num_results: Number of results retrieved (for logging)
    """
    cleanup_memory(aggressive=False, reason=f"after search ({num_results} results)")


def get_memory_stats() -> Dict[str, Any]:
    """
    Get current memory statistics

    Returns:
        Dictionary with CPU and GPU memory stats
    """
    stats = {
        'cpu': {},
        'gpu': {}
    }

    # GPU stats
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            stats['gpu']['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu']['cached_mb'] = torch.cuda.memory_reserved() / 1024**2
            stats['gpu']['max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            stats['gpu']['total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            stats['gpu']['available_mb'] = stats['gpu']['total_mb'] - stats['gpu']['allocated_mb']
        except Exception as e:
            logger.debug(f"GPU stats error: {e}")
            stats['gpu']['error'] = str(e)

    return stats


def log_memory_state(prefix: str = "Memory") -> None:
    """
    Log current memory state

    Args:
        prefix: Prefix for log message
    """
    stats = get_memory_stats()

    if 'allocated_mb' in stats['gpu']:
        logger.info(f"{prefix}: GPU {stats['gpu']['allocated_mb']:.1f}MB allocated, "
                   f"{stats['gpu']['cached_mb']:.1f}MB cached, "
                   f"{stats['gpu']['available_mb']:.1f}MB available")
    else:
        logger.debug(f"{prefix}: GPU not available")


def clear_cache() -> None:
    """
    Clear all caches (KV cache, embedding cache, etc.)

    Use this when switching between different queries/sessions
    to prevent cache pollution.
    """
    # Clear Python garbage
    gc.collect()

    # Clear CUDA cache
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.debug("All caches cleared")


def reset_peak_memory() -> None:
    """
    Reset peak memory tracking

    Useful for benchmarking individual operations.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.debug("Peak memory stats reset")


# Convenience functions for common scenarios

def prepare_for_llm() -> None:
    """Prepare memory for LLM inference"""
    cleanup_before_generation("LLM inference")


def cleanup_after_retrieval(num_docs: int) -> None:
    """Cleanup after document retrieval"""
    cleanup_after_search(num_docs)


def cleanup_after_expansion(pool_size: int) -> None:
    """Cleanup after document expansion"""
    cleanup_memory(aggressive=True, reason=f"after expansion ({pool_size} docs)")
