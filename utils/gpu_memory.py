"""
GPU Memory Management Utilities
Helps prevent OOM errors by detecting available memory and adjusting parameters

File: utils/gpu_memory.py
"""

import torch
from utils.logger_utils import get_logger


def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in GB.

    Returns:
        Available memory in GB, or 0 if no GPU available
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        # Get free memory on first GPU
        free_memory = torch.cuda.mem_get_info()[0]
        return free_memory / (1024 ** 3)  # Convert to GB
    except Exception:
        return 0.0


def get_recommended_max_new_tokens_for_memory(
    thinking_mode: str = 'low',
    default_max_tokens: int = 2048
) -> int:
    """
    DEPRECATED: This function no longer limits max_new_tokens.

    Modern GPUs can handle long contexts. If you experience OOM,
    manually adjust MAX_NEW_TOKENS environment variable.

    Args:
        thinking_mode: Thinking mode ('low', 'medium', 'high')
        default_max_tokens: Default max_new_tokens from config

    Returns:
        The default_max_tokens unchanged
    """
    logger = get_logger("GPUMemory")

    if not torch.cuda.is_available():
        logger.debug("No GPU available, using default max_new_tokens")
        return default_max_tokens

    # Just return the default - don't limit it
    # User can manually set MAX_NEW_TOKENS if needed
    return default_max_tokens


def check_memory_for_generation(
    prompt_length: int,
    max_new_tokens: int,
    thinking_mode: str = 'low'
) -> dict:
    """
    Check if there's likely enough memory for generation.

    Args:
        prompt_length: Length of prompt in characters
        max_new_tokens: Maximum tokens to generate
        thinking_mode: Thinking mode being used

    Returns:
        Dictionary with memory check results
    """
    logger = get_logger("GPUMemory")

    if not torch.cuda.is_available():
        return {
            'has_gpu': False,
            'sufficient_memory': True,  # CPU has plenty of memory usually
            'warning': None
        }

    try:
        available_gb = get_available_gpu_memory()

        # Rough estimation of memory needed
        # Prompt tokens: ~4 chars per token
        estimated_prompt_tokens = prompt_length / 4

        # Memory needed for KV cache and attention
        # Very rough estimate: ~100 MB per 1000 tokens of context
        estimated_memory_gb = (estimated_prompt_tokens + max_new_tokens) / 1000 * 0.1

        sufficient = available_gb >= estimated_memory_gb

        warning = None
        if not sufficient:
            warning = (
                f"WARNING: May run out of memory! "
                f"Available: {available_gb:.2f} GB, "
                f"Estimated need: {estimated_memory_gb:.2f} GB. "
                f"Consider reducing max_new_tokens or using lower thinking mode."
            )
            logger.warning(warning)

        return {
            'has_gpu': True,
            'available_gb': available_gb,
            'estimated_need_gb': estimated_memory_gb,
            'sufficient_memory': sufficient,
            'warning': warning
        }

    except Exception as e:
        logger.warning(f"Failed to check memory: {e}")
        return {
            'has_gpu': True,
            'sufficient_memory': True,  # Assume OK if can't check
            'warning': None
        }


def suggest_memory_optimizations(available_gb: float) -> list:
    """
    Suggest optimizations based on available memory.

    Args:
        available_gb: Available GPU memory in GB

    Returns:
        List of suggestion strings
    """
    suggestions = []

    if available_gb < 4.0:
        suggestions.append("Use --low thinking mode to reduce prompt length")
        suggestions.append("Set MAX_NEW_TOKENS=512 to reduce generation length")
        suggestions.append("Consider using CPU inference (slower but no memory limit)")

    elif available_gb < 6.0:
        suggestions.append("Use --low or --medium thinking mode")
        suggestions.append("Set MAX_NEW_TOKENS=1024 to reduce generation length")
        suggestions.append("Export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    elif available_gb < 8.0:
        suggestions.append("--medium thinking mode should work well")
        suggestions.append("--high thinking mode may require MAX_NEW_TOKENS=1024")

    else:
        suggestions.append("All thinking modes should work well")

    return suggestions
