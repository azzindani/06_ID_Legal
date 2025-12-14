"""
Text Processing Utilities

This module contains utility functions for text parsing and processing,
particularly for parsing LLM responses with special tags.

File: utils/text_utils.py
"""

import re
from typing import Tuple


def parse_think_tags(text: str) -> Tuple[str, str]:
    """
    Extract content from <think> tags and return (thinking, answer)

    LLM responses may contain <think>...</think> tags for chain-of-thought reasoning.
    This function separates the thinking process from the final answer.

    Args:
        text: Raw LLM response text that may contain <think> tags

    Returns:
        Tuple of (thinking_content, answer_content)
        - thinking_content: Content inside <think> tags (empty if no tags)
        - answer_content: Text with <think> tags removed

    Example:
        >>> text = "Let me think. <think>This is reasoning</think>The answer is 42."
        >>> thinking, answer = parse_think_tags(text)
        >>> print(thinking)
        This is reasoning
        >>> print(answer)
        Let me think. The answer is 42.
    """
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)
    thinking = '\n\n'.join(matches) if matches else ''
    answer = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
    return thinking, answer


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix

    Args:
        text: Text to truncate
        max_length: Maximum length (default: 1000)
        suffix: Suffix to add when truncated (default: "...")

    Returns:
        Truncated text with suffix if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """
    Clean excessive whitespace from text

    Args:
        text: Text to clean

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple newlines with max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()
