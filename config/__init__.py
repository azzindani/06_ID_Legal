"""
Configuration package for Indonesian Legal RAG System
Contains vocabulary, patterns, and configuration settings
"""

# Make legal vocabulary easily accessible
from .legal_vocab import INDONESIAN_LEGAL_SYNONYMS, LEGAL_DOMAINS

__all__ = [
    'INDONESIAN_LEGAL_SYNONYMS',
    'LEGAL_DOMAINS',
]
