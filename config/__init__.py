"""
Configuration package for Indonesian Legal RAG System
Re-exports all config from root config.py plus vocabulary
"""

import sys
from pathlib import Path

# Add parent directory to path to import root config.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import everything from root config.py
from config import *

# Remove parent from path
sys.path.pop(0)

# Also make legal vocabulary easily accessible
from .legal_vocab import INDONESIAN_LEGAL_SYNONYMS, LEGAL_DOMAINS

# Export everything
__all__ = [
    'INDONESIAN_LEGAL_SYNONYMS',
    'LEGAL_DOMAINS',
]
