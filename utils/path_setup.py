"""
Centralized Path Import Utility

This module sets up the Python path to enable imports from the project root.
Import this module at the top of scripts that need access to project modules.

Usage:
    import utils.path_setup  # This adds project root to sys.path
    
    # Now you can import project modules
    from pipeline import RAGPipeline
    from config import DEFAULT_CONFIG
"""

import sys
import os
from pathlib import Path

# Get project root (parent of utils directory)
_PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Add to path if not already present
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Export for convenience
PROJECT_ROOT = _PROJECT_ROOT


def ensure_project_path():
    """
    Ensure project root is in sys.path.
    Call this function if you need to explicitly add the path.
    
    Returns:
        Path: The project root path
    """
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    return _PROJECT_ROOT


def get_project_root() -> Path:
    """
    Get the absolute path to the project root.
    
    Returns:
        Path: The project root path
    """
    return _PROJECT_ROOT
