"""
Loader package for KG-Enhanced Indonesian Legal RAG System
"""

# Lazy imports to avoid numpy dependency during testing
def __getattr__(name):
    if name == 'EnhancedKGDatasetLoader':
        from .dataloader import EnhancedKGDatasetLoader
        return EnhancedKGDatasetLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['EnhancedKGDatasetLoader']