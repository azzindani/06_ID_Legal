"""
Knowledge Graph Module

Provides entity extraction, relationship mapping, and graph-based scoring
for enhanced legal document retrieval.
"""

# Lazy imports to avoid dependency issues during testing
def __getattr__(name):
    if name == 'KnowledgeGraphCore':
        from .kg_core import KnowledgeGraphCore
        return KnowledgeGraphCore
    elif name == 'RelationshipGraph':
        from .relationship_graph import RelationshipGraph
        return RelationshipGraph
    elif name == 'CommunityDetector':
        from .community_detection import CommunityDetector
        return CommunityDetector
    elif name == 'EnhancedKnowledgeGraph':
        from .enhanced_kg import EnhancedKnowledgeGraph
        return EnhancedKnowledgeGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'KnowledgeGraphCore',
    'RelationshipGraph',
    'CommunityDetector',
    'EnhancedKnowledgeGraph',
]
