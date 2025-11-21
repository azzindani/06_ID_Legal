"""
Knowledge Graph Module

Provides entity extraction, relationship mapping, and graph-based scoring
for enhanced legal document retrieval.
"""

from .kg_core import KnowledgeGraphCore
from .relationship_graph import RelationshipGraph
from .community_detection import CommunityDetector

__all__ = [
    'KnowledgeGraphCore',
    'RelationshipGraph',
    'CommunityDetector',
]
