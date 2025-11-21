# Knowledge Graph Module

Graph-based document analysis and scoring.

## Components

| File | Description |
|------|-------------|
| `kg_core.py` | Entity extraction and scoring |
| `relationship_graph.py` | Document relationship network |
| `community_detection.py` | Document clustering |

## Usage

```python
from core.knowledge_graph import (
    KnowledgeGraphCore,
    RelationshipGraph,
    CommunityDetector
)

# Extract entities
kg = KnowledgeGraphCore()
entities = kg.extract_entities("UU No. 13 Tahun 2003 Pasal 1")

# Enhance search results
enhanced = kg.enhance_results(results, query, kg_weight=0.3)

# Build relationship graph
graph = RelationshipGraph()
graph.build_from_documents(documents, kg)

# Find related documents
related = graph.get_related_documents('doc-123')

# Detect communities
detector = CommunityDetector()
communities = detector.detect_communities(graph)
```

## Entity Types

- `regulation` - UU, PP, Perpres, etc.
- `article` - Pasal references
- `institution` - Government bodies
- `legal_term` - Legal terminology

## Graph Features

- PageRank-based relevance
- Community detection (Louvain)
- Centrality analysis
- Inter-community bridging
