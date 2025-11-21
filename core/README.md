# Core Module

Core RAG components including search, generation, and knowledge graph.

## Structure

```
core/
├── __init__.py              # All exports
├── search/                  # Search components
│   ├── query_detection.py   # Query classification
│   ├── hybrid_search.py     # Semantic + keyword
│   ├── stages_research.py   # Multi-stage research
│   ├── consensus.py         # Consensus building
│   ├── reranking.py         # Result reranking
│   └── langgraph_orchestrator.py
├── generation/              # LLM generation
│   ├── llm_engine.py        # Model loading
│   ├── generation_engine.py # Generation pipeline
│   ├── prompt_builder.py    # Prompt construction
│   ├── citation_formatter.py
│   └── response_validator.py
└── knowledge_graph/         # KG components
    ├── kg_core.py           # Entity extraction
    ├── relationship_graph.py # Document network
    └── community_detection.py
```

## Quick Usage

```python
from core import (
    QueryDetector,
    HybridSearchEngine,
    ConsensusBuilder,
    GenerationEngine,
    KnowledgeGraphCore
)

# Query detection
detector = QueryDetector()
query_info = detector.detect("Apa sanksi pelanggaran?")
print(f"Type: {query_info['query_type']}")

# Knowledge graph
kg = KnowledgeGraphCore()
entities = kg.extract_entities("UU No. 13 Tahun 2003")
```

## Submodules

- **[search/](search/README.md)** - Search and retrieval
- **[generation/](generation/README.md)** - LLM generation
- **[knowledge_graph/](knowledge_graph/README.md)** - Graph analysis
