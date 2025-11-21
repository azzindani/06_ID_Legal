# Search Module

Core search functionality for the Indonesian Legal RAG System.

## Components

| File | Description |
|------|-------------|
| `query_detection.py` | Query type classification and entity extraction |
| `hybrid_search.py` | Combined semantic and keyword search |
| `stages_research.py` | Multi-stage research with researcher personas |
| `consensus.py` | Consensus building from multiple researchers |
| `reranking.py` | Final result reranking |
| `langgraph_orchestrator.py` | LangGraph workflow orchestration |

## Usage

```python
from core.search import (
    QueryDetector,
    HybridSearchEngine,
    StagesResearch,
    ConsensusBuilder
)

# Detect query type
detector = QueryDetector()
query_info = detector.detect("Apa sanksi pelanggaran UU Ketenagakerjaan?")

# Perform hybrid search
search_engine = HybridSearchEngine(embeddings, dataset)
results = search_engine.search(query, semantic_weight=0.7)

# Multi-stage research
researcher = StagesResearch(config)
research_results = researcher.research(query, initial_results)

# Build consensus
builder = ConsensusBuilder()
final_results = builder.build(research_results)
```

## Query Types

- `definition` - What is X?
- `procedure` - How to do X?
- `requirement` - What are requirements for X?
- `sanctions` - What are penalties for X?
- `general` - Other questions

## Configuration

```python
config = {
    'semantic_weight': 0.7,
    'keyword_weight': 0.3,
    'max_results': 20,
    'consensus_threshold': 0.6,
    'quality_degradation': 0.1,
}
```
