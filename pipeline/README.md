# Pipeline Module

High-level RAG Pipeline API that orchestrates the complete retrieval-augmented generation workflow.

## Purpose

The Pipeline module provides a simplified interface for executing complete RAG queries without needing to manually coordinate individual components. It handles:

- Model initialization (embedding, reranker, LLM)
- Dataset loading
- Query orchestration
- Response generation
- Resource cleanup

## Components

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `rag_pipeline.py` | Main RAGPipeline class |
| `tests/test_rag_pipeline.py` | Unit and integration tests |

## Usage

### Basic Usage

```python
from pipeline import RAGPipeline

# Create and initialize pipeline
pipeline = RAGPipeline()
pipeline.initialize()

# Execute query
result = pipeline.query("Apa sanksi pelanggaran UU Ketenagakerjaan?")

if result['success']:
    print(result['answer'])
    print(f"Sources: {len(result['sources'])}")
    print(f"Time: {result['metadata']['total_time']:.2f}s")
else:
    print(f"Error: {result['error']}")

# Cleanup
pipeline.shutdown()
```

### Context Manager

```python
from pipeline import RAGPipeline

with RAGPipeline() as pipeline:
    result = pipeline.query("Bagaimana prosedur pendirian PT?")
    print(result['answer'])
# Automatic cleanup on exit
```

### Custom Configuration

```python
from pipeline import RAGPipeline

config = {
    'final_top_k': 5,
    'temperature': 0.5,
    'max_new_tokens': 4096,
    'consensus_threshold': 0.7
}

pipeline = RAGPipeline(config)
pipeline.initialize()
```

### With Conversation History

```python
history = [
    {'role': 'user', 'content': 'Apa itu UU Ketenagakerjaan?'},
    {'role': 'assistant', 'content': 'UU Ketenagakerjaan adalah...'}
]

result = pipeline.query(
    "Apa sanksinya?",
    conversation_history=history
)
```

### Streaming Response

```python
for chunk in pipeline.query("Jelaskan prosedur PHK", stream=True):
    if chunk['type'] == 'token':
        print(chunk['token'], end='', flush=True)
    elif chunk['type'] == 'complete':
        print(f"\n\nTotal time: {chunk['metadata']['total_time']:.2f}s")
```

### Progress Callback

```python
def on_progress(step_name, current, total):
    print(f"[{current}/{total}] {step_name}")

pipeline.initialize(progress_callback=on_progress)
```

## API Reference

### RAGPipeline

#### Constructor

```python
RAGPipeline(config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config`: Optional configuration dictionary. Merges with defaults.

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `initialize(progress_callback)` | Load all components | `bool` |
| `query(question, history, stream)` | Execute RAG query | `Dict` or `Generator` |
| `get_pipeline_info()` | Get pipeline status | `Dict` |
| `update_config(**kwargs)` | Update configuration | `None` |
| `shutdown()` | Cleanup resources | `None` |

#### Query Result Structure

```python
{
    'success': bool,
    'answer': str,              # Generated answer
    'sources': [                # Source documents
        {
            'id': int,
            'title': str,
            'type': str,
            'year': str,
            'relevance_score': float,
            'excerpt': str
        }
    ],
    'citations': [              # Citation details
        {
            'id': int,
            'regulation_type': str,
            'regulation_number': str,
            'year': str,
            'article': str,
            'citation_text': str
        }
    ],
    'metadata': {
        'question': str,
        'retrieval_time': float,
        'generation_time': float,
        'total_time': float,
        'results_count': int,
        'tokens_generated': int,
        'query_type': str
    },
    'error': str                # Only if success=False
}
```

## Configuration Options

| Key | Default | Description |
|-----|---------|-------------|
| `final_top_k` | 3 | Number of top results to use |
| `max_rounds` | 5 | Maximum search rounds |
| `initial_quality` | 0.95 | Starting quality threshold |
| `quality_degradation` | 0.1 | Quality decrease per round |
| `min_quality` | 0.5 | Minimum quality threshold |
| `consensus_threshold` | 0.6 | Consensus agreement threshold |
| `temperature` | 0.7 | LLM temperature |
| `max_new_tokens` | 2048 | Max generation tokens |
| `top_p` | 1.0 | Nucleus sampling |
| `top_k` | 20 | Top-k sampling |

## Testing

### Run Tests

```bash
# Run all pipeline tests
pytest pipeline/tests/ -v

# Run with coverage
pytest pipeline/tests/ --cov=pipeline --cov-report=html

# Run specific test
pytest pipeline/tests/test_rag_pipeline.py::test_query_basic -v
```

### Test Categories

| Test | Description |
|------|-------------|
| `test_pipeline_creation` | Pipeline instantiation |
| `test_initialization` | Component loading |
| `test_query_basic` | Basic query execution |
| `test_query_with_history` | Conversation context |
| `test_streaming` | Streaming response |
| `test_config_update` | Configuration changes |
| `test_shutdown` | Resource cleanup |
| `test_context_manager` | With statement |

## Performance Notes

### Initialization Time

Typical initialization times (GPU):
- Model loading: ~10-20s
- Dataset loading: ~30-60s
- Total: ~45-90s

### Query Time

Typical query times:
- Retrieval: ~2-5s
- Generation: ~3-10s
- Total: ~5-15s

### Memory Usage

- Embedding model: ~2GB GPU
- Reranker model: ~2GB GPU
- LLM model: ~8-16GB GPU
- Dataset: ~1-2GB RAM

## Error Handling

The pipeline handles errors gracefully:

```python
result = pipeline.query("...")

if not result['success']:
    error = result['error']

    if 'not initialized' in error:
        pipeline.initialize()
    elif 'timeout' in error:
        # Retry with longer timeout
        pass
    else:
        # Log and handle
        print(f"Query failed: {error}")
```

## Dependencies

- `config` - Configuration settings
- `model_manager` - Model loading
- `loader.dataloader` - Dataset loading
- `core.search.langgraph_orchestrator` - RAG workflow
- `core.generation.generation_engine` - LLM generation

## Future Enhancements

- [ ] Batch query processing
- [ ] Async query execution
- [ ] Query caching
- [ ] Model hot-swapping
- [ ] Distributed inference
