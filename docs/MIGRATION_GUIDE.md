# Migration Guide: Monolithic → Modular

## Overview

This guide helps you migrate from the original monolithic `EnhancedKGSearchEngine` to the new modular architecture.

## Migration Steps

### 1. Install Dependencies
```bash
pip install pytest  # For testing
```

### 2. Copy Files
```bash
# Create structure
mkdir -p core/search research utils config tests

# Copy new modular files
cp core/search/scoring.py your_project/core/search/
cp core/search/diversity.py your_project/core/search/
cp research/individual_researcher.py your_project/research/
cp core/search/search_engine.py your_project/core/search/
cp config/search_config.py your_project/config/
cp utils/logging_config.py your_project/utils/
```

### 3. Update Imports in Original Code

**Before:**
```python
search_engine = EnhancedKGSearchEngine(...)
results = search_engine.parallel_legal_research(query, ...)
```

**After:**
```python
from core.search.search_engine import EnhancedSearchEngine

search_engine = EnhancedSearchEngine(
    records=dataset_loader.all_records,
    embeddings=dataset_loader.embeddings,
    embedding_model=embedding_model,
    knowledge_graph=knowledge_graph,
    config=DEFAULT_RAG_CONFIG
)

results = search_engine.search(
    query=query,
    query_type=query_type,
    top_k=config['final_top_k'],
    regulation_filter=None,
    progress_callback=callback_function
)
```

### 4. Add Logging to Existing Functions

**Pattern:**
```python
from utils.logging_config import get_logger

logger = get_logger(__name__)

def your_function():
    logger.info("Function started")
    try:
        # your code
        logger.debug("Step completed")
        return result
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
```

### 5. Run Tests
```bash
pytest tests/test_modular_search.py -v
```

### 6. Test in Original System

Keep your original code as a fallback:
```python
# Add feature flag
USE_MODULAR_SEARCH = True

if USE_MODULAR_SEARCH:
    from core.search.search_engine import EnhancedSearchEngine
    search_engine = EnhancedSearchEngine(...)
else:
    # Original code
    search_engine = EnhancedKGSearchEngine(...)
```

## Logging Best Practices

### Function Entry/Exit
```python
logger.info(f"search() called with query_type={query_type}")
# ... code ...
logger.info(f"search() completed: {len(results)} results")
```

### Important Decisions
```python
if use_metadata_filter:
    logger.info("Using metadata-first strategy")
else:
    logger.info("Using semantic-first strategy")
```

### Performance Metrics
```python
start = time.time()
# ... expensive operation ...
elapsed = time.time() - start
logger.info(f"Operation completed in {elapsed:.2f}s")
```

### Errors with Context
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Failed at step X with input={input_data}: {e}", exc_info=True)
    raise
```

## Troubleshooting

### Issue: "Module not found"
**Solution:** Ensure `__init__.py` files exist in all directories:
```bash
touch core/__init__.py core/search/__init__.py research/__init__.py utils/__init__.py
```

### Issue: "Circular imports"
**Solution:** Move shared types to separate file:
```python
# types/search_types.py
from typing import Dict, List, Any

SearchResult = Dict[str, Any]
CandidateList = List[SearchResult]
```

### Issue: "Logger not working"
**Solution:** Check log directory exists:
```bash
mkdir -p logs
chmod 755 logs
```

## Performance Comparison

Run benchmarks:
```bash
python scripts/benchmarks.py --compare-implementations
```

Expected results:
- ✅ Same accuracy
- ✅ Similar speed (±5%)
- ✅ Better memory usage (modular imports)
- ✅ Easier debugging (granular logs)