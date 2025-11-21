# Test Suite

Comprehensive test suite for the Indonesian Legal RAG System.

## Test Organization

```
tests/
├── unit/                    # Unit tests (no external deps)
│   ├── test_query_detection.py
│   └── test_consensus.py
├── integration/             # Integration tests (GPU/models)
│   └── test_end_to_end.py
└── README.md
```

## Running Tests

### All Tests
```bash
pytest
```

### By Marker
```bash
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
pytest -m "not gpu"         # Skip GPU tests
```

### By Module
```bash
pytest tests/unit/                          # All unit tests
pytest tests/integration/                   # All integration tests
pytest conversation/tests/                  # Conversation tests
pytest pipeline/tests/                      # Pipeline tests
```

### With Coverage
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Verbose Output
```bash
pytest -v --tb=long
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests, no external dependencies |
| `integration` | Integration tests, may need GPU |
| `slow` | Tests that take >10 seconds |
| `gpu` | Tests requiring GPU |
| `api` | API endpoint tests |

## Fixtures

Common fixtures are defined in `conftest.py`:

- `test_config` - Base test configuration
- `sample_document` - Sample legal document
- `sample_documents` - Multiple documents
- `sample_query` - Sample query string
- `sample_session_data` - Session with turns
- `sample_search_results` - Search results
- `mock_embedding_model` - Mocked embedding model
- `mock_llm_model` - Mocked LLM
- `temp_dir` - Temporary directory

## Writing Tests

### Unit Test Example
```python
import pytest
from core.search.query_detection import QueryDetector

@pytest.mark.unit
class TestQueryDetector:
    @pytest.fixture
    def detector(self):
        return QueryDetector()

    def test_detect_sanctions(self, detector):
        result = detector.detect("Apa sanksi?")
        assert result['query_type'] == 'sanctions'
```

### Integration Test Example
```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        from pipeline import RAGPipeline
        p = RAGPipeline()
        p.initialize()
        yield p
        p.shutdown()

    def test_query(self, pipeline):
        result = pipeline.query("Test question")
        assert 'answer' in result
```

## CI/CD

Tests are configured for CI with:
- Timeout: 300s per test
- Auto-skip GPU tests when `CI=true`
- Strict marker enforcement

## Dependencies

```bash
pip install pytest pytest-cov pytest-timeout
```
