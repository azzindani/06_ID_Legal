# Development Guide

## Project Structure

```
06_ID_Legal/
├── core/                    # Core RAG components
│   ├── search/             # Search and retrieval
│   ├── generation/         # LLM generation
│   └── knowledge_graph/    # Graph enhancement
├── providers/              # LLM provider implementations
├── conversation/           # Session management
├── pipeline/               # High-level pipelines
├── loader/                 # Data loading
├── api/                    # FastAPI server
├── ui/                     # Gradio interface
├── agents/                 # Agentic workflows
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── deploy/                 # Deployment configs
```

## Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Coding Standards

### Style
- Follow PEP 8
- Use type hints
- Maximum line length: 127 characters

### Formatting
```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

### Docstrings
```python
def function_name(param: str) -> Dict[str, Any]:
    """
    Short description.

    Args:
        param: Parameter description

    Returns:
        Return value description

    Raises:
        ValueError: When param is invalid
    """
```

## Testing

### Run Tests
```bash
# All unit tests
pytest -m unit -v

# Specific file
pytest tests/unit/test_providers.py -v

# With coverage
pytest --cov=. --cov-report=html
```

### Writing Tests
```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    @pytest.fixture
    def setup(self):
        return MyClass()

    def test_basic_functionality(self, setup):
        result = setup.method()
        assert result == expected
```

### Test Markers
- `@pytest.mark.unit` - Fast tests, no external deps
- `@pytest.mark.integration` - Requires models/data
- `@pytest.mark.slow` - Long-running tests

## Adding New Features

### 1. Create Module
```python
# core/search/new_feature.py
class NewFeature:
    def __init__(self, config: Dict):
        self.config = config

    def process(self, input: str) -> str:
        pass
```

### 2. Add to __init__.py
```python
# Use lazy import pattern
def __getattr__(name):
    if name == 'NewFeature':
        from .new_feature import NewFeature
        return NewFeature
    ...
```

### 3. Write Tests
```python
# tests/unit/test_new_feature.py
@pytest.mark.unit
class TestNewFeature:
    def test_process(self):
        ...
```

### 4. Update Documentation
- Add to README if user-facing
- Add to API reference if API

## Adding New Provider

```python
# providers/my_provider.py
from .base import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = config.get('api_key')

    def initialize(self) -> bool:
        # Connect to service
        return True

    def generate(self, prompt, max_tokens=1000, **kwargs) -> str:
        # Call API
        return response
```

Register in factory:
```python
# providers/factory.py
PROVIDERS['my_provider'] = MyProvider
```

## Debugging

### Logging
```python
from logger_utils import get_logger
logger = get_logger(__name__)

logger.info("Message", {"key": "value"})
logger.debug("Debug info")
logger.error("Error", {"error": str(e)})
```

### Common Issues

1. **Import errors**: Check lazy imports in __init__.py
2. **Memory issues**: Enable quantization
3. **Slow tests**: Use mocks for external services

## Git Workflow

1. Create feature branch
2. Make changes
3. Run tests
4. Format code
5. Commit with descriptive message
6. Push and create PR

See WORKFLOW.md for detailed process.
