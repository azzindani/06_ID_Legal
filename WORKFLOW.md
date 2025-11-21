# Development Workflow

This document describes the AI-assisted development workflow used to build and maintain this project.

## Overview

This project uses Claude Code as an AI development assistant to:
- Implement features and refactor code
- Write and fix tests
- Manage Git operations
- Ensure CI/CD compliance
- Maintain code quality and alignment

## Branch Strategy

### Feature Branches
- All development happens on feature branches: `claude/<feature-name>-<session-id>`
- Never push directly to `main` or `master`
- Each session gets a unique branch for tracking

### Git Operations
```bash
# Always use -u flag when pushing
git push -u origin claude/<branch-name>

# Commit with descriptive messages using HEREDOC
git commit -m "$(cat <<'EOF'
Short summary line

- Detailed change 1
- Detailed change 2
EOF
)"
```

## Development Cycle

### 1. Task Planning
- Break complex tasks into smaller, trackable items
- Use todo lists to track progress
- Prioritize based on dependencies

### 2. Implementation
- Follow existing code patterns
- Use lazy imports to avoid heavy dependencies during testing
- Prefer editing existing files over creating new ones

### 3. Testing
- Run tests frequently: `pytest -m unit -v`
- Fix failing tests before committing
- Use `pytest.importorskip()` for optional dependencies
- Mark tests appropriately:
  - `@pytest.mark.unit` - No external dependencies
  - `@pytest.mark.integration` - Requires models/GPU

### 4. CI Compliance
- Ensure all unit tests pass locally before pushing
- Use lazy imports in `__init__.py` files to prevent import errors
- Check that requirements.txt has all dependencies uncommented

### 5. Code Review
- Verify alignment with original code (e.g., Kaggle_Demo.ipynb)
- Check that new features integrate properly
- Ensure backward compatibility

## Module Structure

### Lazy Imports Pattern
To prevent import errors during CI testing, use lazy imports:

```python
# In __init__.py
def __getattr__(name):
    if name == 'MyClass':
        from .module import MyClass
        return MyClass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['MyClass']
```

### Test Structure
```
tests/
├── unit/           # Fast tests, no external deps
├── integration/    # Tests requiring models/data
└── conftest.py     # Shared fixtures
```

## Common Commands

### Testing
```bash
# Run all unit tests
pytest -m unit -v

# Run specific test file
pytest tests/unit/test_providers.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Git
```bash
# Check status
git status

# View recent commits
git log --oneline -5

# Stage and commit
git add -A && git commit -m "message"

# Push to remote
git push -u origin <branch-name>
```

### Development
```bash
# Run Gradio UI
python ui/gradio_app.py

# Run FastAPI server
python api/server.py

# Run CLI
python main.py --query "your question"
```

## CI/CD Pipeline

### GitHub Actions Workflow
1. **Test Job** - Runs unit tests on Python 3.9, 3.10, 3.11
2. **Lint Job** - Checks formatting with Black, isort, flake8
3. **Build Job** - Creates distribution package
4. **Docker Job** - Builds and tests Docker image (on main only)

### CI Requirements
- All unit tests must pass
- No critical linting errors (E9, F63, F7, F82)
- Package must build successfully

## Troubleshooting

### Import Errors in CI
**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Use lazy imports in `__init__.py`:
```python
def __getattr__(name):
    if name == 'HeavyClass':
        from .heavy_module import HeavyClass
        return HeavyClass
    raise AttributeError(...)
```

### Test Failures
**Problem**: Tests fail due to missing dependencies

**Solution**: Add skip markers:
```python
import pytest
numpy = pytest.importorskip("numpy")
```

### API Mismatches
**Problem**: Tests use wrong method names

**Solution**: Check actual implementation and update tests:
```python
# Wrong: detector.detect(query)
# Right: detector.analyze_query(query)
```

## Best Practices

### Code Quality
- Follow existing patterns in the codebase
- Add type hints for function signatures
- Include docstrings for public APIs
- Log important operations with `logger_utils`

### Testing
- Test one thing per test function
- Use descriptive test names
- Include both positive and edge cases
- Mock external dependencies

### Documentation
- Update README when adding features
- Document configuration options in .env.example
- Keep directory structure map current

### Git Commits
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Include details in commit body
- Reference issues when applicable

## Session Workflow Example

```
1. User: "Add feature X"

2. Claude: Creates todo list
   - [ ] Research existing code
   - [ ] Implement feature
   - [ ] Write tests
   - [ ] Update documentation

3. Claude: Implements feature, marking todos as complete

4. Claude: Runs tests
   $ pytest -m unit -v

5. Claude: Fixes any failures

6. Claude: Commits and pushes
   $ git add -A
   $ git commit -m "Add feature X"
   $ git push -u origin claude/branch-name

7. User: Reviews and provides feedback

8. Repeat until complete
```

## Alignment Verification

When refactoring from monolithic code (like Jupyter notebooks):

1. **List original components**
   ```bash
   grep "^class \|^def " original.py
   ```

2. **Map to modular structure**
   - Original `ClassName` → `module/class_name.py`

3. **Verify functionality**
   - Check method signatures match
   - Ensure return types are compatible
   - Test with same inputs

4. **Document differences**
   - New features not in original
   - Improved implementations
   - Breaking changes

## Contributing

1. Create feature branch from main
2. Follow development cycle above
3. Ensure CI passes
4. Submit PR with description
5. Address review feedback
6. Merge when approved
