# Demo & Simulation Scripts

Real, runnable test scripts to validate all features of the Indonesian Legal Assistant.

These are **NOT pytest tests** - these are actual simulation scripts you can run to see the system in action and verify it works as expected.

---

## Quick Start

```bash
# Run all demos in sequence
python demos/08_full_system_test.py

# Or run individual demos
python demos/01_basic_rag_query.py
python demos/02_multi_researcher.py
# ... etc
```

---

## Available Demos

| Demo | Description | Duration | Requirements |
|------|-------------|----------|--------------|
| **01_basic_rag_query.py** | Basic end-to-end RAG query | ~30s | GPU recommended |
| **02_multi_researcher.py** | Multi-researcher simulation | ~45s | GPU recommended |
| **03_knowledge_graph.py** | Knowledge Graph enhancement | ~30s | GPU recommended |
| **04_streaming.py** | Streaming response | ~20s | GPU recommended |
| **05_session_management.py** | Session & export features | ~15s | None |
| **06_all_providers.py** | Test all 5 LLM providers | ~2min | API keys optional |
| **07_api_test.py** | REST API endpoints | ~30s | API server running |
| **08_full_system_test.py** | Complete system validation | ~5min | GPU recommended |

---

## Setup

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (optional for API providers)
cp .env.example .env
# Edit .env with your API keys if testing cloud providers
```

### For API Tests

```bash
# Terminal 1: Start API server
python api/server.py

# Terminal 2: Run API tests
python demos/07_api_test.py
```

---

## Demo Details

### 01: Basic RAG Query
**File:** `01_basic_rag_query.py`

**What it tests:**
- Pipeline initialization
- Basic query execution
- Answer generation
- Source retrieval
- Metadata tracking

**Expected output:**
```
=== Indonesian Legal Assistant: Basic RAG Demo ===

[1/5] Initializing pipeline...
  ✓ Models loaded
  ✓ Dataset loaded
  ✓ Pipeline ready

[2/5] Running query: "Apa itu UU Ketenagakerjaan?"
  ✓ Query executed in 8.5s

[3/5] Answer:
UU Ketenagakerjaan adalah Undang-Undang Nomor 13 Tahun 2003 tentang...

[4/5] Sources Retrieved: 3
  - UU 13/2003 - Ketenagakerjaan (score: 0.95)
  - ...

[5/5] Metadata:
  - Total time: 8.5s
  - Retrieval time: 2.1s
  - Generation time: 6.4s
  - Tokens: 256

✅ Basic RAG Demo Complete
```

### 02: Multi-Researcher Simulation
**File:** `02_multi_researcher.py`

**What it tests:**
- Multi-persona research
- Consensus building
- Quality degradation across stages
- Cross-validation

**Shows:** How the system simulates multiple legal researchers with different expertise levels working together.

### 03: Knowledge Graph Enhancement
**File:** `03_knowledge_graph.py`

**What it tests:**
- Entity extraction
- Regulation reference detection
- KG-based scoring
- Citation chain traversal

**Shows:** How the KG enhances search results with legal hierarchy understanding.

### 04: Streaming Response
**File:** `04_streaming.py`

**What it tests:**
- Token-by-token streaming
- Real-time response display
- Streaming metadata

**Shows:** Live streaming output as the LLM generates the response.

### 05: Session Management
**File:** `05_session_management.py`

**What it tests:**
- Session creation
- Multi-turn conversation
- History tracking
- Export to MD/JSON/HTML

**Shows:** Complete conversation flow with export functionality.

### 06: All LLM Providers
**File:** `06_all_providers.py`

**What it tests:**
- Local (HuggingFace)
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- OpenRouter

**Shows:** Same query across all providers with comparison.

**Note:** Requires API keys for cloud providers. Will skip providers without keys.

### 07: API Endpoints
**File:** `07_api_test.py`

**What it tests:**
- Health check
- Search endpoint
- Generate endpoint
- Streaming endpoint
- Session endpoints

**Requires:** API server running (`python api/server.py`)

### 08: Full System Test
**File:** `08_full_system_test.py`

**What it tests:**
- All of the above
- Performance benchmarks
- Error handling
- Resource cleanup

**Generates:** Complete test report with timing and results.

---

## Interpreting Results

### Success Criteria

Each demo will show:
- ✅ **Green checkmarks** for successful steps
- ⚠️ **Yellow warnings** for non-critical issues
- ❌ **Red X** for failures

### Expected Performance

| Component | Expected Time | Notes |
|-----------|--------------|-------|
| Pipeline init | 30-90s | First run downloads models |
| Simple query | 5-15s | Depends on GPU |
| Streaming query | 8-20s | Similar to simple |
| Session export | <1s | Fast file I/O |
| API call | <10s | Plus processing time |

### Troubleshooting

**Problem:** "ModuleNotFoundError"
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem:** "CUDA out of memory"
```bash
# Solution: Enable quantization
export LLM_LOAD_IN_4BIT=true
python demos/01_basic_rag_query.py
```

**Problem:** "Connection refused" (API tests)
```bash
# Solution: Start API server first
python api/server.py &
sleep 10  # Wait for initialization
python demos/07_api_test.py
```

---

## Creating Custom Demos

Template for new demos:

```python
"""
Demo: Your Feature Name

Tests: What this demo validates
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("DEMO: Your Feature Name")
    print("=" * 60)
    print()

    # Import components
    from pipeline import RAGPipeline

    # Setup
    print("[1/N] Setting up...")
    pipeline = RAGPipeline()

    # Test
    print("[2/N] Testing feature...")
    result = pipeline.query("Test query")

    # Validate
    print("[3/N] Validating results...")
    assert result is not None
    print("  ✓ Validation passed")

    # Cleanup
    print("[N/N] Cleaning up...")
    pipeline.shutdown()

    print()
    print("=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Continuous Integration

These demos can be used in CI/CD:

```yaml
# .github/workflows/demo-tests.yml
name: Demo Tests

on: [push, pull_request]

jobs:
  demo-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: python demos/01_basic_rag_query.py
      - run: python demos/05_session_management.py
      # GPU-required demos skip on CI
```

---

## Contributing

When adding new features:
1. Create a demo script in `demos/`
2. Update this README
3. Add to `08_full_system_test.py`

---

## License

Same as main project.
