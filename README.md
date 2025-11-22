# KG-Enhanced Indonesian Legal RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system for Indonesian legal documents, featuring Knowledge Graph enhancement, multi-researcher team simulation, and LangGraph orchestration.

## Overview

This system provides intelligent legal consultation by combining:
- **Semantic Search** - Qwen3 embeddings for deep understanding
- **Knowledge Graph** - Entity relationships and legal hierarchy
- **Multi-Researcher Simulation** - Team of specialized AI researchers
- **Consensus Building** - Cross-validation and agreement scoring
- **LLM Generation** - DeepSeek-based response generation

---

## Project Phases & Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Core RAG + LLM Integration | ✅ Complete |
| **Phase 2** | Production Pipeline & Entry Points | ✅ Complete |
| **Phase 3** | Test Infrastructure | ✅ Complete |
| **Phase 4** | API Layer (FastAPI) | ✅ Complete |
| **Phase 5** | Deployment & Docker | ✅ Complete |
| **Phase 6** | User Interface (Gradio) | ✅ Complete |
| **Phase 7** | Agentic Workflows | ✅ Complete |

### Upcoming Features (Phase 8+)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Local Inference Flexibility** | CPU/GPU split, quantization support | ✅ Complete |
| **API Provider Support** | Claude, Gemini, OpenAI, OpenRouter | ✅ Complete |
| **Context Cache Management** | Efficient conversation caching | ✅ Complete |
| **Multi-GPU Support** | Auto-detection and workload distribution | ✅ Complete |
| **Document Upload & Analysis** | PDF/DOCX parsing and analysis | ✅ Complete |
| **Form Generator** | Auto-generate legal forms | ✅ Complete |
| **Analytics Dashboard** | Query tracking and performance metrics | ✅ Complete |
| **Multi-Database RAG** | Multiple datasets (legal, contracts, etc.) | 🔴 High |
| **Contract Database** | Contract templates and analysis | 🟡 Medium |
| **Multi-language Support** | ID ↔ EN translation | 🟢 Low |
| **Compliance Checker** | Validate against regulations | 🟢 Low |
| **Audit Trail** | Query/response logging | 🟢 Low |

#### Suggested Additional Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Contract Review** | Clause extraction, risk identification | Due diligence |
| **Citation Graph** | Visualize regulation relationships | Legal research |
| **Template Library** | Pre-built document templates | Document drafting |
| **Collaborative Sessions** | Share sessions, team workspaces | Law firms |
| **Legal Glossary** | Term definitions with references | Education |
| **Version Comparison** | Compare regulation versions | Legislative tracking |

---

## Directory Structure Map

```
06_ID_Legal/
│
├── config.py                           # ✅ Centralized configuration
├── model_manager.py                    # ✅ Model loading and management
├── logger_utils.py                     # ✅ Centralized logging
├── main.py                             # ✅ Main entry point
├── requirements.txt                    # ✅ Dependencies
├── setup.py                            # ✅ Package setup
├── pyproject.toml                      # ✅ Modern Python packaging
├── .env.example                        # ✅ Environment template
├── Dockerfile                          # ✅ Docker image
├── docker-compose.yml                  # ✅ Docker orchestration
├── .dockerignore                       # ✅ Docker build exclusions
├── Kaggle_Demo.ipynb                   # ✅ Original reference
│
├── .github/workflows/                  # ✅ CI/CD
│   ├── ci.yml                          # ✅ Test and build
│   └── release.yml                     # ✅ Release automation
│
├── core/
│   ├── __init__.py                     # ✅ Package exports
│   │
│   ├── search/
│   │   ├── __init__.py                 # ✅ Exists
│   │   ├── query_detection.py          # ✅ Query analysis
│   │   ├── hybrid_search.py            # ✅ Semantic + keyword search
│   │   ├── stages_research.py          # ✅ Multi-stage research
│   │   ├── consensus.py                # ✅ Consensus building
│   │   ├── reranking.py                # ✅ Final reranking
│   │   └── langgraph_orchestrator.py   # ✅ LangGraph workflow
│   │
│   ├── generation/
│   │   ├── __init__.py                 # ✅ Exists
│   │   ├── llm_engine.py               # ✅ LLM model management
│   │   ├── generation_engine.py        # ✅ Generation orchestration
│   │   ├── prompt_builder.py           # ✅ Prompt construction
│   │   ├── citation_formatter.py       # ✅ Citation formatting
│   │   └── response_validator.py       # ✅ Response validation
│   │
│   └── knowledge_graph/                # ✅ KG module
│       ├── __init__.py                 # ✅ Package exports
│       ├── README.md                   # ✅ Module documentation
│       ├── kg_core.py                  # ✅ Entity extraction, scoring
│       ├── relationship_graph.py       # ✅ Network analysis
│       └── community_detection.py      # ✅ Dynamic communities
│
├── loader/
│   ├── __init__.py                     # ✅ Exists
│   └── dataloader.py                   # ✅ Dataset loading
│
├── providers/                           # ✅ LLM Provider abstraction
│   ├── __init__.py                     # ✅ Package exports
│   ├── base.py                         # ✅ Abstract base provider
│   ├── factory.py                      # ✅ Provider factory
│   ├── local.py                        # ✅ Local HuggingFace provider
│   ├── openai_provider.py              # ✅ OpenAI GPT provider
│   ├── anthropic_provider.py           # ✅ Anthropic Claude provider
│   ├── google_provider.py              # ✅ Google Gemini provider
│   └── openrouter_provider.py          # ✅ OpenRouter provider
│
├── conversation/                        # ✅ Conversation management
│   ├── __init__.py                     # ✅ Package exports
│   ├── README.md                       # ✅ Module documentation
│   ├── manager.py                      # ✅ Session state, history tracking
│   ├── context_cache.py                # ✅ LRU context cache with compression
│   ├── export/
│   │   ├── __init__.py                 # ✅ Export package
│   │   ├── base_exporter.py            # ✅ Abstract base class
│   │   ├── markdown_exporter.py        # ✅ Markdown export
│   │   ├── json_exporter.py            # ✅ JSON export
│   │   └── html_exporter.py            # ✅ HTML export
│   └── tests/
│       ├── __init__.py                 # ✅ Test package
│       ├── test_manager.py             # ✅ Manager tests
│       └── test_exporters.py           # ✅ Export tests
│
├── api/                                 # ✅ API layer
│   ├── __init__.py                     # ✅ Package exports
│   ├── README.md                       # ✅ API documentation
│   ├── server.py                       # ✅ FastAPI server
│   └── routes/
│       ├── __init__.py                 # ✅ Route exports
│       ├── health.py                   # ✅ Health checks
│       ├── search.py                   # ✅ Search endpoints
│       ├── generate.py                 # ✅ Generation endpoints
│       └── session.py                  # ✅ Session endpoints
│
├── ui/                                  # ✅ UI layer
│   ├── __init__.py                     # ✅ Package exports
│   ├── gradio_app.py                   # ✅ Gradio interface
│   └── components/
│       └── __init__.py                 # ✅ Components package
│
├── agents/                              # ✅ Agentic workflows
│   ├── __init__.py                     # ✅ Package exports
│   ├── tool_registry.py                # ✅ Tool management
│   ├── agent_executor.py               # ✅ Agent execution
│   └── tools/
│       ├── __init__.py                 # ✅ Tools package
│       ├── search_tool.py              # ✅ Search tool
│       ├── citation_tool.py            # ✅ Citation tool
│       └── summary_tool.py             # ✅ Summary tool
│
├── pipeline/                            # ✅ High-level pipelines
│   ├── __init__.py                     # ✅ Package exports
│   ├── README.md                       # ✅ Module documentation
│   ├── rag_pipeline.py                 # ✅ Complete RAG pipeline
│   ├── tests/
│   │   ├── __init__.py                 # ✅ Test package
│   │   └── test_rag_pipeline.py        # ✅ Unit + integration tests
│   ├── streaming_pipeline.py           # ✅ Streaming response
│   └── batch_pipeline.py               # ✅ Batch processing
│
├── tests/                               # ✅ Test infrastructure
│   ├── __init__.py                     # ✅ Test package
│   ├── README.md                       # ✅ Test documentation
│   │
│   ├── unit/                           # ✅ Unit tests
│   │   ├── __init__.py
│   │   ├── test_query_detection.py     # ✅ Query detection tests
│   │   ├── test_consensus.py           # ✅ Consensus tests
│   │   ├── test_providers.py           # ✅ Provider tests
│   │   └── test_context_cache.py       # ✅ Context cache tests
│   │
│   └── integration/                    # ✅ Integration tests
│       ├── __init__.py
│       └── test_end_to_end.py          # ✅ E2E tests
│
├── conftest.py                         # ✅ Root pytest fixtures
├── pytest.ini                          # ✅ Pytest configuration
│
├── scripts/                             # 🔴 Utility scripts
│   ├── initialize_system.py            # Setup script
│   ├── run_server.py                   # Production server
│   ├── run_gradio.py                   # Gradio dev server
│   ├── benchmarks.py                   # Performance benchmarks
│   └── migrate_from_notebook.py        # Migration helper
│
├── docs/                                # 🔴 Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment.md
│   └── development.md
│
├── logs/                                # Log files (gitignored)
├── exports/                             # Exported conversations
├── cache/                               # Model/data cache
│
└── deploy/                              # 🔴 Deployment configs
    ├── kubernetes/
    │   ├── deployment.yaml
    │   └── service.yaml
    ├── nginx/
    │   └── nginx.conf
    └── scripts/
        ├── build.sh
        └── deploy.sh
```

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete and aligned |
| 🟡 | Exists but needs work |
| 🔴 | Not started |

---

## Component Status Detail

### Phase 1: Core RAG + LLM (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Configuration | `config.py` | ✅ | All settings aligned with original |
| Model Manager | `model_manager.py` | ✅ | Embedding, reranker loading |
| Data Loader | `loader/dataloader.py` | ✅ | HuggingFace dataset with KG indexes |
| Query Detection | `core/search/query_detection.py` | ✅ | Query analysis and entity extraction |
| Hybrid Search | `core/search/hybrid_search.py` | ✅ | Semantic + keyword search |
| Stages Research | `core/search/stages_research.py` | ✅ | Multi-stage with quality degradation |
| Consensus | `core/search/consensus.py` | ✅ | Multi-researcher consensus building |
| Reranking | `core/search/reranking.py` | ✅ | Final reranking with reranker model |
| LangGraph | `core/search/langgraph_orchestrator.py` | ✅ | Workflow orchestration |
| LLM Engine | `core/generation/llm_engine.py` | ✅ | Model loading and generation |
| Generation Engine | `core/generation/generation_engine.py` | ✅ | Complete generation pipeline |
| Prompt Builder | `core/generation/prompt_builder.py` | ✅ | Context-aware prompts |
| Citation Formatter | `core/generation/citation_formatter.py` | ✅ | Legal citation formatting |
| Response Validator | `core/generation/response_validator.py` | ✅ | Response validation |

### Phase 2: Production Pipeline (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| RAG Pipeline | `pipeline/rag_pipeline.py` | ✅ | High-level API |
| Pipeline Tests | `pipeline/tests/test_rag_pipeline.py` | ✅ | Unit + integration tests |
| Pipeline Docs | `pipeline/README.md` | ✅ | Module documentation |
| Conversation Manager | `conversation/manager.py` | ✅ | Session and history |
| Manager Tests | `conversation/tests/test_manager.py` | ✅ | Manager unit tests |
| Markdown Export | `conversation/export/markdown_exporter.py` | ✅ | Export to markdown |
| JSON Export | `conversation/export/json_exporter.py` | ✅ | Export to JSON |
| HTML Export | `conversation/export/html_exporter.py` | ✅ | Export to HTML |
| Exporter Tests | `conversation/tests/test_exporters.py` | ✅ | Export unit tests |
| Conversation Docs | `conversation/README.md` | ✅ | Module documentation |
| Main Entry | `main.py` | ✅ | System entry point |

### Phase 3: Test Infrastructure (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Pytest Config | `pytest.ini` | ✅ | Test configuration |
| Root Fixtures | `conftest.py` | ✅ | Shared pytest fixtures |
| Tests README | `tests/README.md` | ✅ | Test documentation |
| Query Detection Tests | `tests/unit/test_query_detection.py` | ✅ | Query type detection |
| Consensus Tests | `tests/unit/test_consensus.py` | ✅ | Consensus building |
| E2E Tests | `tests/integration/test_end_to_end.py` | ✅ | End-to-end tests |

### Phase 4: API Layer (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| FastAPI Server | `api/server.py` | ✅ | REST API server |
| API README | `api/README.md` | ✅ | API documentation |
| Health Routes | `api/routes/health.py` | ✅ | Health checks |
| Search Routes | `api/routes/search.py` | ✅ | Search endpoints |
| Generate Routes | `api/routes/generate.py` | ✅ | Generation endpoints |
| Session Routes | `api/routes/session.py` | ✅ | Session endpoints |

### Phase 5: Deployment (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Dockerfile | `Dockerfile` | ✅ | Container image |
| Docker Compose | `docker-compose.yml` | ✅ | Multi-container setup |
| Docker Ignore | `.dockerignore` | ✅ | Build exclusions |

### Phase 6: User Interface (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Gradio App | `ui/gradio_app.py` | ✅ | Main Gradio interface |
| UI Package | `ui/__init__.py` | ✅ | Package exports |

### Phase 7: Agentic Workflows (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Tool Registry | `agents/tool_registry.py` | ✅ | Tool management |
| Agent Executor | `agents/agent_executor.py` | ✅ | Agent runtime |
| Search Tool | `agents/tools/search_tool.py` | ✅ | Document search |
| Citation Tool | `agents/tools/citation_tool.py` | ✅ | Citation lookup |
| Summary Tool | `agents/tools/summary_tool.py` | ✅ | Summarization |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/azzindani/06_ID_Legal.git
cd 06_ID_Legal

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Running the System

#### Option 1: Command Line Interface
```bash
# Interactive mode
python main.py

# Single query
python main.py --query "Apa sanksi pelanggaran UU Ketenagakerjaan?"

# Export session
python main.py --export SESSION_ID --format md
```

#### Option 2: REST API
```bash
# Start API server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# API docs at http://localhost:8000/docs
```

#### Option 3: Web UI (Gradio)
```bash
python ui/gradio_app.py
# Open http://localhost:7860
```

#### Option 4: Docker
```bash
# API only
docker-compose up

# API + UI
docker-compose --profile ui up
```

### Simple Python Usage

```python
from pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()
pipeline.initialize()

# Query
result = pipeline.query("Apa sanksi pelanggaran UU Ketenagakerjaan?")
print(result['answer'])

# Cleanup
pipeline.shutdown()
```

### Advanced Usage (Phase 1)

```python
from config import get_default_config, DEFAULT_SEARCH_PHASES, DATASET_NAME, EMBEDDING_DIM
from model_manager import load_models
from loader.dataloader import EnhancedKGDatasetLoader
from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator
from core.generation.generation_engine import GenerationEngine

# Initialize configuration
config = get_default_config()
config['search_phases'] = DEFAULT_SEARCH_PHASES

# Load models
embedding_model, reranker_model = load_models()

# Load dataset
loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
loader.load_from_huggingface()

# Create RAG orchestrator
orchestrator = LangGraphRAGOrchestrator(
    data_loader=loader,
    embedding_model=embedding_model,
    reranker_model=reranker_model,
    config=config
)

# Run query
result = orchestrator.run("Apa sanksi pelanggaran UU Ketenagakerjaan?")

# Generate response
gen_engine = GenerationEngine(config)
gen_engine.initialize()

response = gen_engine.generate_answer(
    query="Apa sanksi pelanggaran UU Ketenagakerjaan?",
    retrieved_results=result['final_results']
)

print(response['answer'])
```

---

## Configuration

### Key Settings (Aligned with Original)

```python
# config.py

DATASET_NAME = "Azzindani/ID_REG_DB_2510"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
LLM_MODEL = "Azzindani/Deepseek_ID_Legal_Preview"

DEFAULT_CONFIG = {
    'initial_quality': 0.95,
    'quality_degradation': 0.1,
    'min_quality': 0.5,
    'consensus_threshold': 0.6,
    'final_top_k': 3,
    'max_rounds': 5,
    'temperature': 0.7,
    'max_new_tokens': 2048
}
```

### Search Phase Thresholds

| Phase | Candidates | Semantic | Keyword |
|-------|------------|----------|---------|
| initial_scan | 400 | 0.20 | 0.06 |
| focused_review | 150 | 0.35 | 0.12 |
| deep_analysis | 60 | 0.45 | 0.18 |
| verification | 30 | 0.55 | 0.22 |
| expert_review | 45 | 0.50 | 0.20 |

### Research Team Personas

| Persona | Experience | Accuracy Bonus |
|---------|------------|----------------|
| Senior Legal Researcher | 15 years | +15% |
| Junior Legal Researcher | 3 years | 0% |
| KG Specialist | 8 years | +10% |
| Procedural Expert | 12 years | +8% |
| Devil's Advocate | 10 years | +12% |

---

## Testing

### Cloud Testing

Tests can be run on cloud services (Kaggle, Colab, etc.) with GPU support:

```bash
# Run pipeline unit tests (no GPU required)
pytest pipeline/tests/test_rag_pipeline.py -m "not integration" -v

# Run pipeline integration tests (requires GPU)
pytest pipeline/tests/test_rag_pipeline.py -m integration -v

# Run all pipeline tests
pytest pipeline/tests/test_rag_pipeline.py -v

# Run with coverage
pytest pipeline/tests/ --cov=pipeline --cov-report=html
```

### Current Tests

```bash
# Pipeline tests
pytest pipeline/tests/test_rag_pipeline.py -v

# Conversation tests
pytest conversation/tests/test_manager.py -v
pytest conversation/tests/test_exporters.py -v

# Existing tests
python -m pytest loader/test_dataloader.py
python -m pytest core/search/test_integrated_system.py
python -m pytest core/generation/test_generation.py
```

### Planned Test Structure

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# All tests
pytest tests/
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `integration` | Requires GPU and full model loading |
| `slow` | Performance/benchmark tests |
| (default) | Unit tests, no GPU required |

---

## Performance Notes

- **Embedding Model**: ~600M parameters, GPU recommended
- **Reranker Model**: ~600M parameters
- **LLM Model**: DeepSeek-based, supports streaming
- **Dataset**: ~100K+ regulation chunks with KG metadata

### Memory Optimization

- Lazy JSON parsing for KG data
- Chunked dataset loading (5000 records)
- Compressed embeddings (float16)
- Sparse TF-IDF matrices

---

## Contributing

1. Check the Phase status above
2. Pick a component marked 🔴
3. Follow existing code patterns
4. Include tests
5. Update this README

---

## License

[Specify license]

---

## Acknowledgments

- HuggingFace for model hosting
- Qwen team for embedding/reranker models
- DeepSeek for LLM foundation
