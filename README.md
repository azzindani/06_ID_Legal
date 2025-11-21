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
| **Phase 2** | Production Pipeline & Entry Points | 🟡 In Progress |
| **Phase 3** | Test Infrastructure | 🔴 Not Started |
| **Phase 4** | API Layer (FastAPI) | 🔴 Not Started |
| **Phase 5** | Deployment & Docker | 🔴 Not Started |
| **Phase 6** | User Interface (Gradio) | 🔴 Not Started |
| **Phase 7** | Agentic Workflows | 🔴 Not Started |

---

## Directory Structure Map

```
06_ID_Legal/
│
├── config.py                           # ✅ Centralized configuration
├── model_manager.py                    # ✅ Model loading and management
├── logger_utils.py                     # ✅ Centralized logging
├── main.py                             # 🔴 Main entry point
├── requirements.txt                    # ✅ Dependencies
├── setup.py                            # 🔴 Package setup
├── pyproject.toml                      # 🔴 Modern Python packaging
├── .env.example                        # ✅ Environment template
├── Dockerfile                          # 🔴 Docker image
├── docker-compose.yml                  # 🔴 Docker orchestration
├── Kaggle_Demo.ipynb                   # ✅ Original reference
│
├── core/
│   ├── __init__.py                     # 🔴 Package exports
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
│   └── knowledge_graph/                # 🔴 Separate KG module
│       ├── __init__.py
│       ├── kg_core.py                  # Entity extraction, scoring
│       ├── relationship_graph.py       # Network analysis
│       └── community_detection.py      # Dynamic communities
│
├── loader/
│   ├── __init__.py                     # ✅ Exists
│   └── dataloader.py                   # ✅ Dataset loading
│
├── conversation/                        # 🔴 Conversation management
│   ├── __init__.py
│   ├── manager.py                      # Session state, history tracking
│   ├── context_enhancer.py             # Context-aware enhancements
│   └── export/
│       ├── __init__.py
│       ├── base_exporter.py            # Abstract base class
│       ├── markdown_exporter.py        # Markdown export
│       ├── json_exporter.py            # JSON export
│       └── html_exporter.py            # HTML export
│
├── api/                                 # 🔴 API layer
│   ├── __init__.py
│   ├── server.py                       # FastAPI server
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── search.py                   # Search endpoints
│   │   ├── generate.py                 # Generation endpoints
│   │   └── health.py                   # Health checks
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py                     # Authentication (future)
│       └── rate_limit.py               # Rate limiting (future)
│
├── ui/                                  # 🔴 UI layer
│   ├── __init__.py
│   ├── gradio_app.py                   # Gradio interface
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat_interface.py           # Chat UI component
│   │   ├── settings_panel.py           # Settings panel
│   │   └── export_panel.py             # Export panel
│   └── styles/
│       └── custom_css.py               # Custom styling
│
├── agents/                              # 🔴 Future agentic workflows
│   ├── __init__.py
│   ├── tool_registry.py                # Tool definitions
│   ├── agent_executor.py               # Agent execution
│   └── tools/
│       ├── __init__.py
│       ├── search_tool.py              # Search as tool
│       ├── citation_tool.py            # Citation lookup
│       └── summary_tool.py             # Summarization
│
├── pipeline/                            # 🟡 High-level pipelines
│   ├── __init__.py                     # ✅ Package exports
│   ├── README.md                       # ✅ Module documentation
│   ├── rag_pipeline.py                 # ✅ Complete RAG pipeline
│   ├── tests/
│   │   ├── __init__.py                 # ✅ Test package
│   │   └── test_rag_pipeline.py        # ✅ Unit + integration tests
│   ├── streaming_pipeline.py           # 🔴 Streaming response (future)
│   └── batch_pipeline.py               # 🔴 Batch processing (future)
│
├── tests/                               # 🟡 Needs reorganization
│   ├── __init__.py
│   ├── conftest.py                     # 🔴 Pytest fixtures
│   │
│   ├── unit/                           # 🔴 Unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_query_detection.py
│   │   ├── test_hybrid_search.py
│   │   ├── test_stages_research.py
│   │   ├── test_consensus.py
│   │   ├── test_reranking.py
│   │   ├── test_prompt_builder.py
│   │   ├── test_citation_formatter.py
│   │   └── test_response_validator.py
│   │
│   ├── integration/                    # 🔴 Integration tests
│   │   ├── __init__.py
│   │   ├── test_search_pipeline.py
│   │   ├── test_generation_pipeline.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_langgraph_flow.py
│   │
│   ├── e2e/                            # 🔴 End-to-end tests
│   │   ├── __init__.py
│   │   ├── test_complete_workflow.py
│   │   ├── test_api_endpoints.py
│   │   └── test_gradio_ui.py
│   │
│   └── fixtures/                       # 🔴 Test data
│       ├── __init__.py
│       ├── sample_queries.py
│       ├── sample_records.py
│       └── mock_responses.py
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

### Phase 2: Production Pipeline (🟡 In Progress)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| RAG Pipeline | `pipeline/rag_pipeline.py` | ✅ | High-level API |
| Pipeline Tests | `pipeline/tests/test_rag_pipeline.py` | ✅ | Unit + integration tests |
| Pipeline Docs | `pipeline/README.md` | ✅ | Module documentation |
| Main Entry | `main.py` | 🔴 | System entry point |
| Conversation Manager | `conversation/manager.py` | 🔴 | Session and history |
| Markdown Export | `conversation/export/markdown_exporter.py` | 🔴 | Export to markdown |
| JSON Export | `conversation/export/json_exporter.py` | 🔴 | Export to JSON |
| HTML Export | `conversation/export/html_exporter.py` | 🔴 | Export to HTML |

### Phase 3: User Interface (🔴 Not Started)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Gradio App | `ui/gradio_app.py` | 🔴 | Main Gradio interface |
| Chat Interface | `ui/components/chat_interface.py` | 🔴 | Chat component |
| Settings Panel | `ui/components/settings_panel.py` | 🔴 | User settings |
| Export Panel | `ui/components/export_panel.py` | 🔴 | Export UI |

### Phase 4: API Layer (🔴 Not Started)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| FastAPI Server | `api/server.py` | 🔴 | REST API server |
| Search Routes | `api/routes/search.py` | 🔴 | Search endpoints |
| Generate Routes | `api/routes/generate.py` | 🔴 | Generation endpoints |
| Health Routes | `api/routes/health.py` | 🔴 | Health checks |

### Phase 5: Deployment (🔴 Not Started)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Dockerfile | `Dockerfile` | 🔴 | Container image |
| Docker Compose | `docker-compose.yml` | 🔴 | Multi-container setup |
| K8s Deployment | `deploy/kubernetes/deployment.yaml` | 🔴 | Kubernetes config |

### Phase 6: Agentic Workflows (🔴 Not Started)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Tool Registry | `agents/tool_registry.py` | 🔴 | Tool definitions |
| Agent Executor | `agents/agent_executor.py` | 🔴 | Agent runtime |
| Search Tool | `agents/tools/search_tool.py` | 🔴 | Search as agent tool |

---

## Quick Start

### Current Usage (Phase 1)

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

## Installation

```bash
# Clone repository
git clone <repository>
cd 06_ID_Legal

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env with your settings
```

### Key Dependencies

- torch
- transformers
- langgraph
- gradio
- datasets
- scipy
- igraph
- python-louvain
- fastapi (Phase 4)
- uvicorn (Phase 4)

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
# Pipeline tests (NEW)
pytest pipeline/tests/test_rag_pipeline.py -v

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
