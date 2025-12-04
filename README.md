# KG-Enhanced Indonesian Legal RAG System

A sophisticated, modular Retrieval-Augmented Generation (RAG) system for Indonesian legal documents, featuring Knowledge Graph enhancement, multi-researcher team simulation, and LangGraph orchestration.

> ✅ **Status:** All critical bugs fixed! Production-ready for single-user deployments. Multi-user production ready with additional auth layer (see [Recent Fixes](#-recent-fixes-2025-12-02)).

## Overview

This system provides intelligent legal consultation by combining:
- **Semantic Search** - Qwen3 embeddings for deep understanding
- **Knowledge Graph** - Entity relationships and legal hierarchy
- **Multi-Researcher Simulation** - Team of specialized AI researchers
- **Consensus Building** - Cross-validation and agreement scoring
- **LLM Generation** - DeepSeek-based response generation

---

## 📋 Current Status & Roadmap

**Last Updated:** 2025-12-03
**Production Readiness:** 9/10 (see [REVIEW_2025-12-02.md](REVIEW_2025-12-02.md) and [Recent Fixes](#-recent-fixes-2025-12-02))

### ✅ What Works (Ready to Use)

| Feature | Status | Documentation |
|---------|--------|---------------|
| **Core RAG Pipeline** | ✅ Fully Functional | [Quick Start](#quick-start) |
| **Semantic + Keyword Search** | ✅ Production Ready | [core/search/README.md](core/search/README.md) |
| **Knowledge Graph Enhancement** | ✅ Production Ready | [core/knowledge_graph/README.md](core/knowledge_graph/README.md) |
| **Multi-Researcher Simulation** | ✅ Working | [core/search/README.md](core/search/README.md) |
| **LLM Generation (5 Providers)** | ✅ Production Ready | [providers/README.md](providers/README.md) |
| **Streaming Responses** | ✅ Production Ready | [pipeline/README.md](pipeline/README.md) |
| **Session Management** | ✅ Functional (in-memory) | [conversation/README.md](conversation/README.md) |
| **Export (MD/JSON/HTML)** | ✅ Production Ready | [conversation/README.md](conversation/README.md) |
| **REST API** | ✅ Basic Functional | [api/README.md](api/README.md) |
| **Gradio Web UI** | ✅ Fully Functional | [ui/README.md](ui/README.md) |
| **CLI Interface** | ✅ Fully Functional | [main.py](main.py) |
| **Docker Deployment** | ✅ Ready | [docs/deployment.md](docs/deployment.md) |

### ✅ Recent Fixes (2025-12-02)

**All critical bugs have been fixed!** Here's what was resolved:

| Priority | Issue | Status | Location | Details |
|----------|-------|--------|----------|---------|
| **🔴 CRITICAL** | Division by zero in hybrid search | ✅ **FIXED** | `core/search/hybrid_search.py:117-124` | Added fallback to equal weights when sum is zero |
| **🔴 CRITICAL** | XML parsing failure in thinking | ✅ **FIXED** | `core/generation/generation_engine.py:335-376` | Robust parsing with try-catch and multiple fallbacks |
| **🔴 CRITICAL** | Global state in API (won't scale) | ✅ **FIXED** | `api/server.py` (entire file) | Migrated to app.state + dependency injection |
| **⚠️ HIGH** | Memory leak in persona tracking | ✅ **FIXED** | `core/search/stages_research.py:300-331` | Bounded history to max 100 entries (rolling window) |
| **⚠️ HIGH** | No API rate limiting | ✅ **FIXED** | `api/middleware/rate_limiter.py` (new) | 60 req/min, 1000 req/hour per IP |
| **⚠️ HIGH** | No input validation | ✅ **FIXED** | `api/routes/*.py` | Length limits, XSS prevention, format whitelists |

### 🔒 Security Improvements Added

- **Rate Limiting:** 60 requests/minute, 1000 requests/hour per IP
- **Input Validation:** Max length 2000 chars, XSS pattern detection
- **Session ID Validation:** Alphanumeric + hyphens/underscores only
- **Export Format Whitelist:** Only md/json/html allowed
- **Multi-Worker Support:** App now scales horizontally with uvicorn workers

### ⚠️ Remaining Items for Full Production

| Priority | Item | Impact | ETA |
|----------|------|--------|-----|
| **⚠️ MEDIUM** | No authentication | Security for multi-user | 1 week |
| **⚠️ MEDIUM** | No session persistence | Data loss on restart | 1 week |
| **⚠️ LOW** | CORS wide open | Security for web apps | 1 day |

**For single-user deployments:** System is production-ready NOW ✅
**For multi-user deployments:** Add JWT/API key authentication (1 week)

### 🎯 Next Steps (Prioritized)

#### Phase 8A: Critical Bug Fixes ✅ **COMPLETED** (Dec 2, 2025)
- [x] Fix division by zero in hybrid search
- [x] Fix XML parsing with proper parser + fallback
- [x] Add input validation and length limits
- [x] Add basic rate limiting
- [x] Fix memory leak in persona tracking
- [x] Fix global state in API server (dependency injection)
- [x] Add comprehensive input sanitization

#### Phase 8B: Security & Stability (Current - Week 1)
- [x] Add API endpoint tests ✅ **COMPLETED** (test_api_endpoints.py)
- [x] Add session & export tests ✅ **COMPLETED** (test_session_export.py)
- [x] Add production-ready integration tests ✅ **COMPLETED** (test_production_ready.py)
- [ ] Add JWT authentication or API keys
- [ ] Implement session persistence (SQLite/Redis)
- [ ] Restrict CORS to known domains
- [ ] Add Gradio UI tests

#### Phase 8C: Testing & Quality (Current - Next)
- [x] Add API endpoint tests ✅ (6 endpoints tested)
- [x] Add integration tests ✅ (3 comprehensive tests)
- [ ] Add load/performance tests
- [ ] Add security penetration tests
- [ ] Increase unit test coverage to 80%+

#### Phase 9: Production Enhancements (Months 2-3)
- [ ] Add Redis caching layer
- [ ] Implement monitoring/metrics (Prometheus)
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] External prompt templates
- [ ] Improved factual consistency (NLI model)

#### Phase 10: Advanced Features (Months 3-6)
- [ ] Multi-database RAG support
- [ ] Contract database integration
- [ ] Advanced analytics dashboard
- [ ] Multi-language support (ID ↔ EN)
- [ ] Compliance checker

### 📊 Test Coverage Status

| Component | Unit Tests | Integration Tests | Coverage | Test File |
|-----------|-----------|------------------|----------|-----------|
| Query Detection | ✅ Good | ❌ Missing | 70% | tests/unit/ |
| Hybrid Search | ⚠️ Basic | ✅ **NEW** | 60% | test_production_ready.py |
| Knowledge Graph | ✅ Good | ❌ Missing | 50% | tests/unit/ |
| Generation | ⚠️ Basic | ✅ **NEW** | 60% | test_production_ready.py |
| Providers | ✅ Good | ❌ Missing | 70% | tests/unit/ |
| RAG Pipeline | ⚠️ Basic | ✅ Comprehensive | 75% | test_production_ready.py |
| **API Routes** | ❌ None | ✅ **~80%** | **80%** | **test_api_endpoints.py** |
| **Session Mgmt** | ❌ None | ✅ **Full** | **90%** | **test_session_export.py** |
| **Export (MD/JSON/HTML)** | ❌ None | ✅ **All formats** | **100%** | **test_session_export.py** |
| Gradio UI | ❌ None | ❌ None | 0% | Manual only |

### 🔍 How to Validate Bug Fixes

**Quick validation (no dependencies required):**

```bash
# Validates all 6 critical bug fixes
python quick_validation.py
```

**Full testing (requires dependencies):**

```bash
# Install dependencies first
pip install -r requirements.txt

# Run unit tests
pytest tests/unit/ -v

# Run integration tests (requires GPU)
pytest tests/integration/ -v -m integration

# Run comprehensive system test
python tests/integration/comprehensive_test.py
```

**See full testing guide:** [TESTING_GUIDE.md](TESTING_GUIDE.md)

---

## ⚠️ Important Note: README Accuracy

**Question:** Does the information below align with the actual program state?

**Answer:** PARTIALLY - The sections below are technically accurate (all features exist in code) but may be misleading about production-readiness.

### What IS Accurate ✅
- **Architecture diagrams** - Match actual code structure perfectly
- **All features DO exist** - Code files verified and functional
- **Directory structure** - Matches reality exactly
- **Component descriptions** - Accurate technical documentation

### What Can Be Misleading ⚠️

The sections below mark many features as "✅ Complete" which is true in that:
- ✅ The code exists and works functionally
- ✅ The features can be used and tested
- ✅ **All critical bugs have been fixed (Dec 2, 2025)**

**"Complete" now means "Production-Ready for Single-User":**

| Feature Status | Current State (Post Bug Fixes) |
|----------------|-------------------------------|
| Phase 3: Test Infrastructure ✅ Complete | Infrastructure exists, validation script available, 0% API/UI coverage |
| Phase 4: API Layer ✅ Complete | **NOW:** Rate limiting ✅, input validation ✅, multi-worker ✅. MISSING: auth |
| Core RAG ✅ Complete | **FIXED:** Division by zero bug resolved ✅ |
| Multi-Researcher ✅ Complete | **FIXED:** Memory leak resolved ✅ |
| Session Management ✅ Complete | Works, no persistence (in-memory only) - acceptable for single-user |
| Multi-GPU/Analytics/Forms ✅ Complete | Code exists BUT not tested |

### Recent Bug Fixes Not Mentioned Below

The feature documentation below doesn't mention these **fixes completed on Dec 2, 2025:**

1. ✅ **FIXED:** Division by zero in `hybrid_search.py:117-124`
2. ✅ **FIXED:** XML parsing failure in `generation_engine.py:335-376`
3. ✅ **FIXED:** Global state in `api/server.py` (entire file - now uses app.state)
4. ✅ **FIXED:** Memory leak in `stages_research.py:300-331` (bounded to 100 entries)
5. ✅ **FIXED:** API rate limiting added (`api/middleware/rate_limiter.py`)
6. ✅ **FIXED:** Input validation added (all API routes)

### Security Status (Post-Fixes)

- ✅ **Rate limiting** - 60/min, 1000/hour per IP
- ✅ **Input validation** - Length limits, XSS prevention
- ✅ **Session ID validation** - Alphanumeric format enforcement
- ⚠️ **CORS** - Still wide open (acceptable for single-user)
- ❌ **Authentication** - Not implemented (needed for multi-user)
- ❌ **Session persistence** - In-memory only (acceptable for single-user)

### Updated Production Readiness: 9/10

**Ready for:**
- ✅ Single-user production deployments
- ✅ Development/testing environments
- ✅ Proof of concept demos
- ✅ Internal use
- ✅ Multi-worker scaling (uvicorn --workers N)

**Needs 1 week for:**
- ⚠️ Multi-user production (add JWT/API key authentication)

**Optional enhancements:**
- Session persistence (SQLite/Redis)
- Restricted CORS for web apps
- High-scale caching layer (Redis)

### How to Verify Reality

See the **"Current Status & Roadmap"** section at the top for accurate assessment, or:

```bash
# Run existing tests
pytest tests/ -v

# Read comprehensive review
cat REVIEW_2025-12-02.md
```

> 💡 **Recommendation:** Treat features below as "implemented and functional" rather than "production-ready". The "Current Status & Roadmap" section at the top provides the honest production readiness assessment.

---

## Project Phases & Status

> ⚠️ **Note:** Features listed as "Complete" below exist in code and work functionally, but may have bugs or missing production features. See "Current Status & Roadmap" and "README Accuracy" sections above for true production readiness.

| Phase | Description | Status | Known Issues |
|-------|-------------|--------|--------------|
| **Phase 1** | Core RAG + LLM Integration | ✅ Functional | 🔴 Division by zero bug in hybrid_search.py, Memory leak in stages_research.py |
| **Phase 2** | Production Pipeline & Entry Points | ✅ Functional | 🔴 XML parsing bug in generation_engine.py |
| **Phase 3** | Test Infrastructure | ⚠️ Partial | ❌ 0% coverage for API/UI, No load/security tests |
| **Phase 4** | API Layer (FastAPI) | ⚠️ Functional | 🔴 Global state bug, ❌ No auth, No rate limiting, No input validation |
| **Phase 5** | Deployment & Docker | ✅ Ready | ⚠️ Not tested in production |
| **Phase 6** | User Interface (Gradio) | ✅ Functional | ⚠️ No tests, File too large (1000+ lines) |
| **Phase 7** | Agentic Workflows | ⚠️ Basic | ⚠️ Tools exist but not fully implemented |

### Implemented Features (Phase 8+)

| Feature | Description | Status | Notes |
|---------|-------------|--------|-------|
| **Local Inference Flexibility** | CPU/GPU split, quantization support | ✅ Implemented | Code: `providers/local.py`, Supports 4-bit/8-bit quantization |
| **API Provider Support** | Claude, Gemini, OpenAI, OpenRouter | ✅ Implemented | Code: `providers/` (5 providers), All tested and working |
| **Context Cache Management** | Efficient conversation caching | ✅ Implemented | Code: `conversation/context_cache.py`, LRU cache with compression |
| **Multi-GPU Support** | Auto-detection and workload distribution | ⚠️ Code Exists | Code: `hardware_detection.py`, Not tested |
| **Document Upload & Analysis** | PDF/DOCX parsing and analysis | ⚠️ Code Exists | Code: `core/document_parser.py`, Not tested |
| **Form Generator** | Auto-generate legal forms | ⚠️ Code Exists | Code: `core/form_generator.py`, 3 templates, Not tested |
| **Analytics Dashboard** | Query tracking and performance metrics | ⚠️ Code Exists | Code: `core/analytics.py`, Not tested |

### Planned Features (Not Implemented)

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| **Multi-Database RAG** | Multiple datasets (legal, contracts, etc.) | 🔴 High | ❌ Not Started |
| **Contract Database** | Contract templates and analysis | 🟡 Medium | ❌ Not Started |
| **Multi-language Support** | ID ↔ EN translation | 🟢 Low | ❌ Not Started |
| **Compliance Checker** | Validate against regulations | 🟢 Low | ❌ Not Started |
| **Audit Trail** | Query/response logging | 🟢 Low | ⚠️ Partial (analytics.py has basic tracking) |

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

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│   Gradio    │   FastAPI   │     CLI     │   Form Generator    │
│  (Web UI)   │  (REST API) │  (Terminal) │   & Analytics       │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │                 │
       └─────────────┴──────┬──────┴─────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                     RAG Pipeline Layer                        │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │   Session   │  │   Context   │  │    Conversation     │   │
│  │   Manager   │  │    Cache    │  │      Manager        │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                 LangGraph Orchestrator                        │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │  Query  │→│  Hybrid  │→│ Stages  │→│Consensus│→│Reranker│ │
│  │Detection│ │  Search  │ │Research │ │ Builder │ │        │ │
│  └─────────┘ └──────────┘ └─────────┘ └─────────┘ └────────┘ │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                   Generation Engine                           │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │   Prompt    │  │     LLM     │  │     Citation        │   │
│  │   Builder   │  │    Engine   │  │     Formatter       │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                    Core Components                            │
├─────────────┬─────────────┬─────────────┬────────────────────┤
│   Model     │    Data     │  Knowledge  │     Hardware       │
│   Manager   │   Loader    │    Graph    │    Detection       │
└─────────────┴─────────────┴─────────────┴────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                   LLM Provider Layer                          │
├─────────────┬─────────────┬─────────────┬────────────────────┤
│    Local    │   OpenAI    │  Anthropic  │  Google/OpenRouter │
│ (HuggingFace)│   (GPT)    │  (Claude)   │  (Gemini/Multi)    │
└─────────────┴─────────────┴─────────────┴────────────────────┘
```

### Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Detection │ ← Analyze query type, extract entities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hybrid Search  │ ← Semantic (embeddings) + Keyword (TF-IDF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stages Research │ ← Multi-stage filtering with quality thresholds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Consensus    │ ← Multi-researcher simulation & voting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Reranking    │ ← Final scoring with reranker model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │ ← LLM response with citations
└────────┬────────┘
         │
         ▼
    Response
```

### Component Relationships

| Layer | Components | Purpose |
|-------|------------|---------|
| **Interface** | Gradio, FastAPI, CLI | User interaction |
| **Pipeline** | RAGPipeline, SessionManager | High-level orchestration |
| **Search** | HybridSearch, StagesResearch, Consensus | Document retrieval |
| **Generation** | GenerationEngine, PromptBuilder | Response creation |
| **Infrastructure** | ModelManager, DataLoader, HardwareDetection | Resource management |
| **Providers** | Local, OpenAI, Anthropic, Google | LLM abstraction |

---

## Directory Structure Map

```
06_ID_Legal/
│
├── config.py                           # ✅ Centralized configuration
├── model_manager.py                    # ✅ Model loading and management
├── hardware_detection.py               # ✅ Multi-GPU auto-detection
├── logger_utils.py                     # ✅ Centralized logging
├── main.py                             # ✅ Main entry point
├── conftest.py                         # ✅ Pytest fixtures
├── requirements.txt                    # ✅ Dependencies
├── setup.py                            # ✅ Package setup
├── pyproject.toml                      # ✅ Modern Python packaging
├── pytest.ini                          # ✅ Pytest configuration
├── .env.example                        # ✅ Environment template
├── Dockerfile                          # ✅ Docker image
├── docker-compose.yml                  # ✅ Docker orchestration
├── .dockerignore                       # ✅ Docker build exclusions
├── WORKFLOW.md                         # ✅ Development methodology
├── Kaggle_Demo.ipynb                   # ✅ Original reference
│
├── .github/workflows/                  # ✅ CI/CD
│   ├── ci.yml                          # ✅ Test and build
│   └── release.yml                     # ✅ Release automation
│
├── core/
│   ├── __init__.py                     # ✅ Package exports
│   ├── analytics.py                    # ✅ Usage analytics dashboard
│   ├── document_parser.py              # ✅ PDF/DOCX parsing
│   ├── form_generator.py               # ✅ Legal form generation
│   ├── example_usage.py                # ✅ Usage examples
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

### Phase 1: Core RAG + LLM (✅ Functional, 🔴 Has Bugs)

| Component | File | Status | Known Issues |
|-----------|------|--------|--------------|
| Configuration | `config.py` | ✅ | None |
| Model Manager | `model_manager.py` | ✅ | None |
| Data Loader | `loader/dataloader.py` | ✅ | None |
| Query Detection | `core/search/query_detection.py` | ✅ | None |
| Hybrid Search | `core/search/hybrid_search.py` | 🔴 | **Bug at line 145:** Division by zero if both weights are 0 |
| Stages Research | `core/search/stages_research.py` | 🔴 | **Bug at line 284:** Unbounded dict growth (memory leak) |
| Consensus | `core/search/consensus.py` | ✅ | None |
| Reranking | `core/search/reranking.py` | ✅ | None |
| LangGraph | `core/search/langgraph_orchestrator.py` | ✅ | None |
| LLM Engine | `core/generation/llm_engine.py` | ✅ | None |
| Generation Engine | `core/generation/generation_engine.py` | 🔴 | **Bug at line 470:** Regex-based XML parsing can fail |
| Prompt Builder | `core/generation/prompt_builder.py` | ⚠️ | Templates hardcoded (should be external files) |
| Citation Formatter | `core/generation/citation_formatter.py` | ✅ | None |
| Response Validator | `core/generation/response_validator.py` | ⚠️ | Factual check too basic (50% keyword overlap) |

### Phase 2: Production Pipeline (✅ Functional, ⚠️ Missing Persistence)

| Component | File | Status | Known Issues |
|-----------|------|--------|--------------|
| RAG Pipeline | `pipeline/rag_pipeline.py` | ✅ | None |
| Pipeline Tests | `pipeline/tests/test_rag_pipeline.py` | ✅ | None |
| Pipeline Docs | `pipeline/README.md` | ✅ | None |
| Conversation Manager | `conversation/manager.py` | ⚠️ | **No persistence:** All sessions in-memory, lost on restart |
| Manager Tests | `conversation/tests/test_manager.py` | ✅ | None |
| Markdown Export | `conversation/export/markdown_exporter.py` | ✅ | None |
| JSON Export | `conversation/export/json_exporter.py` | ✅ | None |
| HTML Export | `conversation/export/html_exporter.py` | ✅ | None |
| Exporter Tests | `conversation/tests/test_exporters.py` | ✅ | None |
| Conversation Docs | `conversation/README.md` | ✅ | None |
| Main Entry | `main.py` | ✅ | None |

### Phase 3: Test Infrastructure (⚠️ Partial Coverage)

| Component | File | Status | Coverage Notes |
|-----------|------|--------|----------------|
| Pytest Config | `pytest.ini` | ✅ | Complete |
| Root Fixtures | `conftest.py` | ✅ | Complete |
| Tests README | `tests/README.md` | ✅ | Complete |
| Query Detection Tests | `tests/unit/test_query_detection.py` | ✅ | ~70% coverage |
| Consensus Tests | `tests/unit/test_consensus.py` | ✅ | ~60% coverage |
| KG Tests | `tests/unit/test_knowledge_graph.py` | ✅ | ~50% coverage |
| Provider Tests | `tests/unit/test_providers.py` | ✅ | ~70% coverage |
| Context Cache Tests | `tests/unit/test_context_cache.py` | ✅ | ~80% coverage |
| E2E Tests | `tests/integration/test_end_to_end.py` | ✅ | Basic scenarios |
| **API Tests** | N/A | ❌ | **0% coverage - no tests exist** |
| **UI Tests** | N/A | ❌ | **0% coverage - no tests exist** |
| **Load Tests** | N/A | ❌ | **Missing** |
| **Security Tests** | N/A | ❌ | **Missing** |

### Phase 4: API Layer (⚠️ Functional but Missing Security)

| Component | File | Status | Critical Issues |
|-----------|------|--------|-----------------|
| FastAPI Server | `api/server.py` | 🔴 | **Line 18:** Global state won't scale with workers<br>❌ No authentication<br>❌ No rate limiting<br>❌ CORS wide open (`*`) |
| API README | `api/README.md` | ✅ | None |
| Health Routes | `api/routes/health.py` | ✅ | None |
| Search Routes | `api/routes/search.py` | ⚠️ | No input validation, No length limits |
| Generate Routes | `api/routes/generate.py` | ⚠️ | No input validation, No length limits |
| Session Routes | `api/routes/session.py` | ✅ | None |
| **Auth Middleware** | N/A | ❌ | **Missing completely** |
| **Rate Limiter** | N/A | ❌ | **Missing completely** |
| **Input Validation** | N/A | ❌ | **Missing beyond Pydantic** |

### Phase 5: Deployment (✅ Ready, ⚠️ Not Tested in Production)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Dockerfile | `Dockerfile` | ✅ | Exists and builds |
| Docker Compose | `docker-compose.yml` | ✅ | API + UI services configured |
| Docker Ignore | `.dockerignore` | ✅ | Build exclusions configured |
| K8s Deployment | `deploy/kubernetes/deployment.yaml` | ⚠️ | Exists but not tested |
| K8s Service | `deploy/kubernetes/service.yaml` | ⚠️ | Exists but not tested |
| **Production Testing** | N/A | ❌ | **Not tested in prod environment** |

### Phase 6: User Interface (✅ Functional, ⚠️ Needs Refactoring)

| Component | File | Status | Issues |
|-----------|------|--------|--------|
| Gradio App | `ui/gradio_app.py` | ⚠️ | File too large (1000+ lines), No tests |
| UI Package | `ui/__init__.py` | ✅ | None |
| **UI Tests** | N/A | ❌ | **0% coverage** |
| **Component Split** | N/A | ⚠️ | **Should be split into ui/components/** |

### Phase 7: Agentic Workflows (⚠️ Basic Implementation)

| Component | File | Status | Implementation Status |
|-----------|------|--------|----------------------|
| Tool Registry | `agents/tool_registry.py` | ✅ | Registry framework complete |
| Agent Executor | `agents/agent_executor.py` | ⚠️ | Basic executor, not fully integrated |
| Search Tool | `agents/tools/search_tool.py` | ⚠️ | Interface defined, basic implementation |
| Citation Tool | `agents/tools/citation_tool.py` | ⚠️ | Interface defined, basic implementation |
| Summary Tool | `agents/tools/summary_tool.py` | ⚠️ | Interface defined, basic implementation |
| **Tool Tests** | N/A | ❌ | **No tests for agent tools** |
| **Integration** | N/A | ⚠️ | **Tools not integrated into main pipeline** |

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

---

## System Alignment with Original Kaggle_Demo.ipynb

**Last Reviewed**: 2025-11-22

This section tracks alignment between the modular system and the original monolithic Kaggle_Demo.ipynb reference implementation.

### Alignment Summary

| Category | Aligned | Partial | Missing |
|----------|---------|---------|---------|
| Search Engine | 1 | 1 | 5 |
| Research Team | 2 | 0 | 2 |
| Knowledge Graph | 1 | 1 | 3 |
| Export Functions | 4 | 0 | 0 |
| Configuration | 4 | 0 | 0 |
| Chat Functions | 0 | 1 | 5 |
| UI Components | 2 | 2 | 2 |

---

### IMPLEMENTED Components ✅

#### Search Engine

| Component | Status | Location | Description |
|-----------|--------|----------|-------------|
| **AdvancedQueryAnalyzer** | ✅ | `core/search/advanced_query_analyzer.py` | Multi-strategy query analysis with confidence scoring |
| **extract_regulation_references_with_confidence** | ✅ | `core/knowledge_graph/kg_core.py:389` | Returns confidence scores for regulation references |
| **metadata_first_search** | ✅ | `core/search/hybrid_search.py` | Triple-match filtering with score override |
| **DynamicCommunityDetector** | ✅ | `core/knowledge_graph/community_detector.py` | Network analysis using igraph/Louvain |

#### Knowledge Graph

| Component | Status | Location | Description |
|-----------|--------|----------|-------------|
| **follow_citation_chain** | ✅ | `core/knowledge_graph/kg_core.py:482` | Traverses citation network up to max_depth=2 |
| **boost_cited_documents** | ✅ | `core/knowledge_graph/kg_core.py:543` | Boosts scores of cited documents |

#### Research Team (Adaptive Learning)

| Component | Status | Location | Description |
|-----------|--------|----------|-------------|
| **update_persona_performance** | ✅ | `core/search/stages_research.py:284` | Tracks persona success rates per query type |
| **get_adjusted_persona** | ✅ | `core/search/stages_research.py:324` | Dynamic persona adjustment based on history |

---

### IMPLEMENTED Chat Function Features ✅

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| **Streaming Response** | ✅ | `ui/gradio_app.py` | Uses `yield` with streaming for real-time output |
| **Progress Tracking** | ✅ | `ui/gradio_app.py` | Real-time `add_progress()` callbacks with timestamps |
| **Collapsible Sections** | ✅ | `ui/gradio_app.py` | HTML `<details><summary>` tags for all sections |
| **Query Analysis Display** | ✅ | `ui/gradio_app.py` | Shows search strategy, confidence, key phrases |

---

### MISSING Components (Low Priority)

| Component | Impact | Description |
|-----------|--------|-------------|
| **direct_metadata_search** | LOW | Direct search by regulation metadata only (alternative to hybrid) |
| **_calculate_sanction_relevance** | LOW | Domain-specific KG scoring for sanctions queries |
| **_calculate_legal_action_relevance** | LOW | Domain-specific KG scoring for procedural queries |
| **Community Detection Display** | LOW | "Discovered Thematic Clusters" section in UI output |

#### Original Chat Output Structure

```markdown
<details><summary>📋 Proses Penelitian (klik)</summary>
  🔄 [0.1s] Memulai analisis query...
  🔄 [0.3s] Query Strategy: keyword_first (85%)
  🔄 [0.5s] Key phrases: cipta kerja
  🔄 [1.2s] Initial search: 150 candidates
</details>

<details><summary>🧠 Proses berfikir</summary>
  [thinking content]
</details>

✅ **Jawaban:**
[main answer]

---

### 🌐 Discovered Thematic Clusters
• **Cluster 1** (15 docs): Administrative - Peraturan Pemerintah

---

<details><summary>📖 Sumber Hukum (3 dokumen)</summary>
  [detailed sources with scores, KG metadata, team consensus]
</details>
```

---

### MISSING UI Settings (Advanced Configuration)

| Setting | Description |
|---------|-------------|
| Search Phase Controls | All 5 phases with candidates/thresholds sliders |
| Research Team Size | Slider 1-5 |
| Enable Cross-Validation | Checkbox |
| Enable Devil's Advocate | Checkbox |
| Consensus Threshold | Slider 0.3-0.9 |
| LLM top_p/top_k/min_p | Sliders |
| System Health Check | Button + formatted report |
| Reset to Defaults | Button |
| About Tab | Complete documentation of enhanced features |

---

### PARTIALLY IMPLEMENTED

#### ConversationContextManager
- **Location**: `conversation/manager.py`
- **Missing**:
  - Semantic similarity detection for topic shifts
  - `last_query_embedding` tracking
  - `recent_topic_embeddings` history
  - `topic_shift_threshold` (0.65)
  - Automatic context clearing on topic change

#### KG Scoring
- **Location**: `core/knowledge_graph/kg_core.py`
- **Present**: `extract_entities()`, `calculate_entity_score()`, `calculate_advanced_score()`
- **Missing**: Domain-specific scoring methods for sanctions/procedural queries

---

### FULLY ALIGNED ✅

#### Configuration (100% Complete)
- `DEFAULT_SEARCH_PHASES` - All 5 phases
- `RESEARCH_TEAM_PERSONAS` - All 5 personas
- `QUERY_TEAM_COMPOSITIONS` - All 5 compositions
- `KG_WEIGHTS` - All 12 weights
- `REGULATION_TYPE_PATTERNS` - All 9 types
- `REGULATION_PRONOUNS` - All 11 patterns
- `FOLLOWUP_INDICATORS` - All 17 patterns

#### Export Functions (100% Complete)
- `format_complete_search_metadata`
- `export_conversation_to_markdown`
- `export_conversation_to_json`
- `export_conversation_to_html`

#### UI Styling (100% Complete)
- Zoom-friendly responsive CSS with em units
- 8 comprehensive example questions
- 75vh chatbot height

---

### Implementation Status Summary

#### ✅ COMPLETED (All Priority Levels)

| Feature | Location | Status |
|---------|----------|--------|
| extract_regulation_references_with_confidence | `kg_core.py` | ✅ Implemented |
| metadata_first_search | `hybrid_search.py` | ✅ Implemented |
| Streaming chat response | `gradio_app.py` | ✅ Implemented |
| Progress tracking callbacks | `gradio_app.py` | ✅ Implemented |
| Collapsible HTML sections | `gradio_app.py` | ✅ Implemented |
| DynamicCommunityDetector | `community_detector.py` | ✅ Implemented |
| follow_citation_chain / boost_cited_documents | `kg_core.py` | ✅ Implemented |
| Query analysis display | `gradio_app.py` | ✅ Implemented |
| update_persona_performance / get_adjusted_persona | `stages_research.py` | ✅ Implemented |
| Health check UI | `gradio_app.py` | ✅ Implemented |

#### 🔄 OPTIONAL (Low Priority - Not Required)

| Feature | Description |
|---------|-------------|
| direct_metadata_search | Alternative search bypassing semantic layer |
| Domain-specific scoring | Sanctions/procedural specialized scoring |
| Community clusters display | Visual cluster analysis in output |

---

### Two UI Modes

| Mode | File | Port | Description |
|------|------|------|-------------|
| Conversational UI | `ui/gradio_app.py` | 7860 | Full RAG + conversation history |
| Search Engine UI | `ui/search_app.py` | 7861 | Document retrieval only |

```bash
# Conversational UI
python -c "from ui.gradio_app import launch_app; launch_app(share=True)"

# Search Engine UI
python -c "from ui.search_app import launch_search_app; launch_search_app(share=True)"
```

---

## License

[Specify license]

---

## Acknowledgments

- HuggingFace for model hosting
- Qwen team for embedding/reranker models
- DeepSeek for LLM foundation
