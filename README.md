# Modular RAG System Architecture - Production-Ready Structure

I'll create a comprehensive, future-proof modular architecture for your Indonesian Legal RAG system. This structure is designed for:
- ✅ Independent component testing
- ✅ Easy model/LLM swapping
- ✅ Scalability (API inference, voice, agents)
- ✅ Production-level logging & monitoring
- ✅ Multiple export formats
- ✅ Clean separation of concerns

---

## 📁 **PROPOSED DIRECTORY STRUCTURE**

```
indonesian_legal_rag/
│
├── config/
│   ├── __init__.py
│   ├── model_config.py          # Model paths, HF tokens
│   ├── search_config.py         # Search phases, team personas
│   ├── llm_config.py            # LLM generation params
│   └── app_config.py            # Gradio UI, export settings
│
├── core/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding_model.py   # Embedding model wrapper
│   │   ├── reranker_model.py    # Reranker wrapper
│   │   └── llm_model.py         # LLM wrapper (local + API)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py    # HuggingFace dataset loading
│   │   └── preprocessing.py     # Data cleaning, validation
│   │
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── kg_core.py           # Core KG functionality
│   │   ├── entity_extraction.py # Regulation entity extraction
│   │   ├── relationship_graph.py # Network analysis
│   │   └── community_detection.py # Dynamic communities
│   │
│   └── search/
│       ├── __init__.py
│       ├── search_engine.py     # Main search orchestrator
│       ├── hybrid_search.py     # Metadata + semantic search
│       ├── query_analyzer.py    # Advanced query understanding
│       └── context_manager.py   # Conversation context
│
├── research/
│   ├── __init__.py
│   ├── team_manager.py          # Research team orchestration
│   ├── personas.py              # Researcher persona definitions
│   ├── consensus_builder.py    # Multi-researcher consensus
│   └── adaptive_learning.py    # Performance tracking
│
├── generation/
│   ├── __init__.py
│   ├── llm_generator.py         # LLM response generation
│   ├── prompt_builder.py        # Context-aware prompts
│   └── streaming.py             # Streaming response handler
│
├── conversation/
│   ├── __init__.py
│   ├── history_manager.py       # Conversation state
│   └── export/
│       ├── __init__.py
│       ├── markdown_exporter.py
│       ├── json_exporter.py
│       ├── html_exporter.py
│       └── pdf_exporter.py      # Future: PDF export
│
├── ui/
│   ├── __init__.py
│   ├── gradio_interface.py      # Main Gradio UI
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat_interface.py
│   │   ├── settings_panel.py
│   │   └── export_panel.py
│   └── styles/
│       └── custom_css.py
│
├── agents/                       # Future: Agentic workflows
│   ├── __init__.py
│   ├── tool_registry.py
│   └── agent_executor.py
│
├── utils/
│   ├── __init__.py
│   ├── logging_config.py        # Centralized logging
│   ├── error_handlers.py        # Error recovery
│   ├── validators.py            # Config validation
│   ├── memory_utils.py          # Cache & memory management
│   └── metrics.py               # Performance tracking
│
├── tests/
│   ├── __init__.py
│   ├── unit/                    # Unit tests per module
│   ├── integration/             # End-to-end tests
│   └── fixtures/                # Test data
│
├── scripts/
│   ├── initialize_system.py    # Setup script
│   ├── run_server.py           # Production server
│   └── benchmarks.py           # Performance testing
│
├── logs/                        # Log files (gitignored)
├── exports/                     # Exported conversations
├── requirements.txt
├── setup.py
├── README.md
└── .env.example                 # Environment variables template
```

---
