# KG-Enhanced Indonesian Legal RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system for Indonesian legal documents, featuring Knowledge Graph enhancement, multi-researcher team simulation, and LangGraph orchestration.

## Overview

This system provides intelligent legal consultation by combining:
- **Semantic Search** - Qwen3 embeddings for deep understanding
- **Knowledge Graph** - Entity relationships and legal hierarchy
- **Multi-Researcher Simulation** - Team of specialized AI researchers
- **Consensus Building** - Cross-validation and agreement scoring
- **LLM Generation** - DeepSeek-based response generation

## Directory Structure

```
06_ID_Legal/
├── config.py                    # Centralized configuration
├── model_manager.py             # Model loading and management
├── logger_utils.py              # Centralized logging system
├── requirements.txt             # Dependencies
├── .env.example                 # Environment variables template
├── Kaggle_Demo.ipynb            # Original monolithic reference
│
├── core/
│   ├── search/
│   │   ├── hybrid_search.py         # Hybrid semantic + keyword search
│   │   ├── stages_research.py       # Multi-stage research with quality degradation
│   │   ├── consensus.py             # Consensus building among researchers
│   │   ├── reranking.py             # Final reranking with reranker model
│   │   ├── query_detection.py       # Query type analysis and enhancement
│   │   └── langgraph_orchestrator.py # LangGraph workflow orchestration
│   │
│   └── generation/
│       ├── llm_engine.py            # LLM model management and generation
│       ├── generation_engine.py     # Response generation orchestration
│       ├── prompt_builder.py        # Context-aware prompt construction
│       ├── response_validator.py    # Response validation
│       └── citation_formatter.py    # Legal citation formatting
│
└── loader/
    └── dataloader.py            # HuggingFace dataset loading with KG indexes
```

## Key Components

### 1. Configuration (`config.py`)

Centralized configuration with environment variable support:

```python
from config import (
    DATASET_NAME,           # "Azzindani/ID_REG_DB_2510"
    EMBEDDING_MODEL,        # "Qwen/Qwen3-Embedding-0.6B"
    RERANKER_MODEL,         # "Qwen/Qwen3-Reranker-0.6B"
    LLM_MODEL,              # "Azzindani/Deepseek_ID_Legal_Preview"
    DEFAULT_CONFIG,         # System defaults
    DEFAULT_SEARCH_PHASES,  # Search phase configuration
    RESEARCH_TEAM_PERSONAS  # AI researcher personas
)
```

### 2. Data Loader (`loader/dataloader.py`)

SQLite-based dataset loader with:
- HuggingFace integration
- Compressed embeddings
- TF-IDF vectors
- KG index building (entities, cross-references, domains)

```python
from loader.dataloader import EnhancedKGDatasetLoader
from config import DATASET_NAME, EMBEDDING_DIM

loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
loader.load_from_huggingface(progress_callback=print)
stats = loader.get_statistics()
```

### 3. Model Manager (`model_manager.py`)

Centralized model loading with retry logic:

```python
from model_manager import get_model_manager, load_models

manager = get_model_manager()
embedding_model = manager.load_embedding_model()
reranker_model = manager.load_reranker_model()

# Or use convenience function
embedding_model, reranker_model = load_models()
```

### 4. LangGraph RAG Orchestrator (`core/search/langgraph_orchestrator.py`)

State machine-based workflow:

```python
from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator

orchestrator = LangGraphRAGOrchestrator(
    data_loader=loader,
    embedding_model=embedding_model,
    reranker_model=reranker_model,
    config=config
)

# Run complete workflow
result = orchestrator.run(
    query="Apa sanksi pelanggaran UU Ketenagakerjaan?",
    conversation_history=[]
)

# Access results
final_results = result['final_results']
metadata = result['metadata']
```

### 5. LLM Engine (`core/generation/llm_engine.py`)

LLM generation with streaming support:

```python
from core.generation.llm_engine import get_llm_engine

llm = get_llm_engine(config)
llm.load_model()

# Synchronous generation
result = llm.generate(prompt, max_new_tokens=2048)

# Streaming generation
for chunk in llm.generate_stream(prompt):
    if chunk['done']:
        break
    print(chunk['token'], end='')
```

## Search Phases

The system uses a multi-phase search approach mimicking human research:

| Phase | Candidates | Semantic Threshold | Keyword Threshold | Description |
|-------|------------|-------------------|-------------------|-------------|
| Initial Scan | 400 | 0.20 | 0.06 | Quick broad scan |
| Focused Review | 150 | 0.35 | 0.12 | Review promising candidates |
| Deep Analysis | 60 | 0.45 | 0.18 | Contextual analysis |
| Verification | 30 | 0.55 | 0.22 | Final cross-checking |
| Expert Review | 45 | 0.50 | 0.20 | Complex cases (optional) |

### Quality Degradation

Each round applies quality multiplier to thresholds:
- Initial Quality: 0.95
- Degradation Rate: 0.1 per round
- Minimum Quality: 0.5

## Research Team Personas

Five specialized AI researchers with different strengths:

| Persona | Experience | Specialties | Accuracy Bonus |
|---------|------------|-------------|----------------|
| Senior Legal Researcher | 15 years | Constitutional law, precedents | +15% |
| Junior Legal Researcher | 3 years | Digital search, broad coverage | 0% |
| KG Specialist | 8 years | Knowledge graphs, entities | +10% |
| Procedural Expert | 12 years | Administrative law, procedures | +8% |
| Devil's Advocate | 10 years | Critical analysis, edge cases | +12% |

Team composition adapts to query type:
- **Specific Article**: Senior, KG Specialist, Devil's Advocate
- **Procedural**: Procedural Expert, Junior, Senior
- **Definitional**: Senior, KG Specialist, Junior
- **Sanctions**: Senior, Procedural Expert, Devil's Advocate

## Query Types

Automatic detection based on indicators:

| Type | Indicators | Priority Focus |
|------|------------|----------------|
| Specific Article | pasal, ayat, huruf, angka | Authority hierarchy |
| Procedural | prosedur, tata cara, persyaratan | Legal completeness |
| Definitional | definisi, pengertian, dimaksud dengan | Authority hierarchy |
| Sanctions | sanksi, pidana, denda, hukuman | KG relationships |
| General | (default) | Balanced approach |

## Installation

1. **Clone and setup environment:**
```bash
git clone <repository>
cd 06_ID_Legal
pip install -r requirements.txt
```

2. **Configure environment variables (optional):**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Key dependencies:**
- torch
- transformers
- langgraph
- gradio
- datasets
- scipy
- igraph
- python-louvain

## Configuration

### Environment Variables

```bash
# Dataset
DATASET_NAME=Azzindani/ID_REG_DB_2510
HF_TOKEN=your_token  # if needed

# Models
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
LLM_MODEL=Azzindani/Deepseek_ID_Legal_Preview

# System
DEVICE=cuda
MAX_LENGTH=32768
LOG_DIR=logs
```

### Runtime Configuration

```python
from config import get_default_config

config = get_default_config()
config['final_top_k'] = 5
config['temperature'] = 0.7
config['max_new_tokens'] = 2048
config['consensus_threshold'] = 0.6
```

## Logging

Centralized logging system with structured output:

```python
from logger_utils import get_logger

logger = get_logger("MyModule")
logger.info("Operation started", {"key": "value"})
logger.success("Operation completed", {"results": 10})
logger.error("Operation failed", {"error": str(e)})
```

## Knowledge Graph Features

### Scoring Components

- **Direct Match**: 1.0 weight
- **One-hop Relations**: 0.8 weight
- **Two-hop Relations**: 0.6 weight
- **Cross-references**: 0.6 weight
- **Domain Match**: 0.5 weight
- **Legal Actions**: 0.7 weight (obligations, prohibitions, permissions)
- **Sanctions**: 0.8 weight

### KG Indexes

- Entity lookup (concepts, terms)
- Cross-reference lookup (citations between regulations)
- Domain classification
- Authority hierarchy
- Temporal relevance

## API Usage Example

```python
from config import get_default_config
from model_manager import load_models
from loader.dataloader import EnhancedKGDatasetLoader
from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator
from core.generation.llm_engine import get_llm_engine

# Initialize
config = get_default_config()
config['search_phases'] = DEFAULT_SEARCH_PHASES

# Load models
embedding_model, reranker_model = load_models()

# Load dataset
loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
loader.load_from_huggingface()

# Create orchestrator
orchestrator = LangGraphRAGOrchestrator(
    data_loader=loader,
    embedding_model=embedding_model,
    reranker_model=reranker_model,
    config=config
)

# Run query
result = orchestrator.run("Apa definisi pekerja menurut UU Ketenagakerjaan?")

# Get top results
for doc in result['final_results'][:3]:
    print(f"Score: {doc['final_score']:.4f}")
    print(f"Regulation: {doc['record']['regulation_type']} No. {doc['record']['regulation_number']}/{doc['record']['year']}")
    print(f"Content: {doc['record']['content'][:200]}...")
    print()

# Generate LLM response
llm = get_llm_engine(config)
llm.load_model()

# Build context from results
context = "\n\n".join([
    f"[{i+1}] {r['record']['regulation_type']} No. {r['record']['regulation_number']}/{r['record']['year']}\n{r['record']['content']}"
    for i, r in enumerate(result['final_results'][:3])
])

prompt = f"{SYSTEM_PROMPT}\n\nKonteks:\n{context}\n\nPertanyaan: {query}\n\nJawaban:"
response = llm.generate(prompt)
print(response['generated_text'])
```

## System Prompt

The default system prompt for LLM generation:

```
Anda adalah asisten AI yang ahli di bidang hukum Indonesia. Anda dapat membantu konsultasi hukum, menjawab pertanyaan, dan memberikan analisis berdasarkan peraturan perundang-undangan yang relevan. Untuk setiap respons, Anda harus berfikir dan menjawab dengan Bahasa Indonesia, serta gunakan format: <think> ... </think> Tuliskan jawaban akhir secara jelas, ringkas, profesional, dan berempati jika diperlukan. Gunakan bahasa hukum yang mudah dipahami. Sertakan referensi hukum Indonesia yang relevan. Selalu rekomendasikan konsultasi dengan ahli hukum untuk keputusan final.
```

## Performance Notes

- **Embedding Model**: ~600M parameters, requires GPU for efficient inference
- **Reranker Model**: ~600M parameters, used for final result ordering
- **LLM Model**: DeepSeek-based, supports streaming generation
- **Dataset**: ~100K+ regulation chunks with KG metadata

### Memory Optimization

- Lazy JSON parsing for KG data
- Chunked dataset loading (5000 records per chunk)
- Compressed embeddings (float16)
- Sparse TF-IDF matrices

## License

[Specify license]

## Contributing

[Contribution guidelines]

## Acknowledgments

- HuggingFace for model hosting
- Qwen team for embedding/reranker models
- DeepSeek for LLM foundation
