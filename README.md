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
│   ├── model_config.py          ✅# Model paths, HF tokens
│   ├── search_config.py         ✅# Search phases, team personas
│   ├── llm_config.py            ✅# LLM generation params
│   └── app_config.py            # Gradio UI, export settings
│
├── core/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding_model.py   ✅# Embedding model wrapper
│   │   ├── reranker_model.py    # Reranker wrapper
│   │   └── llm_model.py         # LLM wrapper (local + API)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py    ✅# HuggingFace dataset loading
│   │   └── preprocessing.py     ✅# Data cleaning, validation
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
│       ├── search_engine.py     ✅# Main search orchestrator
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
│   ├── logging_config.py        ✅# Centralized logging
│   ├── error_handlers.py        # Error recovery
│   ├── validators.py            # Config validation
│   ├── memory_utils.py          ✅# Cache & memory management
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

# 📝 **Prompt Template for AI-Assisted Modular Migration**

Here's a comprehensive prompt template you can use with AI assistants to build each component systematically.

---

## 🎯 **Master Prompt Template**

```markdown
# Context Setting Prompt (Use ONCE at the start)

I'm migrating a monolithic Indonesian Legal RAG system into a modular architecture. 

**Original Code Location:** [Attach your notebook/file]

**Target Architecture:**
- Modular, production-ready Python package
- Independent, testable components
- Logging in every function
- Support for model swapping (local/API)
- Future-proof for agentic workflows

**Directory Structure:**
```
indonesian_legal_rag/
├── config/          # Configuration files
├── core/            # Core functionality (models, data, KG, search)
├── research/        # Research team simulation
├── generation/      # LLM generation
├── conversation/    # History & export
├── ui/              # Gradio interface
├── agents/          # Future: Tools
├── utils/           # Logging, error handling
└── tests/           # Unit & integration tests
```

**Logging Standard:**
Every function must include:
```python
from utils.logging_config import get_logger
logger = get_logger(__name__)

def function():
    logger.info("Function started")
    try:
        # logic
        logger.debug("Key step completed")
        logger.info("Function completed successfully")
    except Exception as e:
        logger.error(f"Function failed: {e}", exc_info=True)
        raise
```

**My Task:** I need help building component: [COMPONENT NAME]
```

---

## 🔧 **Component-Specific Prompts**

Copy the master prompt above, then append ONE of these specific component prompts:

---

### **PROMPT 1: Logging Infrastructure**

```markdown
**Component:** `utils/logging_config.py`

**Requirements:**
1. Create a centralized logging system with:
   - Console output (INFO level, simple format)
   - File output (DEBUG level, detailed format with timestamp, module, function, line number)
   - Rotating file handler (10MB max, 5 backups)
   - UTF-8 encoding for Indonesian text
   - Separate logger for each module

2. Function signature:
   ```python
   def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger
   def get_logger(module_name: str) -> logging.Logger
   ```

3. Log format:
   - File: `YYYY-MM-DD HH:MM:SS | module_name | LEVEL | function:line | message`
   - Console: `HH:MM:SS | LEVEL | message`

4. Auto-create `logs/` directory
5. Daily log files: `logs/{module_name}_YYYYMMDD.log`

**Deliverable:** Complete, production-ready `utils/logging_config.py` with docstrings.
```

---

### **PROMPT 2: Configuration System**

```markdown
**Component:** `config/model_config.py`

**Requirements:**
1. Create dataclass-based configuration supporting:
   - Environment variables (HF_TOKEN, API keys)
   - Model names (embedding, reranker, LLM)
   - Device settings (CUDA/CPU auto-detection)
   - API inference toggle (future-proof)
   - Batch sizes and max lengths

2. Include these models:
   - Embedding: `Qwen/Qwen3-Embedding-0.6B`
   - Reranker: `Qwen/Qwen3-Reranker-0.6B`
   - LLM: `Azzindani/Deepseek_ID_Legal_Preview`
   - Dataset: `Azzindani/ID_REG_KG_2510`

3. Add `__post_init__` validation:
   - Check API endpoint if `llm_use_api=True`
   - Validate paths exist
   - Log configuration summary

4. Support both local and API inference modes

**Example Usage:**
```python
from config.model_config import MODEL_CONFIG
print(MODEL_CONFIG.embedding_model_name)
```

**Deliverable:** Complete `config/model_config.py` with type hints and docstrings.
```

---

### **PROMPT 3: Embedding Model Wrapper**

```markdown
**Component:** `core/models/embedding_model.py`

**Original Code Reference:**
```python
# Lines 1500-1600 in original notebook:
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, padding_side='left')
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL, attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map="auto")
embedding_model = embedding_model.eval()
```

**Requirements:**
1. Create `EmbeddingModel` class with methods:
   - `__init__(config)`: Store config, don't load model yet
   - `load()`: Load tokenizer + model, detect embedding dimension
   - `embed(texts: Union[str, List[str]]) -> torch.Tensor`: Generate embeddings
   - `unload()`: Free GPU memory
   - `_get_embedding_dim()`: Auto-detect dimension via test inference

2. Features:
   - Lazy loading (load on demand)
   - Support Flash Attention 2 with fallback
   - L2-normalization option
   - Batch processing
   - Proper device management

3. Logging:
   - Log model loading steps
   - Log embedding dimension detection
   - Log batch sizes
   - Log errors with full traceback

4. Error handling:
   - Catch Flash Attention failures → fallback to standard
   - Validate model loaded before embedding
   - Handle CUDA OOM gracefully

**Deliverable:** Complete `core/models/embedding_model.py` with docstrings and type hints.

**Integration Example:**
```python
from core.models.embedding_model import EmbeddingModel
from config.model_config import MODEL_CONFIG

model = EmbeddingModel(MODEL_CONFIG)
model.load()
embeddings = model.embed(["Hukum Indonesia"])
print(embeddings.shape)  # torch.Size([1, 896])
```
```

---

### **PROMPT 4: Dataset Loader**

```markdown
**Component:** `core/data/dataset_loader.py`

**Original Code Reference:**
```python
# Lines 1800-2300 in original notebook:
class EnhancedKGDatasetLoader:
    def load_from_huggingface(self, progress_callback=None):
        # Loads dataset from HF, processes embeddings, TF-IDF
        # Builds KG indexes (entities, cross-refs, domains)
```

**Requirements:**
1. Migrate `EnhancedKGDatasetLoader` class with:
   - Streaming dataset loading (memory efficient)
   - Chunked processing (1000 records at a time)
   - Embedding extraction from parquet
   - TF-IDF vector handling
   - KG index building (6 indexes: entities, cross_refs, domains, clusters, legal_actions, sanctions)

2. Add these methods:
   - `load_from_huggingface()`: Load dataset with progress tracking
   - `get_statistics()`: Return dataset stats (total records, KG enhancement rate)
   - `_build_enhanced_kg_indexes()`: Build lookup dictionaries
   - `_create_record(row, idx)`: Convert HF row to record dict

3. Memory optimization:
   - Use lazy JSON parsing (store strings, parse on demand)
   - Aggressive garbage collection after chunks
   - Clear CUDA cache between chunks

4. Logging:
   - Log each chunk processed
   - Log total records loaded
   - Log KG enhancement rate
   - Log memory usage after loading

**Key Indexes to Build:**
```python
self.kg_entities_lookup = {}           # doc_id -> entities JSON string
self.kg_cross_references_lookup = {}   # doc_id -> cross_refs JSON string
self.kg_domains_lookup = {}            # doc_id -> domains JSON string
self.authority_index = {}              # authority_tier -> [doc_indices]
self.temporal_index = {}               # temporal_tier -> [doc_indices]
```

**Deliverable:** Complete `core/data/dataset_loader.py` with progress callbacks and memory management.
```

---

### **PROMPT 5: Knowledge Graph Core**

```markdown
**Component:** `core/knowledge_graph/kg_core.py`

**Original Code Reference:**
```python
# Lines 2300-2800 in original notebook:
class EnhancedKnowledgeGraph:
    def extract_entities_from_text(self, text): ...
    def calculate_enhanced_kg_score(self, query_entities, record, query_type): ...
    def get_parsed_kg_data(self, doc_id, data_type): ...
```

**Requirements:**
1. Migrate `EnhancedKnowledgeGraph` class with caching:
   - Parse JSON lazily (don't parse until needed)
   - Cache parsed results (max 1000 entries)
   - Track cache hit rate

2. Core methods:
   - `extract_entities_from_text(text) -> List[Tuple]`: Extract Indonesian regulations
   - `calculate_enhanced_kg_score(query_entities, record, query_type) -> float`: Compute KG relevance
   - `get_parsed_kg_data(doc_id, data_type) -> dict`: Lazy JSON parsing with cache
   - `extract_regulation_references_with_confidence(text) -> List[dict]`: Confidence-based extraction
   - `get_cache_stats() -> dict`: Cache performance metrics

3. Regulation patterns to detect:
   ```python
   REGULATION_TYPES = ['undang-undang', 'peraturan pemerintah', 'peraturan presiden', 
                       'peraturan menteri', 'keputusan presiden']
   
   Formats: "UU No. 13 Tahun 2003", "PP 41/2009", "Perpres 16 Tahun 2018"
   ```

4. KG scoring weights:
   ```python
   KG_WEIGHTS = {
       'direct_match': 1.0,
       'one_hop': 0.8,
       'cross_reference': 0.6,
       'domain_match': 0.5,
       'hierarchy_boost': 0.5
   }
   ```

5. Logging:
   - Log entity extraction counts
   - Log cache hit/miss rates
   - Log KG score calculations
   - Log regulation references found

**Deliverable:** Complete `core/knowledge_graph/kg_core.py` with caching and performance tracking.
```

---

### **PROMPT 6: Query Analyzer**

```markdown
**Component:** `core/search/query_analyzer.py`

**Original Code Reference:**
```python
# Lines 3500-3800 in original notebook:
class AdvancedQueryAnalyzer:
    def analyze_query(self, query: str) -> Dict[str, Any]:
        # Determines search strategy: keyword_first, semantic_first, hybrid_balanced
```

**Requirements:**
1. Migrate `AdvancedQueryAnalyzer` with these features:
   - Detect specific legal phrases (e.g., "cipta kerja", "hak cipta")
   - Identify common law names (e.g., "kepabeanan", "ketenagakerjaan")
   - Analyze query structure (conceptual vs. specific)
   - Return search strategy with confidence score

2. Search strategies:
   - `keyword_first`: Specific legal terms detected (e.g., "tentang cipta kerja")
   - `semantic_first`: Conceptual questions (e.g., "bagaimana prosedur...")
   - `hybrid_balanced`: Mixed or unclear intent
   - `metadata_first`: Exact regulation reference (e.g., "UU 13/2003")

3. Key phrases dictionary:
   ```python
   LEGAL_PHRASES = {
       'cipta kerja': {'priority': 0.95, 'context': 'job creation law'},
       'hak cipta': {'priority': 0.95, 'context': 'copyright'},
       'tenaga kerja': {'priority': 0.90, 'context': 'labor'},
       'bea cukai': {'priority': 0.90, 'context': 'customs'}
   }
   ```

4. Analysis output:
   ```python
   {
       'search_strategy': 'keyword_first',
       'confidence': 0.85,
       'key_phrases': [{'phrase': 'cipta kerja', 'priority': 0.95}],
       'law_name_detected': True,
       'reasoning': 'Specific legal phrase detected',
       'keyword_boost': 0.40,
       'semantic_boost': 0.10
   }
   ```

5. Logging:
   - Log detected strategy and confidence
   - Log key phrases found
   - Log reasoning for strategy selection

**Deliverable:** Complete `core/search/query_analyzer.py` with phrase detection and strategy selection.
```

---

### **PROMPT 7: Hybrid Search Engine**

```markdown
**Component:** `core/search/hybrid_search.py`

**Original Code Reference:**
```python
# Lines 4000-4500 in original notebook:
def hybrid_search_strategy(self, query, query_type, config, progress_callback=None):
    # Executes metadata-first or semantic-first search based on query analysis
```

**Requirements:**
1. Create `HybridSearch` class with two search paths:
   - **Metadata-first**: For exact regulation references (e.g., "UU 13/2003")
   - **Semantic-first**: For conceptual queries (e.g., "hak pekerja")

2. Core methods:
   - `search(query, query_analysis, top_k) -> List[dict]`: Main search orchestrator
   - `_metadata_first_search(query, regulation_ref) -> List[dict]`: Direct metadata lookup
   - `_semantic_first_search(query, query_type) -> List[dict]`: Semantic + keyword search
   - `_apply_regulation_filter(candidates, regulation_filter) -> List[dict]`: Filter by regulation

3. Metadata search (STRICT filtering):
   ```python
   # MUST match ALL three:
   # 1. Regulation type (e.g., "undang-undang")
   # 2. Number (e.g., "13")
   # 3. Year (e.g., "2003")
   
   # Set perfect score (1.0) for exact matches
   ```

4. Semantic search phases:
   ```python
   PHASES = ['initial_scan', 'focused_review', 'deep_analysis', 'verification']
   # Each phase has: candidates count, semantic threshold, keyword threshold
   ```

5. Logging:
   - Log search strategy selected
   - Log metadata matches found
   - Log phase execution (candidates per phase)
   - Log final result count

**Deliverable:** Complete `core/search/hybrid_search.py` with fallback mechanisms and detailed logging.
```

---

### **PROMPT 8: Research Team Manager**

```markdown
**Component:** `research/team_manager.py`

**Original Code Reference:**
```python
# Lines 4500-5500 in original notebook:
def parallel_legal_research(self, query, query_type, config):
    # Assembles research team, conducts parallel research, builds consensus
```

**Requirements:**
1. Create `ResearchTeamManager` class to orchestrate:
   - Team assembly based on query type
   - Parallel individual research
   - Cross-validation between researchers
   - Devil's advocate review
   - Consensus building

2. Team compositions (from `config/search_config.py`):
   ```python
   QUERY_TEAM_COMPOSITIONS = {
       'specific_article': ['senior_legal_researcher', 'specialist_researcher', 'devils_advocate'],
       'procedural': ['procedural_expert', 'junior_legal_researcher', 'senior_legal_researcher'],
       'sanctions': ['senior_legal_researcher', 'procedural_expert', 'devils_advocate']
   }
   ```

3. Core methods:
   - `assemble_team(query_type, team_size) -> List[str]`: Select researchers
   - `conduct_parallel_research(query, team_members) -> dict`: Execute research
   - `cross_validate(individual_results) -> dict`: Validate findings
   - `devils_advocate_review(results) -> dict`: Challenge assumptions
   - `build_consensus(individual_results, consensus_threshold) -> List[dict]`: Final results

4. Consensus algorithm:
   ```python
   # Weight by:
   # 1. Researcher experience (years)
   # 2. Researcher accuracy bonus
   # 3. Number of supporting researchers
   # 4. Cross-validation agreement
   
   final_score = weighted_average + consensus_bonus + cross_validation_bonus
   ```

5. Logging:
   - Log team assembly
   - Log each researcher's findings
   - Log cross-validation results
   - Log devil's advocate challenges
   - Log final consensus

**Deliverable:** Complete `research/team_manager.py` with parallel execution and consensus logic.
```

---

### **PROMPT 9: LLM Generator with API Support**

```markdown
**Component:** `generation/llm_generator.py`

**Original Code Reference:**
```python
# Lines 6000-6500 in original notebook:
class KGEnhancedLLMGenerator:
    def generate_with_kg(self, query, results, query_type, config):
        # Formats context, generates response with streaming
```

**Requirements:**
1. Create `LLMGenerator` class supporting BOTH:
   - **Local inference** (transformers)
   - **API inference** (OpenAI-compatible APIs)

2. Abstraction layer:
   ```python
   class LLMGenerator:
       def __init__(self, config):
           if config.llm_use_api:
               self.backend = APIBackend(config)
           else:
               self.backend = LocalBackend(config)
       
       def generate(self, prompt, stream=True):
           return self.backend.generate(prompt, stream)
   ```

3. Context formatting (anti-hallucination):
   ```
   CRITICAL: You MUST base answers ONLY on provided "Legal References". 
   DO NOT cite regulations not in the list.
   
   SYNTHESIS: For conceptual questions, combine facts logically.
   FACT-FINDING: For specific facts, state clearly if not found.
   ```

4. Streaming support:
   - Extract `<think>` blocks
   - Stream final answer progressively
   - Handle both local (TextIteratorStreamer) and API (SSE) streams

5. Logging:
   - Log prompt length
   - Log generation parameters
   - Log streaming chunks (debug level)
   - Log final response length
   - Log API calls (if API mode)

**Deliverable:** Complete `generation/llm_generator.py` with local/API abstraction and streaming.
```

---

### **PROMPT 10: Export System**

```markdown
**Component:** `conversation/export/`

**Requirements:**
1. Create separate exporter classes:
   - `markdown_exporter.py`: Markdown with collapsible sections
   - `json_exporter.py`: Structured JSON with full metadata
   - `html_exporter.py`: Styled HTML with CSS (tables, collapsible)
   - `pdf_exporter.py`: PDF via `weasyprint` or `reportlab`

2. Base exporter interface:
   ```python
   class BaseExporter(ABC):
       @abstractmethod
       def export(self, conversation_history: List[dict], options: dict) -> str:
           pass
       
       def save_to_file(self, content: str, filename: str) -> Path:
           pass
   ```

3. Export options:
   ```python
   {
       'include_thinking': True,
       'include_metadata': True,
       'include_research_process': True,
       'include_full_content': False,
       'format_tables': True  # For Markdown/HTML
   }
   ```

4. Markdown exporter features:
   - Collapsible sections for metadata
   - Properly render tables
   - Include legal references
   - Timestamp and version info

5. HTML exporter features:
   - Responsive CSS (zoom-friendly)
   - Syntax highlighting for code
   - Collapsible details tags
   - Print-friendly styles

6. Logging:
   - Log export format selected
   - Log file size
   - Log export duration
   - Log any rendering errors

**Deliverable:** Complete export system with 4 formats (Markdown, JSON, HTML, PDF) and unified interface.
```

---

## 🎯 **How to Use These Prompts**

### **Step-by-Step Workflow:**

1. **Start a new chat** with your AI assistant (Claude, ChatGPT, etc.)

2. **Send the "Context Setting Prompt"** (the master prompt at the top)

3. **Send ONE component prompt** (e.g., PROMPT 1 for logging)

4. **Review the generated code**:
   - Check it compiles
   - Verify logging is present
   - Test basic functionality

5. **Ask follow-up questions** if needed:
   ```
   "Can you add error handling for X?"
   "How do I test this component in isolation?"
   "Can you show an example usage?"
   ```

6. **Move to next component** - Send PROMPT 2, then PROMPT 3, etc.

---

## 📋 **Recommended Order**

```
PROMPT 1  (Logging)           → Foundation, needed by all
PROMPT 2  (Config)            → Needed by models
PROMPT 3  (Embedding Model)   → First model wrapper
PROMPT 4  (Dataset Loader)    → Needed by search
PROMPT 5  (Knowledge Graph)   → Needed by search
PROMPT 6  (Query Analyzer)    → Needed by search
PROMPT 7  (Hybrid Search)     → Core search logic
PROMPT 8  (Research Team)     → Advanced feature
PROMPT 9  (LLM Generator)     → Generation component
PROMPT 10 (Export System)     → User-facing feature
```

---

## 💡 **Pro Tips**

1. **Test each component** before moving to next
2. **Ask for unit tests**: "Can you write pytest tests for this class?"
3. **Request usage examples**: "Show me how to use this in the main pipeline"
4. **Validate integration**: "How does this connect to [previous component]?"
5. **Ask for error scenarios**: "What edge cases should I handle?"

---

## 🚀 **Quick Start Command**

Copy this entire message, then in a NEW chat with AI:

```
[Paste Master Prompt]

[Paste PROMPT 1]

Please generate the complete, production-ready code for this component.
Include:
- Full docstrings
- Type hints
- Error handling
- Logging
- Usage example
```

---

Would you like me to:
1. ✅ Create **additional specialized prompts** (e.g., for testing, deployment)?
2. ✅ Provide a **validation checklist** for each component?
3. ✅ Create a **GitHub project template** with issues for each component?
4. ✅ Generate **pytest test templates** for each component?

Let me know what would be most helpful! 🎯