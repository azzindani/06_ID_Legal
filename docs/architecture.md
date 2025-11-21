# Architecture

## System Overview

The Indonesian Legal RAG System uses a multi-stage, multi-persona approach for accurate legal information retrieval.

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Detector  │ ← Analyzes query type, entities, complexity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hybrid Search   │ ← Semantic + Keyword search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Multi-Persona   │ ← Senior, Specialist, Junior researchers
│ Research        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Consensus       │ ← Cross-validation, Devil's Advocate
│ Builder         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reranker        │ ← Final relevance scoring
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generation      │ ← LLM response with citations
│ Engine          │
└────────┬────────┘
         │
         ▼
    Response
```

## Core Components

### 1. Query Detection (`core/search/query_detection.py`)
- Classifies query type (sanctions, definition, procedure, etc.)
- Extracts legal entities (regulations, articles, years)
- Determines team composition based on query complexity

### 2. Hybrid Search (`core/search/hybrid_search.py`)
- Combines semantic search (embeddings) with keyword search (BM25)
- Uses FAISS for fast similarity search
- Configurable weighting between search types

### 3. Knowledge Graph (`core/knowledge_graph/`)
- Entity extraction and relationship mapping
- Graph-based scoring for authority hierarchy
- Community detection for related documents

### 4. Multi-Stage Research (`core/search/stages_research.py`)
- Multiple researcher personas with different focus areas
- Staged retrieval (initial scan → deep analysis → verification)
- Parallel processing for efficiency

### 5. Consensus Building (`core/search/consensus.py`)
- Cross-validation between researcher results
- Devil's advocate challenges for accuracy
- Agreement scoring and conflict resolution

### 6. Generation Engine (`core/generation/`)
- LLM-based answer generation
- Citation formatting and validation
- Response quality scoring

## Provider Architecture

```
┌─────────────────────────────────┐
│       Provider Factory          │
└────────────┬────────────────────┘
             │
    ┌────────┼────────┬────────┬────────┐
    │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│Local │ │OpenAI│ │Claude│ │Gemini│ │OpenRouter│
└──────┘ └──────┘ └──────┘ └──────┘ └──────────┘
```

## Data Flow

1. **Input Processing**: Query → Query Detector → Analysis
2. **Retrieval**: Analysis → Search → Research → Consensus → Rerank
3. **Generation**: Retrieved docs → Prompt → LLM → Response
4. **Output**: Response → Citation → Validation → User

## Scalability

- **Horizontal**: Multiple workers, load balancing
- **Vertical**: GPU acceleration, quantization
- **Caching**: Context cache, embedding cache
