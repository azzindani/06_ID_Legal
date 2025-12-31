# Production Deployment Guide - Indonesian Legal RAG System

## System Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Server | FastAPI + Uvicorn | RESTful API with SSE streaming |
| RAG Pipeline | LangGraph + Custom Orchestrator | Multi-agent retrieval with consensus |
| Embedding | Qwen3-Embedding-0.6B | Semantic document vectorization |
| Reranker | Qwen3-Reranker-0.6B | Cross-encoder relevance scoring |
| LLM | Local (Deepseek) / OpenRouter | Answer generation |
| UI | Gradio | Web interface |

---

## Infrastructure Requirements

### Minimum Production (Single Node)
```
CPU: 8-core AMD/Intel
RAM: 32GB
GPU: NVIDIA RTX 3090/4090 (24GB VRAM)
Storage: 500GB SSD
Network: 1Gbps
```

### Recommended Production (Multi-Node)
```
API Nodes: 2-4x (8-core, 16GB RAM each)
GPU Nodes: 2-4x (RTX 4090 or A100)
Load Balancer: NGINX/HAProxy
Database: PostgreSQL (sessions) + Redis (cache)
Storage: NFS/S3 for shared models
```

### ChatGPT-Scale Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ API-1   │   │ API-2   │   │ API-N   │  (Stateless API pods)
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
   ┌──────────────────────────────────────────────────────┐
   │              Message Queue (Redis/RabbitMQ)          │
   └──────────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ GPU-1   │   │ GPU-2   │   │ GPU-N   │  (LLM inference workers)
   └─────────┘   └─────────┘   └─────────┘
```

---

## Deployment Modes

### 1. Retrieval-Only Mode (No GPU Required)
```bash
python -m api.server --llm-provider none --host 0.0.0.0 --port 8000
```

### 2. Full Local Mode (GPU Required)
```bash
python -m api.server --llm-provider local --host 0.0.0.0 --port 8000
```

### 3. Hybrid Mode (Valve Architecture)
```bash
python -m api.server --llm-provider local
# Switch at runtime: POST /api/v1/llm/config {"provider": "openrouter"}
```

---

## Environment Configuration (.env)

```bash
# Core
DATASET_NAME=Azzindani/ID_REG_DB_2510
HF_TOKEN=your_token

# Models
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
LLM_MODEL=Azzindani/Deepseek_ID_Legal_Preview

# Local models
USE_LOCAL_MODELS=false
LOCAL_MODEL_DIR=/opt/legal-rag/models

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-xxxxx
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free

# Performance
MAX_LENGTH=32768
ENABLE_CONTEXT_CACHE=true
```

---

## Security Checklist

- [ ] API key authentication middleware
- [ ] Rate limiting (60 req/min recommended)
- [ ] HTTPS via reverse proxy
- [ ] CORS configuration
- [ ] Secrets management (vault)
- [ ] Container isolation (Docker)

---

## Monitoring Metrics

- `request_latency_ms` (p50, p95, p99)
- `gpu_memory_usage_gb`
- `cache_hit_rate`
- `error_rate`

**Stack**: Prometheus + Grafana + ELK

---

## Performance Benchmarks

| Metric | Single GPU | Multi-GPU (4x) |
|--------|------------|----------------|
| Concurrent Users | 10-20 | 50-100 |
| Queries/sec | 2-5 | 10-20 |
| Avg Latency | 3-5s | 2-4s |

---

## Deployment Checklist

- [ ] Configure `.env` with production values
- [ ] Set up HTTPS via reverse proxy
- [ ] Enable API authentication
- [ ] Configure rate limiting
- [ ] Set up monitoring
- [ ] Configure log aggregation
- [ ] Test failover scenarios
- [ ] Load test with expected traffic
