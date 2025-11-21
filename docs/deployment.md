# Deployment Guide

## Local Development

### Prerequisites
- Python 3.9+
- 16GB+ RAM (for local LLM)
- GPU with 8GB+ VRAM (optional, for faster inference)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd 06_ID_Legal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize system
python scripts/initialize_system.py

# Run Gradio UI
python ui/gradio_app.py
```

## Docker Deployment

### Build Image

```bash
docker build -t id-legal-rag:latest .
```

### Run Container

```bash
# CPU only
docker run -p 7860:7860 id-legal-rag:latest

# With GPU
docker run --gpus all -p 7860:7860 id-legal-rag:latest
```

### Docker Compose

```bash
docker-compose up -d
```

Access:
- Gradio UI: http://localhost:7860
- FastAPI: http://localhost:8000

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster
- kubectl configured
- Persistent storage (for models)

### Deploy

```bash
# Create namespace
kubectl create namespace legal-rag

# Deploy
kubectl apply -f deploy/kubernetes/ -n legal-rag

# Check status
kubectl get pods -n legal-rag
```

### Scaling

```bash
# Scale workers
kubectl scale deployment legal-rag --replicas=3 -n legal-rag
```

## Production Configuration

### Environment Variables

```bash
# Required
LLM_PROVIDER=local
DATASET_NAME=azzindani/ID_Legal_EN

# Optional - Device Configuration
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
LLM_DEVICE=cuda

# Optional - Quantization
LLM_LOAD_IN_4BIT=true

# Optional - API Keys (if using API providers)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Resource Requirements

| Component | CPU | RAM | GPU VRAM |
|-----------|-----|-----|----------|
| Embeddings | 2 | 4GB | - |
| Reranker | 2 | 4GB | - |
| LLM (4-bit) | 4 | 8GB | 6GB |
| LLM (full) | 4 | 16GB | 16GB |

### Monitoring

- Health endpoint: `/health`
- Metrics: Integration with Prometheus
- Logs: JSON format to stdout

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
export BATCH_SIZE=8

# Enable quantization
export LLM_LOAD_IN_4BIT=true
```

### Slow Inference

```bash
# Use GPU
export LLM_DEVICE=cuda

# Or use API provider
export LLM_PROVIDER=openai
```

### Connection Issues

Check that all services are running:
```bash
docker-compose ps
kubectl get pods -n legal-rag
```
