# Indonesian Legal RAG System - Docker Image
# Multi-stage build with GPU support for production deployment
#
# Build: docker build -t legal-rag .
# Run:   docker-compose up
# Or:    docker run --gpus all -p 8000:8000 -p 7860:7860 legal-rag

# =============================================================================
# Base image with CUDA for GPU support
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# =============================================================================
# Dependencies stage
# =============================================================================
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# =============================================================================
# Production stage
# =============================================================================
FROM dependencies AS production

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/.data /app/exports /app/logs

# Environment variables
ENV PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    UI_HOST=0.0.0.0 \
    UI_PORT=7860

# Expose ports
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command - production launcher with health monitoring
CMD ["python", "launch_production.py"]
