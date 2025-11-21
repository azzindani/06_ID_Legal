# API Module

FastAPI REST API for the Indonesian Legal RAG System.

## Quick Start

```bash
# Run server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Or with auto-reload
uvicorn api.server:app --reload
```

## Endpoints

### Health
- `GET /api/v1/health` - System health check
- `GET /api/v1/ready` - Readiness check
- `GET /api/v1/live` - Liveness check

### Search
- `POST /api/v1/search` - Search documents

### Generate
- `POST /api/v1/generate` - Generate answer
- `POST /api/v1/generate/stream` - Streaming answer

### Sessions
- `POST /api/v1/sessions` - Create session
- `GET /api/v1/sessions` - List sessions
- `GET /api/v1/sessions/{id}` - Get session
- `GET /api/v1/sessions/{id}/history` - Get history
- `DELETE /api/v1/sessions/{id}` - Delete session
- `POST /api/v1/sessions/{id}/export` - Export session

## Usage Examples

### Generate Answer
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa sanksi pelanggaran UU Ketenagakerjaan?"}'
```

### With Session
```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{}'

# Generate with session
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa sanksinya?", "session_id": "YOUR_SESSION_ID"}'
```

### Search Documents
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "ketenagakerjaan", "max_results": 5}'
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

The API uses the global configuration from `config.py`.

Environment variables:
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
