# API Reference

## REST API Endpoints

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Search

```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "Apa sanksi UU Ketenagakerjaan?",
  "top_k": 10,
  "include_metadata": true
}
```

Response:
```json
{
  "results": [
    {
      "id": "doc-123",
      "content": "...",
      "score": 0.95,
      "metadata": {...}
    }
  ],
  "total": 10,
  "query_type": "sanctions"
}
```

### Generate

```http
POST /api/v1/generate
Content-Type: application/json

{
  "query": "Apa sanksi UU Ketenagakerjaan?",
  "session_id": "session-abc",
  "stream": false
}
```

Response:
```json
{
  "answer": "...",
  "sources": [...],
  "metadata": {
    "query_type": "sanctions",
    "processing_time": 2.5,
    "total_results": 10
  }
}
```

### Session Management

```http
POST /api/v1/session
```

Response:
```json
{
  "session_id": "session-abc",
  "created_at": "2024-01-01T00:00:00Z"
}
```

```http
GET /api/v1/session/{session_id}
```

```http
DELETE /api/v1/session/{session_id}
```

## Python SDK

### Pipeline

```python
from pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline({'llm_provider': 'local'})
pipeline.initialize()

# Query
result = pipeline.query(
    "Apa sanksi pelanggaran UU?",
    stream=False
)

print(result['answer'])
print(result['sources'])

# Cleanup
pipeline.shutdown()
```

### Providers

```python
from providers import create_provider, list_providers

# List available providers
providers = list_providers()
print(providers)  # ['local', 'openai', 'anthropic', ...]

# Create provider
provider = create_provider('openai', {
    'model': 'gpt-4',
    'api_key': '...'
})

# Generate
response = provider.generate(
    prompt="Your prompt",
    max_tokens=1000
)
```

### Conversation Management

```python
from conversation import ConversationManager

manager = ConversationManager()

# Start session
session_id = manager.start_session()

# Add turn
manager.add_turn(
    session_id=session_id,
    query="User question",
    answer="System response"
)

# Get context
context = manager.get_context_for_query(session_id)

# Export
from conversation import MarkdownExporter
exporter = MarkdownExporter()
content = exporter.export(manager.get_session(session_id))
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource not found |
| 500 | Internal Error - Server error |
| 503 | Service Unavailable - Not initialized |
