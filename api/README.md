# Indonesian Legal RAG API

A high-performance, secure REST API for the Indonesian Legal RAG system. This API exposes advanced retrieval, deep research, and conversational capabilities over legal documents.

## 🚀 Quick Start

```bash
# Start the server
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔐 Authentication

All API endpoints require authentication via an API Key.
You must include the `X-API-Key` header in every request.

```http
X-API-Key: your_secure_api_key_here
```

> **Note**: Configure your keys in the `LEGAL_API_KEYS_ADDITIONAL` environment variable or `config.py`.

---

## 📚 Endpoints

### 1. Retrieve Documents (`/api/v1/rag/retrieve`)
Fast, pure document retrieval without LLM generation. Ideal for search bars or getting raw context.

**Request:**
```json
POST /api/v1/rag/retrieve
{
  "query": "Apa syarat pendirian PT?",
  "top_k": 5,              // Optional (default: 5)
  "min_score": 0.3         // Optional (default: 0.0)
}
```

**Response:**
```json
{
  "query": "Apa syarat pendirian PT?",
  "total_retrieved": 5,
  "search_time": 0.12,
  "documents": [
    {
      "content": "...",
      "score": 0.85,
      "metadata": {
        "source": "UU No. 40 Tahun 2007",
        "category": "Undang-Undang"
      }
    }
  ]
}
```

### 2. Deep Research (`/api/v1/rag/research`)
Full RAG pipeline with multi-step reasoning, team consensus, and deep analysis.

**Request:**
```json
POST /api/v1/rag/research
{
  "query": "Jelaskan prosedur likuidasi perseroan terbatas menurut UU terbaru",
  "thinking_level": "medium",  // "low" (fast), "medium" (balanced), "high" (deep)
  "team_size": 3               // Number of AI researchers (default: 3)
}
```

**Response:**
```json
{
  "answer": "Berdasarkan UU No. 40 Tahun 2007...",
  "research_time": 4.5,
  "legal_references": [
    "Pasal 142 UU No. 40 Tahun 2007",
    "Pasal 143 UU No. 40 Tahun 2007"
  ],
  "confidence_score": 0.92,
  "thinking_process": "..."   // Detailed reasoning steps
}
```

### 3. Chat (`/api/v1/rag/chat`)
Conversational endpoint with session management and context retention.

**Request:**
```json
POST /api/v1/rag/chat
{
  "query": "Apa sanksinya?",
  "session_id": "user_123_session",  // Optional: creates new if omitted
  "thinking_level": "low",
  "stream": false
}
```

**Response:**
```json
{
  "answer": "Sanksi pelanggaran tersebut meliputi...",
  "session_id": "user_123_session",
  "history_length": 2,
  "references": [...]
}
```

---

## ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thinking_level` | string | "medium" | Controls analysis depth. Options: `low`, `medium`, `high`. |
| `team_size` | int | 3 | Number of AI agents reaching consensus (1-5). |
| `top_k` | int | 5 | Number of documents to retrieve initially. |

---

## 🛡️ Security Features

This API includes a comprehensive modular security system:

1.  **Rate Limiting**: Token-bucket algorithm prevents abuse (default: 60 req/min).
2.  **Input Sanitization**: Automatically blocks XSS, SQL Injection, and Prompt Injection attacks.
3.  **Secure Headers**: Responses include HSTS, X-Content-Type-Options, etc.
4.  **File Protection**: Validates magic bytes and MIME types for uploads (if enabled).

### Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| `401` | Unauthorized | Check your `X-API-Key` header. |
| `422` | Validation Error | Check JSON body format or invalid parameters. |
| `429` | Too Many Requests | Slow down request rate. |
| `400` | Security Block | Input detected as malicious (XSS/Injection). |
| `500` | Server Error | Check server logs for details. |

---

## 🧪 Testing

Run the integration tests to verify API functionality:

```bash
# Run HTTP-level tests
python tests/integration/test_api_http.py --verbose

# Run concurrent load tests
python tests/integration/test_concurrent_users.py --users 10
```

---

## 📓 Jupyter Notebook Usage

You can test the API directly within a Jupyter Notebook (e.g., on Kaggle) by running the server in a background subprocess.

Copy and paste the following cell into your notebook:


# ============================================
# 📓 API SERVER + INTERACTIVE TESTS
# ============================================
# This cell allows you to run the FastAPI server in the background
# and test it immediately with curl commands.
# ============================================

import subprocess
import time
import os
import sys

# --------------------------------------------
# 1. AUTHENTICATION SETUP
# --------------------------------------------
# We verify the API Key mechanism works
os.environ['LEGAL_API_KEY'] = "test_integration_key_12345"
print(f"✅ API Key configured: {os.environ['LEGAL_API_KEY']}")

# --------------------------------------------
# 2. START SERVER (BACKGROUND)
# --------------------------------------------
print("\n🚀 Starting FastAPI Server in background...")
# We use sys.executable to ensure we use the same Python environment
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Give it time to initialize the RAG pipeline
print("⏳ Waiting 15 seconds for server initialization...")
time.sleep(15)

try:
    # --------------------------------------------
    # 3. HEALTH CHECK
    # --------------------------------------------
    print("\n[TEST 1] Checking System Health...")
    !curl -s http://127.0.0.1:8000/api/v1/health | python -m json.tool

    # --------------------------------------------
    # 4. TEST: RETRIEVAL ENDPOINT
    # --------------------------------------------
    print("\n[TEST 2] Testing Retrieval (No LLM)...")
    print("Query: 'Apa syarat pendirian PT?'")
    !curl -s -X POST http://127.0.0.1:8000/api/v1/rag/retrieve \
      -H "Content-Type: application/json" \
      -H "X-API-Key: test_integration_key_12345" \
      -d '{"query": "Apa syarat pendirian PT?", "top_k": 2}' | python -m json.tool

    # --------------------------------------------
    # 5. TEST: RESEARCH ENDPOINT (DEEP REASONING)
    # --------------------------------------------
    print("\n[TEST 3] Testing Deep Research (With LLM)...")
    print("Query: 'Jelaskan prosedur likuidasi PT'")
    print("Note: This performs multi-step reasoning, please wait...")
    !curl -s -X POST http://127.0.0.1:8000/api/v1/rag/research \
      -H "Content-Type: application/json" \
      -H "X-API-Key: test_integration_key_12345" \
      -d '{"query": "Jelaskan prosedur likuidasi PT", "thinking_level": "low", "team_size": 2}' | python -m json.tool

    # --------------------------------------------
    # 6. TEST: CHAT ENDPOINT (CONTEXTUAL)
    # --------------------------------------------
    print("\n[TEST 4] Testing Conversational Chat...")
    print("Query: 'Apa itu UU Ketenagakerjaan?'")
    !curl -s -X POST http://127.0.0.1:8000/api/v1/rag/chat \
      -H "Content-Type: application/json" \
      -H "X-API-Key: test_integration_key_12345" \
      -d '{"query": "Apa itu UU Ketenagakerjaan?", "session_id": "nb_session_1", "stream": false}' | python -m json.tool

finally:
    # --------------------------------------------
    # 7. CLEANUP
    # --------------------------------------------
    print("\n🛑 Stopping background server...")
    server.terminate()
    try:
        server.wait(timeout=5)
        print("✅ Server stopped successfully")
    except subprocess.TimeoutExpired:
        server.kill()
        print("⚠️ Server killed forcefully")

print("\n🎉 All interactive tests completed!")
