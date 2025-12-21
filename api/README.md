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
# 📓 API SERVER + INTERACTIVE TESTS (PURE PYTHON)
# ============================================
# This cell runs the server and tests it using Python's requests library.
# It handles server startup, waiting, testing, and cleanup automatically.
# ============================================

import subprocess
import time
import os
import sys
import requests
import json
import signal

# --------------------------------------------
# 1. SETUP
# --------------------------------------------
API_KEY = "test_integration_key_12345"
os.environ['LEGAL_API_KEY'] = API_KEY
BASE_URL = "http://127.0.0.1:8000/api/v1"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def check_server_alive():
    try:
        requests.get(f"{BASE_URL}/health", timeout=1)
        return True
    except:
        return False

# --------------------------------------------
# 2. START SERVER
# --------------------------------------------
print(f"🚀 Starting API Server on Port 8000...")
# Use sys.executable to ensure we use the same Python environment
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

# --------------------------------------------
# 3. WAIT FOR HEALTHY (MAX 90s)
# --------------------------------------------
print("⏳ Waiting for server to be ready (may take ~300s for models)...", end="", flush=True)
server_ready = False
# Try polling explicitly
for _ in range(300):
    try:
        # We use a short timeout so we don't hang if server is dead
        r = requests.get(f"{BASE_URL}/health", timeout=1)
        if r.status_code == 200:
            print("\n✅ Server is UP and READY!")
            server_ready = True
            break
    except Exception:
        time.sleep(1)
        print(".", end="", flush=True)

if not server_ready:
    print("\n❌ Server failed to start.")
    server.terminate()
    print("\n❌ Server failed to start.")
    server.terminate()
    # Output already printed to stdout/stderr
else:
    try:
        # TEST A: RETRIEVAL
        # ------------------
        print("\n[TEST A] Retrieval (curl)...")
        cmd_a = [
            "curl", "-X", "POST", f"{BASE_URL}/rag/retrieve",
            "-H", f"X-API-Key: {API_KEY}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"query": "Apa syarat pendirian PT?", "top_k": 2}),
            "--max-time", "300" 
        ]
        subprocess.run(cmd_a)

        # CHECK SERVER
        if not check_server_alive():
            print("\n\n❌ Server crashed after Test A (Likely OOM - Try reducing top_k)", flush=True)
        else:
            # TEST B: DEEP RESEARCH
            # ------------------
            print("\n\n[TEST B] Deep Research (curl) - This may take time...", flush=True)
            cmd_b = [
                "curl", "-X", "POST", f"{BASE_URL}/rag/research",
                "-H", f"X-API-Key: {API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"query": "Apa itu PT?", "thinking_level": "low", "team_size": 1}),
                "--max-time", "600"
            ]
            subprocess.run(cmd_b)

        # CHECK SERVER
        if not check_server_alive():
             print("\n\n❌ Server crashed after Test B", flush=True)
        else:
            # TEST C: CHAT
            # ------------------
            print("\n\n[TEST C] Chat (curl)...", flush=True)
            session_id = f"demo_{int(time.time())}"
            
            # 1. CREATE SESSION
            print(f"Creating session: {session_id}...")
            cmd_c_init = [
                "curl", "-X", "POST", f"{BASE_URL}/sessions",
                "-H", f"X-API-Key: {API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"session_id": session_id}),
                "--max-time", "30"
            ]
            subprocess.run(cmd_c_init)

            # 2. TURN 1
            print("\n\nTurn 1...", flush=True)
            cmd_c1 = [
                "curl", "-X", "POST", f"{BASE_URL}/rag/chat",
                "-H", f"X-API-Key: {API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"query": "Apa itu UU Ketenagakerjaan?", "session_id": session_id, "stream": False}),
                "--max-time", "300"
            ]
            subprocess.run(cmd_c1)

            # 3. TURN 2 (Follow-up)
            print("\n\nTurn 2 (Follow-up)...", flush=True)
            cmd_c2 = [
                "curl", "-X", "POST", f"{BASE_URL}/rag/chat",
                "-H", f"X-API-Key: {API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"query": "Apa sanksi pidananya?", "session_id": session_id, "stream": False}),
                "--max-time", "300"
            ]
            subprocess.run(cmd_c2)

            # CHECK SERVER
            if not check_server_alive():
                print("\n\n❌ Server crashed after Test C", flush=True)
            else:
                # TEST D: STREAMING CHAT (Real-time Thinking)
                # ------------------
                print("\n\n[TEST D] Streaming Chat (Real-time Thinking)...", flush=True)
                print("Testing live thinking process and answer streaming...")
                
                try:
                    import requests
                    payload = {
                        "query": "Jelaskan perbedaan PT dan CV secara mendalam.", 
                        "session_id": session_id, 
                        "thinking_level": "medium", # Use medium for faster results
                        "stream": True
                    }
                    
                    response = requests.post(
                        f"{BASE_URL}/rag/chat", 
                        headers=HEADERS, 
                        json=payload, 
                        stream=True,
                        timeout=600
                    )
                    
                    print(f"HTTP Status: {response.status_code}")
                    if response.status_code == 200:
                        print("--- STREAM START ---")
                        for line in response.iter_lines():
                            if line:
                                content = line.decode('utf-8')
                                if content.startswith('data: '):
                                    data = json.loads(content[6:])
                                    ev_type = data.get('type')
                                    
                                    if ev_type == 'progress':
                                        print(f"[PROGRESS] {data.get('message')}")
                                    elif ev_type == 'thinking':
                                        print(data.get('content'), end="", flush=True) # Stream thinking
                                    elif ev_type == 'chunk':
                                        print(data.get('content'), end="", flush=True) # Stream answer
                                    elif ev_type == 'done':
                                        print("\n--- STREAM END ---")
                        print("✅ Streaming test finished successfully")
                    else:
                        print(f"❌ Streaming failed: {response.text}")
                except Exception as e:
                    print(f"❌ Error during streaming test: {e}")

    finally:
        # --------------------------------------------
        # 5. CLEANUP
        # --------------------------------------------
        print("\n🛑 Shutting down server...")
        server.terminate()
        try:
            server.wait(timeout=5)
            print("✅ Server stopped successfully")
        except:
            server.kill()
            print("⚠️ Server killed forcefully")

print("\n🎉 All interactive tests completed!")
