"""
Multi-turn API Test Suite for Debugging Blocking Issues

This script tests both RAG search and conversational endpoints
with multiple sequential requests to identify blocking points.

Usage:
    1. Start the API server first
    2. Run this script: python tests/debug_api_blocking.py

File: tests/debug_api_blocking.py
"""

import requests
import time
import json
import sys
import os

# Configuration
API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/api/v1")
API_KEY = os.environ.get("LEGAL_API_KEY", "")
TIMEOUT = 120  # seconds

def log(msg, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def test_health():
    """Test API health endpoint"""
    log("Testing health endpoint...")
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if resp.status_code == 200:
            log("✅ Health check passed", "SUCCESS")
            return True
        else:
            log(f"❌ Health check failed: {resp.status_code}", "ERROR")
            return False
    except Exception as e:
        log(f"❌ Health check error: {e}", "ERROR")
        return False

def test_retrieve(query: str, request_num: int):
    """Test RAG retrieve endpoint (non-streaming)"""
    log(f"[Request {request_num}] Testing retrieve: '{query[:50]}...'")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    payload = {
        "query": query,
        "top_k": 3
    }
    
    start_time = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL}/rag/retrieve",
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            data = resp.json()
            sources_count = len(data.get("sources", []))
            log(f"[Request {request_num}] ✅ Retrieve completed in {elapsed:.2f}s, got {sources_count} sources", "SUCCESS")
            return True, elapsed
        else:
            log(f"[Request {request_num}] ❌ Retrieve failed: {resp.status_code} - {resp.text[:200]}", "ERROR")
            return False, elapsed
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        log(f"[Request {request_num}] ❌ Retrieve TIMEOUT after {elapsed:.2f}s", "ERROR")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        log(f"[Request {request_num}] ❌ Retrieve error after {elapsed:.2f}s: {e}", "ERROR")
        return False, elapsed

def test_chat_stream(query: str, request_num: int, session_id: str = "test-session"):
    """Test RAG chat streaming endpoint"""
    log(f"[Request {request_num}] Testing chat stream: '{query[:50]}...'")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    payload = {
        "query": query,
        "session_id": session_id,
        "thinking_level": "low",
        "top_k": 3,
        "stream": True
    }
    
    start_time = time.time()
    chunks_received = 0
    try:
        with requests.post(
            f"{API_BASE_URL}/rag/chat",
            headers=headers,
            json=payload,
            stream=True,
            timeout=TIMEOUT
        ) as resp:
            first_chunk_time = None
            
            if resp.status_code != 200:
                elapsed = time.time() - start_time
                log(f"[Request {request_num}] ❌ Chat failed: {resp.status_code}", "ERROR")
                return False, elapsed
            
            for line in resp.iter_lines():
                if line:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        log(f"[Request {request_num}] 📡 First chunk at {first_chunk_time:.2f}s")
                    
                    chunks_received += 1
                    content = line.decode('utf-8')
                    
                    # Check for done event
                    if content.startswith('data: '):
                        try:
                            event = json.loads(content[6:])
                            if event.get('type') == 'done':
                                break
                        except json.JSONDecodeError:
                            pass
        
        elapsed = time.time() - start_time
        log(f"[Request {request_num}] ✅ Chat completed in {elapsed:.2f}s, {chunks_received} chunks", "SUCCESS")
        return True, elapsed
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        log(f"[Request {request_num}] ❌ Chat TIMEOUT after {elapsed:.2f}s (got {chunks_received} chunks)", "ERROR")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        log(f"[Request {request_num}] ❌ Chat error after {elapsed:.2f}s: {e}", "ERROR")
        return False, elapsed

def run_multi_turn_test():
    """Run multi-turn test combining retrieve and chat"""
    log("=" * 60)
    log("MULTI-TURN API BLOCKING TEST")
    log("=" * 60)
    
    # Test queries
    queries = [
        "Apa sanksi pelanggaran UU Ketenagakerjaan?",
        "Bagaimana prosedur PHK menurut hukum?",
        "Apa hak pekerja yang terkena PHK?",
    ]
    
    results = {
        "health": False,
        "retrieve": [],
        "chat": []
    }
    
    # Step 1: Health check
    log("\n--- STEP 1: Health Check ---")
    results["health"] = test_health()
    if not results["health"]:
        log("❌ Cannot proceed - API not healthy", "ERROR")
        return results
    
    # Step 2: Test retrieve endpoint (multiple times)
    log("\n--- STEP 2: Retrieve Tests (3 sequential) ---")
    for i, query in enumerate(queries, 1):
        success, elapsed = test_retrieve(query, i)
        results["retrieve"].append({
            "request_num": i,
            "success": success,
            "elapsed": elapsed,
            "query": query[:50]
        })
        time.sleep(0.5)  # Small delay between requests
    
    # Step 3: Test chat endpoint (multiple times) 
    log("\n--- STEP 3: Chat Stream Tests (3 sequential) ---")
    for i, query in enumerate(queries, 1):
        success, elapsed = test_chat_stream(query, i)
        results["chat"].append({
            "request_num": i,
            "success": success,
            "elapsed": elapsed,
            "query": query[:50]
        })
        time.sleep(0.5)  # Small delay between requests
    
    # Step 4: Mixed test (retrieve then chat then retrieve)
    log("\n--- STEP 4: Mixed Tests (retrieve -> chat -> retrieve) ---")
    log("Testing retrieve...")
    test_retrieve("Apa itu kontrak kerja waktu tertentu?", 1)
    log("Testing chat...")
    test_chat_stream("Jelaskan tentang PKWT", 2)
    log("Testing retrieve again...")
    test_retrieve("Apa perbedaan PKWT dan PKWTT?", 3)
    
    # Summary
    log("\n" + "=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)
    
    retrieve_success = sum(1 for r in results["retrieve"] if r["success"])
    chat_success = sum(1 for r in results["chat"] if r["success"])
    
    log(f"Health: {'✅' if results['health'] else '❌'}")
    log(f"Retrieve: {retrieve_success}/{len(results['retrieve'])} passed")
    log(f"Chat: {chat_success}/{len(results['chat'])} passed")
    
    if retrieve_success < len(results["retrieve"]) or chat_success < len(results["chat"]):
        log("\n❌ BLOCKING ISSUE DETECTED", "ERROR")
        log("Check which request number failed to identify the pattern", "ERROR")
    else:
        log("\n✅ ALL TESTS PASSED", "SUCCESS")
    
    return results

if __name__ == "__main__":
    # Allow custom API URL from command line
    if len(sys.argv) > 1:
        API_BASE_URL = sys.argv[1]
    
    log(f"API URL: {API_BASE_URL}")
    log(f"Timeout: {TIMEOUT}s")
    
    try:
        results = run_multi_turn_test()
    except KeyboardInterrupt:
        log("\nTest interrupted by user", "WARN")
    except Exception as e:
        log(f"Test failed with error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
