"""
API Blocking Test - HTTP/API Layer Testing with Full Cache Management

Tests the API endpoints via HTTP (same as Gradio UI does) to identify blocking issues
in the network/API layer. Includes aggressive RAM/VRAM clearing and diagnostic output.

Prerequisites:
    1. API server must be running: python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
    2. Then run this script

Usage:
    python tests/debug_api_blocking.py [API_URL]
    
    Default API_URL: http://127.0.0.1:8000/api/v1

File: tests/debug_api_blocking.py
"""

import requests
import time
import json
import sys
import os
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration
API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/api/v1")
API_KEY = os.environ.get("LEGAL_API_KEY", "")
TIMEOUT = 180  # seconds


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    stats = {
        'ram_used_mb': 0,
        'vram_used_mb': 0,
        'vram_total_mb': 0,
        'vram_percent': 0
    }
    
    try:
        import psutil
        process = psutil.Process()
        stats['ram_used_mb'] = process.memory_info().rss / 1024 / 1024
    except:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            stats['vram_used_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['vram_total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            stats['vram_percent'] = (stats['vram_used_mb'] / stats['vram_total_mb']) * 100 if stats['vram_total_mb'] > 0 else 0
    except:
        pass
    
    return stats


def clear_all_cache(context: str = "") -> Dict[str, Any]:
    """Aggressively clear both RAM and VRAM caches"""
    before = get_memory_stats()
    
    gc.collect()
    gc.collect()
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    
    gc.collect()
    
    after = get_memory_stats()
    
    return {
        'context': context,
        'before': before,
        'after': after,
        'freed_ram_mb': before['ram_used_mb'] - after['ram_used_mb'],
        'freed_vram_mb': before['vram_used_mb'] - after['vram_used_mb']
    }


# =============================================================================
# LOGGING / DIAGNOSTICS
# =============================================================================

class DiagnosticLogger:
    """Collects diagnostic information for analysis"""
    
    def __init__(self):
        self.start_time = time.time()
        self.events: List[Dict[str, Any]] = []
        self.test_results: List[Dict[str, Any]] = []
        self.http_client = requests.Session()  # Use session for connection reuse
        
    def log(self, msg: str, level: str = "INFO", **extra):
        timestamp = time.strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time
        
        event = {
            'timestamp': timestamp,
            'elapsed_s': round(elapsed, 2),
            'level': level,
            'message': msg,
            **extra
        }
        self.events.append(event)
        
        prefix = {"SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(level, "")
        print(f"[{timestamp}] [{level}] {prefix} {msg}", flush=True)
        if extra:
            for k, v in extra.items():
                print(f"    {k}: {v}", flush=True)
    
    def record_test(self, test_name: str, success: bool, elapsed_s: float, 
                    error: str = None, memory: Dict = None, **extra):
        result = {
            'test_name': test_name,
            'success': success,
            'elapsed_s': round(elapsed_s, 2),
            'error': error,
            'memory': memory,
            'timestamp': datetime.now().isoformat(),
            **extra
        }
        self.test_results.append(result)
        return result
    
    def generate_summary(self) -> str:
        """Generate diagnostic summary for AI analysis"""
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['success'])
        failed = total_tests - passed
        
        summary_lines = [
            "\n" + "=" * 70,
            "API BLOCKING DIAGNOSTIC SUMMARY",
            "=" * 70,
            f"API URL: {API_BASE_URL}",
            f"Total Tests: {total_tests}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Total Time: {time.time() - self.start_time:.1f}s",
            "",
            "--- TEST RESULTS ---"
        ]
        
        for i, r in enumerate(self.test_results, 1):
            status = "✅ PASS" if r['success'] else "❌ FAIL"
            test_type = "RETRIEVE" if "retrieve" in r['test_name'].lower() else "CHAT"
            summary_lines.append(f"[{i}] [{test_type}] {r['test_name']}: {status} ({r['elapsed_s']:.2f}s)")
            if r.get('chunks_received'):
                summary_lines.append(f"    Chunks: {r['chunks_received']}")
            if r.get('first_chunk_s'):
                summary_lines.append(f"    First chunk: {r['first_chunk_s']:.2f}s")
            if r.get('http_status'):
                summary_lines.append(f"    HTTP Status: {r['http_status']}")
            if r.get('error'):
                summary_lines.append(f"    Error: {r['error'][:200]}")
        
        # Failure analysis
        if failed > 0:
            summary_lines.append("")
            summary_lines.append("--- FAILURE ANALYSIS ---")
            
            fail_indices = [i+1 for i, r in enumerate(self.test_results) if not r['success']]
            summary_lines.append(f"Failed at test(s): {fail_indices}")
            
            # Analyze patterns
            retrieve_fails = sum(1 for r in self.test_results if not r['success'] and 'retrieve' in r['test_name'].lower())
            chat_fails = sum(1 for r in self.test_results if not r['success'] and 'chat' in r['test_name'].lower())
            
            if chat_fails > 0 and retrieve_fails == 0:
                summary_lines.append("⚠️ PATTERN: Only CHAT (streaming) requests fail")
                summary_lines.append("   LIKELY CAUSE: Streaming response handling / connection not closed")
            elif retrieve_fails > 0 and chat_fails == 0:
                summary_lines.append("⚠️ PATTERN: Only RETRIEVE requests fail")
                summary_lines.append("   LIKELY CAUSE: Server-side processing issue")
            elif len(fail_indices) > 0 and fail_indices[0] > 1:
                summary_lines.append(f"⚠️ PATTERN: First {fail_indices[0]-1} request(s) pass, then fails")
                summary_lines.append("   LIKELY CAUSE: Resource exhaustion or lock deadlock")
            
            # Check for timeouts
            timeouts = [r for r in self.test_results if r.get('error') and 'timeout' in str(r['error']).lower()]
            if timeouts:
                summary_lines.append(f"⚠️ {len(timeouts)} request(s) timed out")
                summary_lines.append("   LIKELY CAUSE: Server blocked or GPU busy")
        else:
            summary_lines.append("")
            summary_lines.append("--- ALL TESTS PASSED ---")
            summary_lines.append("API layer works correctly for multiple sequential requests.")
            summary_lines.append("If blocking occurs in Gradio UI, the issue is in Gradio/UI code.")
        
        summary_lines.append("=" * 70)
        
        return "\n".join(summary_lines)


# =============================================================================
# API TESTS
# =============================================================================

# Realistic UI query sequence
UI_TEST_QUERIES = [
    {
        'query': "Apa sanksi pelanggaran UU Ketenagakerjaan?",
        'description': "Initial query about labor law",
    },
    {
        'query': "Bagaimana prosedur PHK menurut hukum?",
        'description': "Follow-up about termination",
    },
    {
        'query': "Apa itu kontrak kerja waktu tertentu?",
        'description': "Topic switch to contracts",
    },
    {
        'query': "Jelaskan perbedaan PKWT dan PKWTT",
        'description': "Quick follow-up",
    },
]


def test_health(logger: DiagnosticLogger) -> bool:
    """Test API health endpoint"""
    logger.log("Testing health endpoint...")
    start = time.time()
    
    try:
        resp = logger.http_client.get(f"{API_BASE_URL}/health", timeout=10)
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            logger.log(f"Health check passed in {elapsed:.2f}s", "SUCCESS")
            logger.record_test("Health", True, elapsed, http_status=200)
            return True
        else:
            logger.log(f"Health check failed: {resp.status_code}", "ERROR")
            logger.record_test("Health", False, elapsed, error=f"Status {resp.status_code}", http_status=resp.status_code)
            return False
    except Exception as e:
        elapsed = time.time() - start
        logger.log(f"Health check error: {e}", "ERROR")
        logger.record_test("Health", False, elapsed, error=str(e))
        return False


def test_retrieve(logger: DiagnosticLogger, query: str, test_num: int) -> bool:
    """Test RAG retrieve endpoint (non-streaming)"""
    logger.log(f"[RETRIEVE {test_num}] Query: \"{query[:40]}...\"")
    
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    payload = {"query": query, "top_k": 3}
    
    pre_cache = clear_all_cache(f"before_retrieve_{test_num}")
    start = time.time()
    
    try:
        resp = logger.http_client.post(
            f"{API_BASE_URL}/rag/retrieve",
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        post_cache = clear_all_cache(f"after_retrieve_{test_num}")
        
        if resp.status_code == 200:
            data = resp.json()
            sources_count = len(data.get("sources", []))
            logger.log(f"PASSED: {sources_count} sources in {elapsed:.2f}s", "SUCCESS")
            logger.record_test(
                f"Retrieve_{test_num}",
                True, elapsed,
                memory=post_cache,
                http_status=200,
                sources_count=sources_count
            )
            return True
        else:
            logger.log(f"FAILED: HTTP {resp.status_code}", "ERROR")
            logger.record_test(
                f"Retrieve_{test_num}",
                False, elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                memory=post_cache,
                http_status=resp.status_code
            )
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        logger.log(f"TIMEOUT after {elapsed:.2f}s", "ERROR")
        logger.record_test(f"Retrieve_{test_num}", False, elapsed, error="Request timeout")
        return False
    except Exception as e:
        elapsed = time.time() - start
        logger.log(f"ERROR: {e}", "ERROR")
        logger.record_test(f"Retrieve_{test_num}", False, elapsed, error=str(e))
        return False


def test_chat_stream(logger: DiagnosticLogger, query: str, test_num: int, session_id: str = "test-session") -> bool:
    """Test RAG chat streaming endpoint (as UI does)"""
    logger.log(f"[CHAT {test_num}] Query: \"{query[:40]}...\"")
    
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    payload = {
        "query": query,
        "session_id": session_id,
        "thinking_level": "low",
        "top_k": 3,
        "stream": True
    }
    
    pre_cache = clear_all_cache(f"before_chat_{test_num}")
    start = time.time()
    chunks_received = 0
    first_chunk_time = None
    error_msg = None
    success = False
    
    try:
        # Use context manager to ensure connection is closed
        with logger.http_client.post(
            f"{API_BASE_URL}/rag/chat",
            headers=headers,
            json=payload,
            stream=True,
            timeout=TIMEOUT
        ) as resp:
            
            if resp.status_code != 200:
                elapsed = time.time() - start
                error_msg = f"HTTP {resp.status_code}"
                logger.log(f"FAILED: {error_msg}", "ERROR")
                logger.record_test(
                    f"Chat_{test_num}",
                    False, elapsed,
                    error=error_msg,
                    http_status=resp.status_code
                )
                return False
            
            # Read all chunks
            for line in resp.iter_lines():
                if line:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start
                        logger.log(f"First chunk at {first_chunk_time:.2f}s")
                    
                    chunks_received += 1
                    content = line.decode('utf-8')
                    
                    if content.startswith('data: '):
                        try:
                            event = json.loads(content[6:])
                            if event.get('type') == 'done':
                                success = True
                                break
                            elif event.get('type') == 'error':
                                error_msg = event.get('message', 'Unknown error')
                                break
                        except json.JSONDecodeError:
                            pass
        
        elapsed = time.time() - start
        post_cache = clear_all_cache(f"after_chat_{test_num}")
        
        if success:
            logger.log(f"PASSED: {chunks_received} chunks in {elapsed:.2f}s", "SUCCESS")
        else:
            if not error_msg:
                error_msg = "Stream ended without 'done' event"
            logger.log(f"FAILED: {error_msg}", "ERROR")
        
        logger.record_test(
            f"Chat_{test_num}",
            success, elapsed,
            error=error_msg,
            memory=post_cache,
            http_status=200,
            chunks_received=chunks_received,
            first_chunk_s=first_chunk_time
        )
        return success
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        logger.log(f"TIMEOUT after {elapsed:.2f}s (got {chunks_received} chunks)", "ERROR")
        logger.record_test(
            f"Chat_{test_num}",
            False, elapsed,
            error="Request timeout",
            chunks_received=chunks_received,
            first_chunk_s=first_chunk_time
        )
        return False
    except Exception as e:
        elapsed = time.time() - start
        logger.log(f"ERROR: {e}", "ERROR")
        logger.record_test(f"Chat_{test_num}", False, elapsed, error=str(e))
        return False


def run_full_test_suite(logger: DiagnosticLogger):
    """Run complete test suite simulating UI usage"""
    
    logger.log("=" * 60)
    logger.log("API BLOCKING TEST (Simulating UI via HTTP)")
    logger.log(f"API URL: {API_BASE_URL}")
    logger.log("=" * 60)
    
    # Step 1: Health check
    logger.log("")
    logger.log("--- STEP 1: Health Check ---")
    if not test_health(logger):
        logger.log("API not healthy, cannot continue", "ERROR")
        return
    
    # Step 2: Retrieve tests (no LLM generation)
    logger.log("")
    logger.log("--- STEP 2: Retrieve Tests (Non-streaming) ---")
    for i, test_case in enumerate(UI_TEST_QUERIES[:3], 1):  # First 3 queries
        test_retrieve(logger, test_case['query'], i)
        time.sleep(0.5)
    
    # Step 3: Chat tests (streaming with LLM)
    logger.log("")
    logger.log("--- STEP 3: Chat Stream Tests (Streaming) ---")
    for i, test_case in enumerate(UI_TEST_QUERIES, 1):  # All 4 queries
        test_chat_stream(logger, test_case['query'], i)
        time.sleep(0.5)
    
    # Step 4: Mixed test (realistic usage pattern)
    logger.log("")
    logger.log("--- STEP 4: Mixed Tests (Retrieve -> Chat -> Retrieve -> Chat) ---")
    test_retrieve(logger, "Cari info kontrak kerja", 10)
    time.sleep(0.5)
    test_chat_stream(logger, "Jelaskan tentang PKWT", 10)
    time.sleep(0.5)
    test_retrieve(logger, "Cari tentang PHK", 11)
    time.sleep(0.5)
    test_chat_stream(logger, "Prosedur PHK yang benar", 11)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global API_BASE_URL
    
    # Allow custom API URL from command line
    if len(sys.argv) > 1:
        API_BASE_URL = sys.argv[1]
    
    logger = DiagnosticLogger()
    
    logger.log("=" * 70)
    logger.log("API BLOCKING DIAGNOSTIC TEST")
    logger.log(f"Started: {datetime.now().isoformat()}")
    logger.log(f"API URL: {API_BASE_URL}")
    logger.log(f"Timeout: {TIMEOUT}s")
    logger.log("=" * 70)
    
    try:
        run_full_test_suite(logger)
    except KeyboardInterrupt:
        logger.log("Tests interrupted by user", "WARN")
    except Exception as e:
        logger.log(f"Test suite crashed: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc()[:1000], "ERROR")
    
    # Print diagnostic summary
    summary = logger.generate_summary()
    print(summary)
    
    # Save results to file
    try:
        output_file = os.path.join(os.path.dirname(__file__), "api_test_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'api_url': API_BASE_URL,
                'events': logger.events,
                'test_results': logger.test_results,
                'summary': summary
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    except:
        pass


if __name__ == "__main__":
    main()
