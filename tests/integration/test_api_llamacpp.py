"""
LlamaCpp API Integration Test

Tests the LlamaCpp provider through the API endpoints.
Requires API server to be running.

Usage:
    # Start server first
    python -m api.server --llm-provider llamacpp
    
    # Then run tests
    python tests/integration/test_api_llamacpp.py

File: tests/integration/test_api_llamacpp.py
"""

import sys
import os
import time
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import requests
except ImportError:
    print("❌ requests not installed. Run: pip install requests")
    sys.exit(1)


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, success: bool, message: str = ""):
    """Print a test result"""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {name}: {message}")


BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/api/v1"


def test_api_health():
    """Test API is running"""
    print_header("TEST 1: API Health Check")
    
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        success = r.status_code == 200
        print(f"Status: {r.status_code}")
        if success:
            print(f"Response: {r.json()}")
        print_result("API Health", success, "Server is running")
        return success
    except requests.exceptions.ConnectionError:
        print_result("API Health", False, "Server not running. Start with: python -m api.server --llm-provider llamacpp")
        return False


def test_api_llm_providers():
    """Test /llm/providers endpoint includes llamacpp"""
    print_header("TEST 2: LLM Providers List")
    
    try:
        r = requests.get(f"{API_URL}/llm/providers", timeout=10)
        if r.status_code == 200:
            providers = r.json()
            print(f"Providers: {json.dumps(providers, indent=2)}")
            
            # Check for llamacpp
            has_llamacpp = any(p.get('id') == 'llamacpp' for p in providers)
            print_result("Providers", has_llamacpp, "llamacpp in provider list" if has_llamacpp else "llamacpp NOT found")
            return has_llamacpp
        else:
            print_result("Providers", False, f"Status {r.status_code}")
            return False
    except Exception as e:
        print_result("Providers", False, str(e))
        return False


def test_api_llm_status():
    """Test /llm/status endpoint"""
    print_header("TEST 3: LLM Status")
    
    try:
        r = requests.get(f"{API_URL}/llm/status", timeout=10)
        if r.status_code == 200:
            status = r.json()
            print(f"Provider: {status.get('provider')}")
            print(f"Model: {status.get('model')}")
            print(f"Available: {status.get('available')}")
            
            is_llamacpp = status.get('provider') == 'llamacpp'
            print_result("Status", True, f"Current provider: {status.get('provider')}")
            return True
        else:
            print_result("Status", False, f"Status {r.status_code}")
            return False
    except Exception as e:
        print_result("Status", False, str(e))
        return False


def test_api_switch_to_llamacpp():
    """Test switching to llamacpp provider via API"""
    print_header("TEST 4: Switch to LlamaCpp")
    
    try:
        r = requests.post(
            f"{API_URL}/llm/config",
            json={"provider": "llamacpp"},
            timeout=120  # Model loading may take time
        )
        
        if r.status_code == 200:
            result = r.json()
            print(f"Success: {result.get('success')}")
            print(f"Provider: {result.get('provider')}")
            print(f"Available: {result.get('available')}")
            
            success = result.get('provider') == 'llamacpp'
            print_result("Switch", success, "Switched to llamacpp" if success else "Switch failed")
            return success
        else:
            print_result("Switch", False, f"Status {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print_result("Switch", False, str(e))
        return False


def test_api_chat_llamacpp():
    """Test RAG chat with llamacpp provider"""
    print_header("TEST 5: RAG Chat with LlamaCpp")
    
    try:
        r = requests.post(
            f"{API_URL}/rag/chat",
            json={
                "query": "Apa itu hukum pidana?",
                "session_id": "test-llamacpp-session",
                "thinking_mode": "low"
            },
            timeout=120
        )
        
        if r.status_code == 200:
            result = r.json()
            print(f"Success: {result.get('success')}")
            print(f"Answer preview: {result.get('answer', '')[:200]}...")
            print(f"Sources: {len(result.get('sources', []))} documents")
            
            print_result("Chat", result.get('success', False), "Got response from llamacpp")
            return result.get('success', False)
        else:
            print_result("Chat", False, f"Status {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print_result("Chat", False, str(e))
        return False


def test_api_stream_llamacpp():
    """Test SSE streaming with llamacpp provider"""
    print_header("TEST 6: SSE Streaming with LlamaCpp")
    
    try:
        r = requests.post(
            f"{API_URL}/rag/chat/stream",
            json={
                "query": "Jelaskan hukum perdata singkat",
                "session_id": "test-llamacpp-stream",
                "thinking_mode": "low"
            },
            stream=True,
            timeout=120
        )
        
        if r.status_code == 200:
            print("[Streaming response...]")
            token_count = 0
            full_text = ""
            
            for line in r.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = line_str[6:]
                        if data != '[DONE]':
                            try:
                                chunk = json.loads(data)
                                token = chunk.get('token', '')
                                full_text += token
                                token_count += 1
                                print(token, end='', flush=True)
                            except json.JSONDecodeError:
                                pass
            
            print(f"\n\nTokens received: {token_count}")
            print_result("Stream", token_count > 0, f"{token_count} tokens streamed")
            return token_count > 0
        else:
            print_result("Stream", False, f"Status {r.status_code}")
            return False
    except Exception as e:
        print_result("Stream", False, str(e))
        return False


def test_api_research_llamacpp():
    """Test deep research with llamacpp provider"""
    print_header("TEST 7: Deep Research with LlamaCpp")
    
    try:
        r = requests.post(
            f"{API_URL}/rag/research",
            json={
                "query": "Bagaimana prosedur pengajuan gugatan perdata?",
                "session_id": "test-llamacpp-research"
            },
            timeout=180  # Research takes longer
        )
        
        if r.status_code == 200:
            result = r.json()
            print(f"Success: {result.get('success')}")
            print(f"Answer preview: {result.get('answer', '')[:200]}...")
            print(f"Sources: {len(result.get('sources', []))} documents")
            
            print_result("Research", result.get('success', False), "Research completed")
            return result.get('success', False)
        else:
            print_result("Research", False, f"Status {r.status_code}")
            return False
    except Exception as e:
        print_result("Research", False, str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="LlamaCpp API Integration Test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--skip-stream", action="store_true", help="Skip streaming test")
    parser.add_argument("--skip-research", action="store_true", help="Skip research test")
    args = parser.parse_args()
    
    global BASE_URL, API_URL
    BASE_URL = args.base_url
    API_URL = f"{BASE_URL}/api/v1"
    
    print("\n" + "="*60)
    print("  🧪 LlamaCpp API Integration Test")
    print("="*60)
    print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    
    results = []
    
    # Health check
    if not test_api_health():
        print("\n⚠️ API server not running. Start with:")
        print("   python -m api.server --llm-provider llamacpp")
        return 1
    results.append(("API Health", True))
    
    # Provider tests
    results.append(("Providers List", test_api_llm_providers()))
    results.append(("LLM Status", test_api_llm_status()))
    results.append(("Switch to LlamaCpp", test_api_switch_to_llamacpp()))
    
    # Chat tests
    results.append(("RAG Chat", test_api_chat_llamacpp()))
    
    if not args.skip_stream:
        results.append(("SSE Streaming", test_api_stream_llamacpp()))
    
    if not args.skip_research:
        results.append(("Deep Research", test_api_research_llamacpp()))
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        emoji = "✅" if success else "❌"
        print(f"  {emoji} {name}")
    
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
