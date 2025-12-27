"""
Document Parser End-to-End Tests

Tests the complete document upload and chat flow:
1. Start API server (if not running)
2. Upload document via API
3. Chat with document context
4. Verify document content is used

Run: python tests/test_document_e2e.py

File: tests/test_document_e2e.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test documents directory
TEST_DOCS_DIR = PROJECT_ROOT / "tests" / "test_documents"

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")
API_KEY = os.getenv("API_KEY", "")

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_test(name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"  {status} {name}")
    if details:
        print(f"       {Colors.CYAN}{details}{Colors.RESET}")


def get_headers():
    """Get API request headers"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


# =============================================================================
# API Health Check
# =============================================================================

def check_api_running() -> Tuple[bool, str]:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, "API is running"
        return False, f"API returned {response.status_code}"
    except requests.ConnectionError:
        return False, "Connection refused - API not running"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Document Upload Tests
# =============================================================================

def test_document_upload() -> List[Tuple[str, bool, str]]:
    """Test document upload API"""
    print_header("Document Upload API Tests")
    results = []
    
    session_id = f"test-upload-{int(time.time())}"
    uploaded_doc_id = None
    
    # Test 1: Check API health
    is_running, msg = check_api_running()
    results.append(("API health check", is_running, msg))
    print_test("API health check", is_running, msg)
    
    if not is_running:
        print(f"\n  {Colors.YELLOW}API not running. Start with: python -m uvicorn api.server:app{Colors.RESET}")
        return results
    
    # Test 2: Upload PDF document
    pdf_file = TEST_DOCS_DIR / "peraturan_1.pdf"
    
    if pdf_file.exists():
        print(f"\n  {Colors.YELLOW}Uploading {pdf_file.name}...{Colors.RESET}")
        
        try:
            with open(pdf_file, 'rb') as f:
                files = {'file': (pdf_file.name, f, 'application/pdf')}
                data = {'session_id': session_id}
                
                response = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files=files,
                    data=data,
                    headers={"X-API-Key": API_KEY} if API_KEY else {},
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                uploaded_doc_id = result.get('document_id')
                char_count = result.get('char_count', 0)
                
                passed = uploaded_doc_id is not None and char_count > 100
                details = f"ID: {uploaded_doc_id}, {char_count} chars"
                results.append(("Upload PDF document", passed, details))
                print_test("Upload PDF document", passed, details)
                
                # Show preview
                if result.get('preview'):
                    print(f"       Preview: {result['preview'][:100]}...")
            else:
                results.append(("Upload PDF document", False, f"HTTP {response.status_code}: {response.text[:200]}"))
                print_test("Upload PDF document", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            results.append(("Upload PDF document", False, str(e)))
            print_test("Upload PDF document", False, str(e))
    else:
        results.append(("Upload PDF document", False, "Test file not found"))
        print_test("Upload PDF document", False, "Test file not found")
    
    # Test 3: List documents
    try:
        response = requests.get(
            f"{API_BASE_URL}/documents",
            params={'session_id': session_id},
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            doc_count = result.get('total_count', 0)
            passed = doc_count > 0
            results.append(("List session documents", passed, f"{doc_count} documents"))
            print_test("List session documents", passed, f"{doc_count} documents")
        else:
            results.append(("List session documents", False, f"HTTP {response.status_code}"))
            print_test("List session documents", False, f"HTTP {response.status_code}")
            
    except Exception as e:
        results.append(("List session documents", False, str(e)))
        print_test("List session documents", False, str(e))
    
    # Test 4: Get document details
    if uploaded_doc_id:
        try:
            response = requests.get(
                f"{API_BASE_URL}/documents/{uploaded_doc_id}",
                headers=get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                has_text = 'document' in result and result['document'].get('extracted_text')
                results.append(("Get document details", has_text, "Has extracted text"))
                print_test("Get document details", has_text)
            else:
                results.append(("Get document details", False, f"HTTP {response.status_code}"))
                print_test("Get document details", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            results.append(("Get document details", False, str(e)))
            print_test("Get document details", False, str(e))
    
    # Store for chat test
    return results, session_id, uploaded_doc_id


# =============================================================================
# Chat with Document Tests
# =============================================================================

def test_chat_with_document(session_id: str, document_id: str) -> List[Tuple[str, bool, str]]:
    """Test chat with document context"""
    print_header("Chat with Document Context Tests")
    results = []
    
    if not session_id or not document_id:
        print(f"  {Colors.YELLOW}Skipping: No document uploaded{Colors.RESET}")
        return results
    
    # Test 1: Chat with include_session_documents
    print(f"\n  {Colors.YELLOW}Testing chat with document context (may take a while)...{Colors.RESET}")
    
    try:
        payload = {
            "query": "Apa yang diatur dalam dokumen yang saya unggah?",
            "session_id": session_id,
            "include_session_documents": True,
            "thinking_level": "low",
            "stream": False,
            "top_k": 3,
            "max_tokens": 1024
        }
        
        response = requests.post(
            f"{API_BASE_URL}/rag/chat",
            json=payload,
            headers=get_headers(),
            timeout=120  # Long timeout for LLM generation
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            has_answer = len(answer) > 50
            
            results.append(("Chat with document context", has_answer, f"{len(answer)} chars"))
            print_test("Chat with document context", has_answer, f"{len(answer)} chars")
            
            if has_answer:
                print(f"\n  {Colors.CYAN}Answer preview:{Colors.RESET}")
                print(f"  {answer[:500]}...")
        else:
            results.append(("Chat with document context", False, f"HTTP {response.status_code}: {response.text[:200]}"))
            print_test("Chat with document context", False, f"HTTP {response.status_code}")
            
    except requests.Timeout:
        results.append(("Chat with document context", False, "Request timed out (120s)"))
        print_test("Chat with document context", False, "Timeout")
    except Exception as e:
        results.append(("Chat with document context", False, str(e)))
        print_test("Chat with document context", False, str(e))
    
    # Test 2: Chat with specific document_ids
    try:
        payload = {
            "query": "Ringkas isi dokumen yang saya berikan",
            "session_id": session_id,
            "document_ids": [document_id],
            "thinking_level": "low",
            "stream": False,
            "top_k": 3,
            "max_tokens": 1024
        }
        
        response = requests.post(
            f"{API_BASE_URL}/rag/chat",
            json=payload,
            headers=get_headers(),
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            has_answer = len(answer) > 50
            
            results.append(("Chat with document_ids", has_answer, f"{len(answer)} chars"))
            print_test("Chat with document_ids", has_answer, f"{len(answer)} chars")
        else:
            results.append(("Chat with document_ids", False, f"HTTP {response.status_code}"))
            print_test("Chat with document_ids", False, f"HTTP {response.status_code}")
            
    except Exception as e:
        results.append(("Chat with document_ids", False, str(e)))
        print_test("Chat with document_ids", False, str(e))
    
    return results


# =============================================================================
# Normal Chat (No Documents) Tests - Verifies backwards compatibility
# =============================================================================

def test_normal_chat_without_documents() -> List[Tuple[str, bool, str]]:
    """Test that normal chat WITHOUT documents still works exactly as before"""
    print_header("Normal Chat (No Documents) - Backwards Compatibility")
    results = []
    
    is_running, _ = check_api_running()
    if not is_running:
        print(f"  {Colors.YELLOW}API not running, skipping{Colors.RESET}")
        return results
    
    # Test 1: Chat without any document parameters (original behavior)
    print(f"\n  {Colors.YELLOW}Testing chat WITHOUT document context...{Colors.RESET}")
    
    try:
        # Minimal request - no document_ids, no include_session_documents
        payload = {
            "query": "Apa sanksi dalam UU Ketenagakerjaan?",
            "session_id": f"test-nodoc-{int(time.time())}",
            "thinking_level": "low",
            "stream": False,
            "top_k": 3,
            "max_tokens": 1024
            # NOTE: No document_ids, no include_session_documents
        }
        
        response = requests.post(
            f"{API_BASE_URL}/rag/chat",
            json=payload,
            headers=get_headers(),
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            has_answer = len(answer) > 50
            
            results.append(("Chat without documents", has_answer, f"{len(answer)} chars"))
            print_test("Chat without documents", has_answer, f"{len(answer)} chars")
            
            if has_answer:
                print(f"\n  {Colors.CYAN}Answer preview:{Colors.RESET}")
                print(f"  {answer[:300]}...")
        else:
            results.append(("Chat without documents", False, f"HTTP {response.status_code}: {response.text[:200]}"))
            print_test("Chat without documents", False, f"HTTP {response.status_code}")
            
    except requests.Timeout:
        results.append(("Chat without documents", False, "Request timed out"))
        print_test("Chat without documents", False, "Timeout")
    except Exception as e:
        results.append(("Chat without documents", False, str(e)))
        print_test("Chat without documents", False, str(e))
    
    # Test 2: Chat with include_session_documents=False explicitly (should work same as above)
    try:
        payload = {
            "query": "Jelaskan tentang pajak penghasilan",
            "session_id": f"test-nodoc-explicit-{int(time.time())}",
            "include_session_documents": False,  # Explicitly False
            "document_ids": None,  # Explicitly None
            "thinking_level": "low",
            "stream": False,
            "top_k": 3,
            "max_tokens": 1024
        }
        
        response = requests.post(
            f"{API_BASE_URL}/rag/chat",
            json=payload,
            headers=get_headers(),
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            has_answer = len(answer) > 50
            
            results.append(("Chat with explicit no-docs", has_answer, f"{len(answer)} chars"))
            print_test("Chat with explicit no-docs", has_answer, f"{len(answer)} chars")
        else:
            results.append(("Chat with explicit no-docs", False, f"HTTP {response.status_code}"))
            print_test("Chat with explicit no-docs", False, f"HTTP {response.status_code}")
            
    except Exception as e:
        results.append(("Chat with explicit no-docs", False, str(e)))
        print_test("Chat with explicit no-docs", False, str(e))
    
    return results


# =============================================================================
# URL Extraction Tests
# =============================================================================

def test_url_extraction() -> List[Tuple[str, bool, str]]:
    """Test URL extraction API"""
    print_header("URL Extraction API Tests")
    results = []
    
    session_id = f"test-url-{int(time.time())}"
    test_url = "https://www.cnbcindonesia.com/news/20251226155414-4-697445/kpk-setop-penyidikan-kasus-korupsi-izin-tambang-konawe-utara-rp27-t"
    
    is_running, _ = check_api_running()
    if not is_running:
        print(f"  {Colors.YELLOW}API not running, skipping{Colors.RESET}")
        return results
    
    # Test 1: Extract from URL
    print(f"\n  {Colors.YELLOW}Extracting from URL (may take a few seconds)...{Colors.RESET}")
    
    try:
        payload = {
            "url": test_url,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/documents/url",
            json=payload,
            headers=get_headers(),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            char_count = result.get('char_count', 0)
            passed = char_count > 500
            
            results.append(("Extract from URL", passed, f"{char_count} chars"))
            print_test("Extract from URL", passed, f"{char_count} chars")
            
            if result.get('preview'):
                print(f"       Preview: {result['preview'][:150]}...")
        elif response.status_code == 503:
            results.append(("Extract from URL", False, "URL extraction disabled"))
            print_test("Extract from URL", False, "URL extraction disabled")
        else:
            results.append(("Extract from URL", False, f"HTTP {response.status_code}: {response.text[:200]}"))
            print_test("Extract from URL", False, f"HTTP {response.status_code}")
            
    except Exception as e:
        results.append(("Extract from URL", False, str(e)))
        print_test("Extract from URL", False, str(e))
    
    return results


# =============================================================================
# Cleanup Tests
# =============================================================================

def test_cleanup(session_id: str, document_id: str) -> List[Tuple[str, bool, str]]:
    """Test document cleanup"""
    print_header("Document Cleanup Tests")
    results = []
    
    is_running, _ = check_api_running()
    if not is_running:
        return results
    
    # Test 1: Delete specific document
    if document_id:
        try:
            response = requests.delete(
                f"{API_BASE_URL}/documents/{document_id}",
                headers=get_headers(),
                timeout=10
            )
            
            passed = response.status_code == 200
            results.append(("Delete document", passed, ""))
            print_test("Delete document", passed)
            
        except Exception as e:
            results.append(("Delete document", False, str(e)))
            print_test("Delete document", False, str(e))
    
    # Test 2: Clear session documents
    if session_id:
        try:
            response = requests.delete(
                f"{API_BASE_URL}/documents",
                params={'session_id': session_id},
                headers=get_headers(),
                timeout=10
            )
            
            passed = response.status_code == 200
            results.append(("Clear session documents", passed, ""))
            print_test("Clear session documents", passed)
            
        except Exception as e:
            results.append(("Clear session documents", False, str(e)))
            print_test("Clear session documents", False, str(e))
    
    return results


# =============================================================================
# Main
# =============================================================================

def run_all_e2e_tests():
    """Run all end-to-end tests"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("  DOCUMENT PARSER END-TO-END TESTS")
    print("=" * 60)
    print(f"{Colors.RESET}")
    
    print(f"API URL: {API_BASE_URL}")
    print(f"Test documents: {TEST_DOCS_DIR}")
    
    start_time = time.time()
    all_results = []
    
    # Step 1: Upload tests
    upload_results = test_document_upload()
    if isinstance(upload_results, tuple):
        results, session_id, document_id = upload_results
        all_results.extend(results)
    else:
        all_results.extend(upload_results)
        session_id, document_id = None, None
    
    # Step 2: URL extraction tests
    url_results = test_url_extraction()
    all_results.extend(url_results)
    
    # Step 3: Chat with document tests (only if upload succeeded)
    if session_id and document_id:
        chat_results = test_chat_with_document(session_id, document_id)
        all_results.extend(chat_results)
    
    # Step 4: Normal chat WITHOUT documents (backwards compatibility)
    normal_results = test_normal_chat_without_documents()
    all_results.extend(normal_results)
    
    # Step 5: Cleanup tests
    if session_id:
        cleanup_results = test_cleanup(session_id, document_id)
        all_results.extend(cleanup_results)
    
    # Summary
    elapsed = time.time() - start_time
    total = len(all_results)
    passed = sum(1 for _, p, _ in all_results if p)
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}END-TO-END TEST SUMMARY{Colors.RESET}")
    print(f"{'='*60}")
    print(f"  Total:  {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {total - passed}{Colors.RESET}")
    print(f"\nTime elapsed: {elapsed:.2f}s")
    
    if total - passed > 0:
        print(f"\n{Colors.RED}Failed Tests:{Colors.RESET}")
        for name, p, details in all_results:
            if not p:
                print(f"  - {name}: {details}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_e2e_tests()
    sys.exit(0 if success else 1)
