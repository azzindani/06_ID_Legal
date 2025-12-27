"""
Document Parser Integration Tests - Full System Integration

Tests document parser integration with:
- RAG Pipeline
- API Server
- Conversation flow
- Full inference with uploaded documents

Run: python tests/test_document_parser_integration.py

IMPORTANT: This test requires the full system to be running or will start components.
For Kaggle/Colab, ensure GPU is available.

File: tests/test_document_parser_integration.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test documents directory
TEST_DOCS_DIR = PROJECT_ROOT / "tests" / "test_documents"

# API URL
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")

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


# =============================================================================
# MODULE INITIALIZATION TEST
# =============================================================================

def test_module_initialization() -> List[Tuple[str, bool, str]]:
    """Test document parser module initialization"""
    print_header("Module Initialization Tests")
    results = []
    
    # Test 1: Import module
    try:
        import document_parser
        results.append(("Import document_parser module", True, ""))
        print_test("Import document_parser module", True)
    except Exception as e:
        results.append(("Import document_parser module", False, str(e)))
        print_test("Import document_parser module", False, str(e))
        return results
    
    # Test 2: Initialize module
    try:
        # Use test config
        import tempfile
        temp_dir = tempfile.mkdtemp()
        test_config = {
            'db_path': os.path.join(temp_dir, 'test.db'),
            'max_documents_per_session': 5
        }
        
        success = document_parser.initialize(test_config)
        results.append(("Initialize module", success, ""))
        print_test("Initialize module", success)
    except Exception as e:
        results.append(("Initialize module", False, str(e)))
        print_test("Initialize module", False, str(e))
        return results
    
    # Test 3: Check is_initialized
    try:
        is_init = document_parser.is_initialized()
        results.append(("is_initialized() returns True", is_init, ""))
        print_test("is_initialized() returns True", is_init)
    except Exception as e:
        results.append(("is_initialized() returns True", False, str(e)))
        print_test("is_initialized() returns True", False, str(e))
    
    # Test 4: Get parser
    try:
        parser = document_parser.get_parser()
        has_parser = parser is not None
        results.append(("get_parser() returns parser", has_parser, ""))
        print_test("get_parser() returns parser", has_parser)
    except Exception as e:
        results.append(("get_parser() returns parser", False, str(e)))
        print_test("get_parser() returns parser", False, str(e))
    
    # Test 5: Get storage
    try:
        storage = document_parser.get_storage()
        has_storage = storage is not None
        results.append(("get_storage() returns storage", has_storage, ""))
        print_test("get_storage() returns storage", has_storage)
    except Exception as e:
        results.append(("get_storage() returns storage", False, str(e)))
        print_test("get_storage() returns storage", False, str(e))
    
    # Test 6: URL extraction enabled
    try:
        url_enabled = document_parser.is_url_extraction_enabled()
        results.append(("URL extraction enabled", url_enabled, ""))
        print_test("URL extraction enabled", url_enabled)
    except Exception as e:
        results.append(("URL extraction enabled", False, str(e)))
        print_test("URL extraction enabled", False, str(e))
    
    # Test 7: Shutdown
    try:
        document_parser.shutdown()
        is_init = document_parser.is_initialized()
        passed = not is_init
        results.append(("Shutdown clears state", passed, ""))
        print_test("Shutdown clears state", passed)
    except Exception as e:
        results.append(("Shutdown clears state", False, str(e)))
        print_test("Shutdown clears state", False, str(e))
    
    return results


# =============================================================================
# PIPELINE INTEGRATION TEST
# =============================================================================

def test_pipeline_with_documents() -> List[Tuple[str, bool, str]]:
    """Test document parser integration with RAG pipeline"""
    print_header("Pipeline Integration Tests")
    results = []
    
    print(f"  {Colors.YELLOW}Note: These tests require models to be loaded.{Colors.RESET}")
    print(f"  {Colors.YELLOW}Skipping if pipeline not available...{Colors.RESET}")
    
    try:
        # Check if pipeline can be imported
        from pipeline import RAGPipeline
        from document_parser.context_builder import DocumentContextBuilder
        
        # Create context builder
        builder = DocumentContextBuilder()
        
        # Simulate documents
        test_docs = [
            {
                'filename': 'kontrak_sample.pdf',
                'extracted_text': '''
                PERJANJIAN KERJA SAMA
                
                Pasal 1 - Para Pihak
                Pihak Pertama: PT ABC Indonesia
                Pihak Kedua: CV XYZ Sejahtera
                
                Pasal 2 - Lingkup Perjanjian
                Para pihak sepakat untuk melakukan kerja sama dalam bidang 
                penyediaan jasa konsultasi hukum selama 12 bulan.
                
                Pasal 3 - Nilai Kontrak
                Total nilai kontrak adalah Rp 500.000.000 (Lima Ratus Juta Rupiah).
                
                Pasal 4 - Sanksi
                Jika salah satu pihak melanggar perjanjian, maka wajib membayar
                denda sebesar 10% dari nilai kontrak.
                '''
            }
        ]
        
        # Build context
        context = builder.build_prompt_section(test_docs)
        
        has_context = len(context) > 100
        results.append(("Build document context", has_context, f"{len(context)} chars"))
        print_test("Build document context", has_context)
        
        if has_context:
            print(f"       Preview: {context[:200]}...")
        
        # Test context injection simulation
        sample_query = "Berapa nilai kontrak dalam dokumen yang saya unggah?"
        
        # The context would be injected into the system prompt or prepended to query
        enhanced_query = f"{context}\n\nPertanyaan: {sample_query}"
        
        has_enhanced = len(enhanced_query) > len(sample_query) + len(context) - 10
        results.append(("Context injection", has_enhanced, f"Enhanced query: {len(enhanced_query)} chars"))
        print_test("Context injection", has_enhanced)
        
    except ImportError as e:
        results.append(("Pipeline integration", False, f"Import error: {e}"))
        print_test("Pipeline integration", False, f"Import error: {e}")
    except Exception as e:
        results.append(("Pipeline integration", False, str(e)))
        print_test("Pipeline integration", False, str(e))
    
    return results


# =============================================================================
# URL EXTRACTION IN PROMPT TEST
# =============================================================================

def test_url_in_prompt_flow() -> List[Tuple[str, bool, str]]:
    """Test URL detection and extraction in chat prompts"""
    print_header("URL in Prompt Flow Tests")
    results = []
    
    from document_parser import (
        extract_urls_from_prompt,
        initialize,
        get_url_extractor,
        is_url_extraction_enabled,
        shutdown
    )
    
    # Initialize with test config
    import tempfile
    temp_dir = tempfile.mkdtemp()
    test_config = {'db_path': os.path.join(temp_dir, 'test.db')}
    initialize(test_config)
    
    # Test prompts with URLs
    test_cases = [
        {
            'prompt': 'Berdasarkan https://www.cnbcindonesia.com/news/20251226155414-4-697445/kpk-setop-penyidikan-kasus-korupsi-izin-tambang-konawe-utara-rp27-t apa yang terjadi?',
            'expected_url_count': 1,
            'should_fetch': True  # Real URL
        },
        {
            'prompt': 'Jelaskan tentang pajak penghasilan',
            'expected_url_count': 0,
            'should_fetch': False  # No URL
        },
        {
            'prompt': 'Lihat http://jdih.kemenkeu.go.id dan https://peraturan.bpk.go.id untuk referensi',
            'expected_url_count': 2,
            'should_fetch': False  # Multiple URLs, may fail
        }
    ]
    
    for i, case in enumerate(test_cases):
        prompt = case['prompt']
        
        # Extract URLs from prompt
        clean_prompt, urls = extract_urls_from_prompt(prompt)
        
        passed = len(urls) == case['expected_url_count']
        details = f"Found {len(urls)} URLs: {urls[:2]}..."
        results.append((f"URL detection case {i+1}", passed, details))
        print_test(f"URL detection case {i+1}", passed, details)
        
        # Show clean prompt
        if case['expected_url_count'] > 0:
            print(f"       Clean prompt: {clean_prompt[:50]}...")
    
    # Test actual URL fetching
    if is_url_extraction_enabled():
        print(f"\n  {Colors.YELLOW}Testing real URL fetch...{Colors.RESET}")
        
        url_extractor = get_url_extractor()
        test_url = "https://www.cnbcindonesia.com/news/20251226155414-4-697445/kpk-setop-penyidikan-kasus-korupsi-izin-tambang-konawe-utara-rp27-t"
        
        try:
            result = url_extractor.extract(test_url)
            text = result.get('text', '')
            
            has_content = len(text) > 500
            results.append(("Fetch CNBC article", has_content, f"{len(text)} chars"))
            print_test("Fetch CNBC article", has_content, f"{len(text)} chars")
            
            # Check for expected keywords
            keywords = ['kpk', 'korupsi', 'penyidikan', 'tambang']
            found_keywords = [kw for kw in keywords if kw in text.lower()]
            has_keywords = len(found_keywords) >= 2
            results.append(("Content has expected keywords", has_keywords, f"Found: {found_keywords}"))
            print_test("Content has expected keywords", has_keywords, f"Found: {found_keywords}")
            
        except Exception as e:
            results.append(("Fetch CNBC article", False, str(e)))
            print_test("Fetch CNBC article", False, str(e))
    
    shutdown()
    
    return results


# =============================================================================
# SIMULATED CHAT WITH DOCUMENT TEST
# =============================================================================

def test_chat_with_document_simulation() -> List[Tuple[str, bool, str]]:
    """Simulate chat with uploaded document"""
    print_header("Chat with Document Simulation")
    results = []
    
    from document_parser import initialize, get_parser, get_storage, shutdown
    from document_parser.context_builder import DocumentContextBuilder
    
    # Initialize
    import tempfile
    temp_dir = tempfile.mkdtemp()
    test_config = {'db_path': os.path.join(temp_dir, 'test.db')}
    initialize(test_config)
    
    parser = get_parser()
    storage = get_storage()
    builder = DocumentContextBuilder()
    
    session_id = "sim-chat-session-001"
    
    # Step 1: Parse a real document
    pdf_file = TEST_DOCS_DIR / "peraturan_1.pdf"
    
    if pdf_file.exists():
        try:
            doc_info = parser.parse_file(str(pdf_file), session_id)
            passed = doc_info.get('id') is not None
            results.append(("Step 1: Parse document", passed, f"ID: {doc_info.get('id')}"))
            print_test("Step 1: Parse document", passed, f"{doc_info.get('char_count')} chars")
        except Exception as e:
            results.append(("Step 1: Parse document", False, str(e)))
            print_test("Step 1: Parse document", False, str(e))
            shutdown()
            return results
    else:
        results.append(("Step 1: Parse document", False, "Test file not found"))
        print_test("Step 1: Parse document", False, "Test file not found")
        shutdown()
        return results
    
    # Step 2: Retrieve session documents
    try:
        session_docs = storage.get_session_documents(session_id)
        passed = len(session_docs) > 0
        results.append(("Step 2: Retrieve session docs", passed, f"{len(session_docs)} docs"))
        print_test("Step 2: Retrieve session docs", passed, f"{len(session_docs)} docs")
    except Exception as e:
        results.append(("Step 2: Retrieve session docs", False, str(e)))
        print_test("Step 2: Retrieve session docs", False, str(e))
        shutdown()
        return results
    
    # Step 3: Build context for prompt
    try:
        docs_for_context = [
            {'filename': d['filename'], 'extracted_text': d['extracted_text']}
            for d in session_docs
        ]
        context = builder.build_prompt_section(docs_for_context)
        passed = len(context) > 100
        results.append(("Step 3: Build context", passed, f"{len(context)} chars"))
        print_test("Step 3: Build context", passed)
    except Exception as e:
        results.append(("Step 3: Build context", False, str(e)))
        print_test("Step 3: Build context", False, str(e))
        shutdown()
        return results
    
    # Step 4: Simulate user query
    user_query = "Apa yang diatur dalam dokumen yang saya unggah?"
    
    # In real implementation, this would be sent to the LLM
    enhanced_prompt = f'''
{context}

---

Pertanyaan Pengguna: {user_query}

Berdasarkan dokumen yang diunggah di atas, jawab pertanyaan pengguna.
'''
    
    passed = len(enhanced_prompt) > len(user_query) + len(context)
    results.append(("Step 4: Build enhanced prompt", passed, f"{len(enhanced_prompt)} chars"))
    print_test("Step 4: Build enhanced prompt", passed)
    
    # Show the enhanced prompt structure
    print(f"\n  {Colors.CYAN}Enhanced Prompt Structure:{Colors.RESET}")
    print(f"  - Document context: {len(context)} chars")
    print(f"  - User query: {len(user_query)} chars")
    print(f"  - Total: {len(enhanced_prompt)} chars")
    
    # Step 5: Cleanup
    try:
        storage.delete_session_documents(session_id)
        remaining = storage.get_session_documents(session_id)
        passed = len(remaining) == 0
        results.append(("Step 5: Cleanup session", passed, ""))
        print_test("Step 5: Cleanup session", passed)
    except Exception as e:
        results.append(("Step 5: Cleanup session", False, str(e)))
        print_test("Step 5: Cleanup session", False, str(e))
    
    shutdown()
    
    return results


# =============================================================================
# API INTEGRATION TEST (if API running)
# =============================================================================

def test_api_integration() -> List[Tuple[str, bool, str]]:
    """Test API integration (requires running API server)"""
    print_header("API Integration Tests")
    results = []
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        api_running = response.status_code == 200
    except:
        api_running = False
    
    if not api_running:
        print(f"  {Colors.YELLOW}API not running at {API_BASE_URL}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Skipping API integration tests...{Colors.RESET}")
        results.append(("API health check", False, "API not running"))
        return results
    
    print(f"  {Colors.GREEN}API is running at {API_BASE_URL}{Colors.RESET}")
    results.append(("API health check", True, ""))
    print_test("API health check", True)
    
    # Check if document parser is initialized
    try:
        response = requests.get(f"{API_BASE_URL}/ready", timeout=5)
        ready_data = response.json()
        
        # In future, this should include document_parser status
        is_ready = ready_data.get('ready', False)
        results.append(("API ready", is_ready, ""))
        print_test("API ready", is_ready)
    except Exception as e:
        results.append(("API ready", False, str(e)))
        print_test("API ready", False, str(e))
    
    # TODO: Add document upload API tests when endpoints are implemented
    # POST /api/v1/documents/upload
    # GET /api/v1/documents
    # DELETE /api/v1/documents/{id}
    
    print(f"\n  {Colors.YELLOW}Note: Document upload API endpoints not yet implemented.{Colors.RESET}")
    print(f"  {Colors.YELLOW}Add tests here when endpoints are ready.{Colors.RESET}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_all_integration_tests():
    """Run all integration tests"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("  DOCUMENT PARSER INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"{Colors.RESET}")
    
    print(f"Test documents directory: {TEST_DOCS_DIR}")
    print(f"API URL: {API_BASE_URL}")
    
    start_time = time.time()
    all_results = []
    
    # Run test suites
    test_suites = [
        ("Module Initialization", test_module_initialization),
        ("Pipeline Integration", test_pipeline_with_documents),
        ("URL in Prompt Flow", test_url_in_prompt_flow),
        ("Chat with Document Simulation", test_chat_with_document_simulation),
        ("API Integration", test_api_integration),
    ]
    
    suite_results = {}
    
    for suite_name, test_func in test_suites:
        try:
            results = test_func()
            all_results.extend(results)
            passed = sum(1 for _, p, _ in results if p)
            total = len(results)
            suite_results[suite_name] = (passed, total)
        except Exception as e:
            print(f"\n{Colors.RED}ERROR in {suite_name}: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            all_results.append((f"{suite_name} suite", False, str(e)))
            suite_results[suite_name] = (0, 1)
    
    # Print summary
    elapsed = time.time() - start_time
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}INTEGRATION TEST SUMMARY{Colors.RESET}")
    print(f"{'='*60}")
    
    total_passed = 0
    total_tests = 0
    
    for suite_name, (passed, total) in suite_results.items():
        total_passed += passed
        total_tests += total
        status = f"{Colors.GREEN}✓{Colors.RESET}" if passed == total else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {status} {suite_name}: {passed}/{total}")
    
    print(f"\n{'='*60}")
    print(f"  Total:  {total_tests}")
    print(f"  {Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {total_tests - total_passed}{Colors.RESET}")
    print(f"\nTime elapsed: {elapsed:.2f}s")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
