"""
Document Parser Tests - Comprehensive Test Suite

This module contains simulation tests (not pytest) for the document parser.
Run directly: python tests/test_document_parser.py

Tests included:
1. Extractor unit tests (PDF, DOCX, HTML, etc.)
2. URL extraction tests
3. Storage tests
4. Parser integration tests

File: tests/test_document_parser.py
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test documents directory
TEST_DOCS_DIR = PROJECT_ROOT / "tests" / "test_documents"

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
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"  {status} {name}")
    if details:
        print(f"       {Colors.CYAN}{details}{Colors.RESET}")


def print_summary(results: List[Tuple[str, bool, str]]):
    """Print test summary"""
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.RESET}")
    print(f"{'='*60}")
    print(f"  Total:  {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
    
    if failed > 0:
        print(f"\n{Colors.RED}Failed Tests:{Colors.RESET}")
        for name, passed, details in results:
            if not passed:
                print(f"  - {name}: {details}")
    
    return failed == 0


# =============================================================================
# EXTRACTOR TESTS
# =============================================================================

def test_pdf_extractor() -> List[Tuple[str, bool, str]]:
    """Test PDF extraction"""
    print_header("PDF Extractor Tests")
    results = []
    
    from document_parser.extractors import PDFExtractor
    
    extractor = PDFExtractor()
    
    # Test 1: Check availability
    available = extractor.is_available()
    results.append(("PDF extractor available", available, "" if available else "pypdf2/pdfplumber not installed"))
    print_test("PDF extractor available", available)
    
    if not available:
        return results
    
    # Test 2: Simple PDF
    pdf_files = [
        "peraturan_1.pdf",
        "contract_sample_1.pdf",
        "putusan_mahkamah_agung_1.pdf"
    ]
    
    for pdf_file in pdf_files:
        pdf_path = TEST_DOCS_DIR / pdf_file
        if pdf_path.exists():
            try:
                result = extractor.extract(str(pdf_path))
                has_text = len(result.get('text', '')) > 100
                passed = has_text and result.get('page_count', 0) > 0
                details = f"{len(result['text'])} chars, {result['page_count']} pages"
                results.append((f"Extract {pdf_file}", passed, details))
                print_test(f"Extract {pdf_file}", passed, details)
            except Exception as e:
                results.append((f"Extract {pdf_file}", False, str(e)))
                print_test(f"Extract {pdf_file}", False, str(e))
        else:
            results.append((f"Extract {pdf_file}", False, "File not found"))
            print_test(f"Extract {pdf_file}", False, "File not found")
    
    # Test 3: PDF with tables
    table_pdf = TEST_DOCS_DIR / "peraturan_tabel_1.pdf"
    if table_pdf.exists():
        try:
            result = extractor.extract(str(table_pdf))
            has_text = len(result.get('text', '')) > 100
            results.append(("Extract PDF with tables", has_text, f"{len(result['text'])} chars"))
            print_test("Extract PDF with tables", has_text, f"{len(result['text'])} chars")
        except Exception as e:
            results.append(("Extract PDF with tables", False, str(e)))
            print_test("Extract PDF with tables", False, str(e))
    
    return results


def test_docx_extractor() -> List[Tuple[str, bool, str]]:
    """Test DOCX extraction"""
    print_header("DOCX Extractor Tests")
    results = []
    
    from document_parser.extractors import DOCXExtractor
    
    extractor = DOCXExtractor()
    
    # Test 1: Check availability
    available = extractor.is_available()
    results.append(("DOCX extractor available", available, "" if available else "python-docx not installed"))
    print_test("DOCX extractor available", available)
    
    if not available:
        return results
    
    # Test 2: Extract DOCX
    docx_file = TEST_DOCS_DIR / "surat_kuasa_1.docx"
    if docx_file.exists():
        try:
            result = extractor.extract(str(docx_file))
            has_text = len(result.get('text', '')) > 50
            details = f"{len(result['text'])} chars"
            results.append(("Extract surat_kuasa_1.docx", has_text, details))
            print_test("Extract surat_kuasa_1.docx", has_text, details)
            
            # Print sample
            if has_text:
                print(f"       Sample: {result['text'][:200]}...")
        except Exception as e:
            results.append(("Extract surat_kuasa_1.docx", False, str(e)))
            print_test("Extract surat_kuasa_1.docx", False, str(e))
    
    return results


def test_html_extractor() -> List[Tuple[str, bool, str]]:
    """Test HTML extraction"""
    print_header("HTML Extractor Tests")
    results = []
    
    from document_parser.extractors import HTMLExtractor
    
    extractor = HTMLExtractor()
    
    # Test 1: Check availability (BeautifulSoup)
    available = extractor.is_available()
    results.append(("HTML extractor available", available, "" if available else "BeautifulSoup not installed (will use regex fallback)"))
    print_test("HTML extractor available", available, "Using BeautifulSoup" if available else "Using regex fallback")
    
    # Test 2: Extract HTML file
    html_file = TEST_DOCS_DIR / "article_1.html"
    if html_file.exists():
        try:
            result = extractor.extract(str(html_file))
            has_text = len(result.get('text', '')) > 100
            details = f"{len(result['text'])} chars"
            results.append(("Extract article_1.html", has_text, details))
            print_test("Extract article_1.html", has_text, details)
            
            # Print sample
            if has_text:
                print(f"       Sample: {result['text'][:200]}...")
        except Exception as e:
            results.append(("Extract article_1.html", False, str(e)))
            print_test("Extract article_1.html", False, str(e))
    
    return results


def test_image_extractor() -> List[Tuple[str, bool, str]]:
    """Test Image/OCR extraction"""
    print_header("Image/OCR Extractor Tests")
    results = []
    
    from document_parser.extractors import ImageExtractor
    
    extractor = ImageExtractor()
    
    # Test 1: Check OCR availability
    providers = extractor.get_available_providers()
    tesseract_available = providers.get('tesseract', False)
    easyocr_available = providers.get('easyocr', False)
    
    any_ocr = tesseract_available or easyocr_available
    details = f"Tesseract: {'✓' if tesseract_available else '✗'}, EasyOCR: {'✓' if easyocr_available else '✗'}"
    results.append(("OCR provider available", any_ocr, details))
    print_test("OCR provider available", any_ocr, details)
    
    if not any_ocr:
        print(f"       {Colors.YELLOW}Skip image tests: No OCR provider available{Colors.RESET}")
        print(f"       {Colors.YELLOW}Install: pip install pytesseract (+ tesseract binary) or pip install easyocr{Colors.RESET}")
        return results
    
    # Test 2: Extract text from real image (surat_somasi_1.jpg)
    image_file = TEST_DOCS_DIR / "surat_somasi_1.jpg"
    
    if image_file.exists():
        print(f"\n  {Colors.YELLOW}Running OCR on surat_somasi_1.jpg (may take a moment)...{Colors.RESET}")
        
        try:
            result = extractor.extract(str(image_file))
            text = result.get('text', '')
            has_text = len(text) > 50
            
            # Check for expected Indonesian legal terms
            legal_terms = ['somasi', 'surat', 'kuasa', 'kepada', 'dengan', 'perihal']
            found_terms = [term for term in legal_terms if term.lower() in text.lower()]
            
            passed = has_text
            details = f"{len(text)} chars, method: {result.get('method')}, terms found: {found_terms[:3]}"
            results.append(("OCR surat_somasi_1.jpg", passed, details))
            print_test("OCR surat_somasi_1.jpg", passed, details)
            
            if has_text:
                # Show sample of extracted text
                sample = text[:300].replace('\n', ' ')
                print(f"       Sample: {sample}...")
                
        except Exception as e:
            results.append(("OCR surat_somasi_1.jpg", False, str(e)))
            print_test("OCR surat_somasi_1.jpg", False, str(e))
    else:
        results.append(("OCR surat_somasi_1.jpg", False, "File not found"))
        print_test("OCR surat_somasi_1.jpg", False, "File not found")
    
    # Test 3: Check image preprocessing
    try:
        from PIL import Image
        
        # Open and check image can be loaded
        img = Image.open(str(image_file))
        img_info = f"{img.size[0]}x{img.size[1]}, mode: {img.mode}"
        
        results.append(("Load image for preprocessing", True, img_info))
        print_test("Load image for preprocessing", True, img_info)
    except ImportError:
        results.append(("Load image for preprocessing", False, "Pillow not installed"))
        print_test("Load image for preprocessing", False, "Pillow not installed")
    except Exception as e:
        results.append(("Load image for preprocessing", False, str(e)))
        print_test("Load image for preprocessing", False, str(e))
    
    # Test 4: Note about scanned PDF
    scanned_pdf = TEST_DOCS_DIR / "peraturan_scan_1.pdf"
    if scanned_pdf.exists():
        print(f"\n       {Colors.YELLOW}Note: peraturan_scan_1.pdf is available for scanned PDF OCR testing{Colors.RESET}")
        print(f"       {Colors.YELLOW}(Requires PDF-to-image conversion, not implemented in basic extractor){Colors.RESET}")
    
    return results


# =============================================================================
# URL EXTRACTION TESTS
# =============================================================================

def test_url_extractor() -> List[Tuple[str, bool, str]]:
    """Test URL extraction"""
    print_header("URL Extractor Tests")
    results = []
    
    from document_parser.extractors import URLExtractor, find_urls, extract_urls_from_prompt
    
    extractor = URLExtractor(timeout=15)
    
    # Test 1: Check availability
    available = extractor.is_available()
    results.append(("URL extractor available", available, "" if available else "requests library not installed"))
    print_test("URL extractor available", available)
    
    if not available:
        return results
    
    # Test 2: URL detection in prompts
    test_prompts = [
        ("Analisis https://example.com/doc.pdf ini", ["https://example.com/doc.pdf"]),
        ("Berdasarkan http://jdih.go.id/uu.pdf dan https://google.com", ["http://jdih.go.id/uu.pdf", "https://google.com"]),
        ("Tidak ada URL di sini", []),
    ]
    
    for prompt, expected_urls in test_prompts:
        found_urls = find_urls(prompt)
        passed = set(found_urls) == set(expected_urls)
        details = f"Found: {found_urls}"
        results.append((f"URL detection: '{prompt[:30]}...'", passed, details))
        print_test(f"URL detection: '{prompt[:30]}...'", passed, details)
    
    # Test 3: Extract URLs from prompt
    test_prompt = "Berdasarkan https://example.com/legal.pdf jelaskan sanksinya"
    clean_prompt, urls = extract_urls_from_prompt(test_prompt)
    passed = len(urls) == 1 and "https://example.com/legal.pdf" in urls
    results.append(("Extract URLs from prompt", passed, f"Clean: '{clean_prompt}', URLs: {urls}"))
    print_test("Extract URLs from prompt", passed, f"Clean: '{clean_prompt}'")
    
    # Test 4: URL validation (block private IPs)
    blocked_urls = [
        "http://localhost/secret",
        "http://127.0.0.1/admin",
        "http://192.168.1.1/config",
        "http://10.0.0.1/internal",
    ]
    
    for url in blocked_urls:
        is_valid, error = extractor.is_valid_url(url)
        passed = not is_valid  # Should be blocked
        results.append((f"Block private URL: {url}", passed, error if not is_valid else "Not blocked!"))
        print_test(f"Block private URL: {url}", passed, "Blocked" if passed else "NOT BLOCKED!")
    
    # Test 5: Real URL extraction (CNBC Indonesia)
    real_url = "https://www.cnbcindonesia.com/news/20251226155414-4-697445/kpk-setop-penyidikan-kasus-korupsi-izin-tambang-konawe-utara-rp27-t"
    print(f"\n  {Colors.YELLOW}Testing real URL (may take a few seconds)...{Colors.RESET}")
    
    try:
        result = extractor.extract(real_url)
        text = result.get('text', '')
        has_content = len(text) > 500
        
        # Check for expected content
        has_kpk = 'kpk' in text.lower() or 'korupsi' in text.lower()
        
        passed = has_content and has_kpk
        details = f"{len(text)} chars, method: {result.get('method')}"
        results.append(("Extract CNBC article", passed, details))
        print_test("Extract CNBC article", passed, details)
        
        if has_content:
            # Show sample of extracted text
            sample = text[:300].replace('\n', ' ')
            print(f"       Sample: {sample}...")
            
    except Exception as e:
        results.append(("Extract CNBC article", False, str(e)))
        print_test("Extract CNBC article", False, str(e))
    
    return results


# =============================================================================
# STORAGE TESTS
# =============================================================================

def test_document_storage() -> List[Tuple[str, bool, str]]:
    """Test document storage"""
    print_header("Document Storage Tests")
    results = []
    
    import tempfile
    from document_parser.storage import DocumentStorage
    
    # Create temp database for testing
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_sessions.db")
    
    config = {
        'db_path': db_path,
        'max_documents_per_session': 3,
        'document_ttl_hours': 1
    }
    
    storage = DocumentStorage(config)
    
    # Test 1: Initialize
    try:
        storage.initialize()
        results.append(("Storage initialization", True, db_path))
        print_test("Storage initialization", True, db_path)
    except Exception as e:
        results.append(("Storage initialization", False, str(e)))
        print_test("Storage initialization", False, str(e))
        return results
    
    # Test 2: Store document
    test_session = "test-session-001"
    test_text = "Ini adalah teks dokumen hukum Indonesia untuk pengujian."
    
    try:
        doc_info = storage.store_document(
            session_id=test_session,
            filename="test_doc.pdf",
            extracted_text=test_text,
            format="pdf",
            original_size_bytes=1024,
            page_count=1,
            extraction_method="test"
        )
        
        has_id = 'id' in doc_info
        results.append(("Store document", has_id, f"ID: {doc_info.get('id')}"))
        print_test("Store document", has_id, f"ID: {doc_info.get('id')}")
        
        doc_id = doc_info.get('id')
    except Exception as e:
        results.append(("Store document", False, str(e)))
        print_test("Store document", False, str(e))
        return results
    
    # Test 3: Get document
    try:
        retrieved = storage.get_document(doc_id)
        passed = retrieved is not None and retrieved.get('extracted_text') == test_text
        results.append(("Retrieve document", passed, f"Text matches: {passed}"))
        print_test("Retrieve document", passed)
    except Exception as e:
        results.append(("Retrieve document", False, str(e)))
        print_test("Retrieve document", False, str(e))
    
    # Test 4: Get session documents
    try:
        docs = storage.get_session_documents(test_session)
        passed = len(docs) == 1
        results.append(("Get session documents", passed, f"Count: {len(docs)}"))
        print_test("Get session documents", passed, f"Count: {len(docs)}")
    except Exception as e:
        results.append(("Get session documents", False, str(e)))
        print_test("Get session documents", False, str(e))
    
    # Test 5: Deduplication (store same content again)
    try:
        doc_info2 = storage.store_document(
            session_id=test_session,
            filename="test_doc_copy.pdf",
            extracted_text=test_text,  # Same text
            format="pdf",
            original_size_bytes=1024,
            page_count=1,
            extraction_method="test"
        )
        
        # Should return existing document (dedup)
        passed = doc_info2.get('id') == doc_id
        results.append(("Deduplication", passed, "Returned existing doc" if passed else "Created duplicate"))
        print_test("Deduplication", passed, "Returned existing doc" if passed else "Created duplicate")
    except Exception as e:
        results.append(("Deduplication", False, str(e)))
        print_test("Deduplication", False, str(e))
    
    # Test 6: Document limit
    try:
        # Store 2 more documents to hit limit (max=3)
        for i in range(2):
            storage.store_document(
                session_id=test_session,
                filename=f"extra_doc_{i}.pdf",
                extracted_text=f"Extra document {i} content",
                format="pdf",
                original_size_bytes=512,
                page_count=1,
                extraction_method="test"
            )
        
        # Try to store 4th document (should fail)
        from document_parser.exceptions import DocumentLimitExceededError
        try:
            storage.store_document(
                session_id=test_session,
                filename="overflow_doc.pdf",
                extracted_text="This should fail",
                format="pdf",
                original_size_bytes=512,
                page_count=1,
                extraction_method="test"
            )
            passed = False
            details = "Limit not enforced!"
        except DocumentLimitExceededError:
            passed = True
            details = "Limit enforced correctly"
        
        results.append(("Document limit enforcement", passed, details))
        print_test("Document limit enforcement", passed, details)
    except Exception as e:
        results.append(("Document limit enforcement", False, str(e)))
        print_test("Document limit enforcement", False, str(e))
    
    # Test 7: Delete document
    try:
        deleted = storage.delete_document(doc_id)
        passed = deleted is True
        results.append(("Delete document", passed, ""))
        print_test("Delete document", passed)
    except Exception as e:
        results.append(("Delete document", False, str(e)))
        print_test("Delete document", False, str(e))
    
    # Test 8: Get statistics
    try:
        stats = storage.get_statistics()
        has_stats = 'total_documents' in stats
        results.append(("Get statistics", has_stats, f"Total: {stats.get('total_documents')}"))
        print_test("Get statistics", has_stats, f"Total: {stats.get('total_documents')}")
    except Exception as e:
        results.append(("Get statistics", False, str(e)))
        print_test("Get statistics", False, str(e))
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return results


# =============================================================================
# PARSER INTEGRATION TESTS
# =============================================================================

def test_document_parser() -> List[Tuple[str, bool, str]]:
    """Test full document parser integration"""
    print_header("Document Parser Integration Tests")
    results = []
    
    import tempfile
    from document_parser.parser import DocumentParser
    from document_parser.storage import DocumentStorage
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_sessions.db")
    
    config = {
        'db_path': db_path,
        'max_documents_per_session': 10,
        'max_file_size_mb': 5,
        'max_chars_per_document': 50000,
        'temp_upload_dir': os.path.join(temp_dir, 'uploads')
    }
    
    # Initialize storage first (parser depends on it)
    storage = DocumentStorage(config)
    storage.initialize()
    
    # Monkey-patch get_storage to use our test storage
    import document_parser
    original_get_storage = document_parser.get_storage
    document_parser.get_storage = lambda: storage
    
    parser = DocumentParser(config)
    
    # Test 1: Initialize parser
    try:
        parser.initialize()
        results.append(("Parser initialization", True, ""))
        print_test("Parser initialization", True)
    except Exception as e:
        results.append(("Parser initialization", False, str(e)))
        print_test("Parser initialization", False, str(e))
        return results
    
    # Test 2: Get supported formats
    try:
        formats = parser.get_supported_formats()
        has_formats = len(formats) > 5
        results.append(("Get supported formats", has_formats, f"{len(formats)} formats"))
        print_test("Get supported formats", has_formats, f"{formats}")
    except Exception as e:
        results.append(("Get supported formats", False, str(e)))
        print_test("Get supported formats", False, str(e))
    
    # Test 3: Get extraction capabilities
    try:
        caps = parser.get_extraction_capabilities()
        has_caps = '.pdf' in caps
        available_count = sum(1 for c in caps.values() if c.get('available'))
        results.append(("Get extraction capabilities", has_caps, f"{available_count} available"))
        print_test("Get extraction capabilities", has_caps, f"{available_count} extractors available")
    except Exception as e:
        results.append(("Get extraction capabilities", False, str(e)))
        print_test("Get extraction capabilities", False, str(e))
    
    # Test 4: Parse real PDF
    test_session = "parser-test-session"
    pdf_file = TEST_DOCS_DIR / "peraturan_1.pdf"
    
    if pdf_file.exists():
        try:
            doc_info = parser.parse_file(str(pdf_file), test_session)
            has_text = doc_info.get('char_count', 0) > 100
            results.append((
                "Parse PDF file",
                has_text,
                f"ID: {doc_info.get('id')}, {doc_info.get('char_count')} chars"
            ))
            print_test("Parse PDF file", has_text, f"{doc_info.get('char_count')} chars")
            
            # Show preview
            if doc_info.get('preview'):
                print(f"       Preview: {doc_info['preview'][:150]}...")
        except Exception as e:
            results.append(("Parse PDF file", False, str(e)))
            print_test("Parse PDF file", False, str(e))
    
    # Test 5: Parse DOCX
    docx_file = TEST_DOCS_DIR / "surat_kuasa_1.docx"
    
    if docx_file.exists():
        try:
            doc_info = parser.parse_file(str(docx_file), test_session)
            has_text = doc_info.get('char_count', 0) > 50
            results.append((
                "Parse DOCX file",
                has_text,
                f"{doc_info.get('char_count')} chars"
            ))
            print_test("Parse DOCX file", has_text, f"{doc_info.get('char_count')} chars")
        except Exception as e:
            results.append(("Parse DOCX file", False, str(e)))
            print_test("Parse DOCX file", False, str(e))
    
    # Test 6: Parse HTML
    html_file = TEST_DOCS_DIR / "article_1.html"
    
    if html_file.exists():
        try:
            doc_info = parser.parse_file(str(html_file), test_session)
            has_text = doc_info.get('char_count', 0) > 100
            results.append((
                "Parse HTML file", 
                has_text,
                f"{doc_info.get('char_count')} chars"
            ))
            print_test("Parse HTML file", has_text, f"{doc_info.get('char_count')} chars")
        except Exception as e:
            results.append(("Parse HTML file", False, str(e)))
            print_test("Parse HTML file", False, str(e))
    
    # Restore original get_storage
    document_parser.get_storage = original_get_storage
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return results


# =============================================================================
# CONTEXT BUILDER TESTS
# =============================================================================

def test_context_builder() -> List[Tuple[str, bool, str]]:
    """Test context builder"""
    print_header("Context Builder Tests")
    results = []
    
    from document_parser.context_builder import DocumentContextBuilder
    
    builder = DocumentContextBuilder()
    
    # Test 1: Build context from documents
    test_docs = [
        {'filename': 'uu_pajak.pdf', 'extracted_text': 'Undang-Undang tentang Pajak Penghasilan...'},
        {'filename': 'pp_pelaksanaan.pdf', 'extracted_text': 'Peraturan Pemerintah tentang pelaksanaan...'},
    ]
    
    try:
        context = builder.build_context(test_docs)
        has_content = len(context) > 50
        has_headers = 'Dokumen' in context
        passed = has_content and has_headers
        results.append(("Build context", passed, f"{len(context)} chars"))
        print_test("Build context", passed, f"{len(context)} chars")
        
        if has_content:
            print(f"       Sample: {context[:200]}...")
    except Exception as e:
        results.append(("Build context", False, str(e)))
        print_test("Build context", False, str(e))
    
    # Test 2: Build prompt section
    try:
        section = builder.build_prompt_section(test_docs)
        has_header = 'DOKUMEN YANG DIUNGGAH' in section
        results.append(("Build prompt section", has_header, ""))
        print_test("Build prompt section", has_header)
    except Exception as e:
        results.append(("Build prompt section", False, str(e)))
        print_test("Build prompt section", False, str(e))
    
    # Test 3: Get document summary
    try:
        summary = builder.get_document_summary(test_docs)
        has_emoji = '📄' in summary
        results.append(("Get document summary", has_emoji, ""))
        print_test("Get document summary", has_emoji)
        print(f"       {summary}")
    except Exception as e:
        results.append(("Get document summary", False, str(e)))
        print_test("Get document summary", False, str(e))
    
    # Test 4: Empty documents
    try:
        empty_context = builder.build_context([])
        passed = empty_context == ""
        results.append(("Handle empty documents", passed, "Returns empty string"))
        print_test("Handle empty documents", passed)
    except Exception as e:
        results.append(("Handle empty documents", False, str(e)))
        print_test("Handle empty documents", False, str(e))
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all document parser tests"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("  DOCUMENT PARSER TEST SUITE")
    print("=" * 60)
    print(f"{Colors.RESET}")
    
    print(f"Test documents directory: {TEST_DOCS_DIR}")
    print(f"Documents found: {len(list(TEST_DOCS_DIR.glob('*')))}")
    
    start_time = time.time()
    all_results = []
    
    # Run all test suites
    test_suites = [
        ("PDF Extractor", test_pdf_extractor),
        ("DOCX Extractor", test_docx_extractor),
        ("HTML Extractor", test_html_extractor),
        ("Image/OCR Extractor", test_image_extractor),
        ("URL Extractor", test_url_extractor),
        ("Document Storage", test_document_storage),
        ("Document Parser", test_document_parser),
        ("Context Builder", test_context_builder),
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
            all_results.append((f"{suite_name} suite", False, str(e)))
            suite_results[suite_name] = (0, 1)
    
    # Print overall summary
    elapsed = time.time() - start_time
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}SUITE SUMMARY{Colors.RESET}")
    print(f"{'='*60}")
    
    for suite_name, (passed, total) in suite_results.items():
        status = f"{Colors.GREEN}✓{Colors.RESET}" if passed == total else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {status} {suite_name}: {passed}/{total}")
    
    all_passed = print_summary(all_results)
    
    print(f"\nTime elapsed: {elapsed:.2f}s")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
