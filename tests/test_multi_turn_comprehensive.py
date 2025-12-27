"""
Multi-Turn Comprehensive Document Integration Test (API-Based)

8-turn conversation test validating:
- Document upload and context injection
- Document switching between turns
- Conversation memory with document context
- Backwards compatibility (no document turns)
- URL extraction integration

AUDIT-STYLE OUTPUT:
- Full streaming thinking + answer display
- Complete legal references and citations
- Chat history transparency
- Timing and performance metrics
- JSON report generation

Run: python tests/test_multi_turn_comprehensive.py

File: tests/test_multi_turn_comprehensive.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")
TEST_DOCS_DIR = PROJECT_ROOT / "tests" / "test_documents"
REPORT_DIR = PROJECT_ROOT / "tests" / "test_reports"

# GENEROUS TIMEOUTS for Kaggle
CHAT_TIMEOUT = 600  # 10 minutes for LLM with large context
UPLOAD_TIMEOUT = 180  # 3 minutes for large file upload
URL_TIMEOUT = 120  # 2 minutes for URL extraction

# Colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_box(title: str, content: List[str], width: int = 100):
    """Print a bordered box"""
    print(f"\n┌{'─' * (width-2)}┐")
    print(f"│ {title:<{width-4}} │")
    print(f"├{'─' * (width-2)}┤")
    for line in content:
        # Handle long lines
        while len(line) > width - 4:
            print(f"│ {line[:width-4]} │")
            line = line[width-4:]
        print(f"│ {line:<{width-4}} │")
    print(f"└{'─' * (width-2)}┘")


# =============================================================================
# TEST CONFIGURATION - 8 TURNS
# =============================================================================

TURN_CONFIG = [
    {
        "turn": 1,
        "description": "Upload PDF #1 (Peraturan BPK) and ask about it",
        "upload_file": "peraturan_1.pdf",
        "clear_docs": True,
        "include_session_docs": True,
        "query": "Apa yang diatur dalam peraturan BPK yang saya unggah ini? Jelaskan secara singkat fokus pengaturannya.",
        "expected_keywords": ["BPK", "tata kerja", "peraturan"],
    },
    {
        "turn": 2,
        "description": "Follow-up on same document (memory + doc)",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Berdasarkan dokumen yang sama, siapa saja pejabat atau struktur yang disebutkan di dalamnya?",
        "expected_keywords": ["ketua", "anggota", "BPK"],
    },
    {
        "turn": 3,
        "description": "General question WITHOUT document (backwards compat)",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": False,
        "query": "Jelaskan tentang UU Ketenagakerjaan No. 13 Tahun 2003 secara singkat, terutama hak-hak pekerja.",
        "expected_keywords": ["ketenagakerjaan", "pekerja", "hak"],
    },
    {
        "turn": 4,
        "description": "Upload PDF #2 (contract) - switch document",
        "upload_file": "contract_sample_1.pdf",
        "clear_docs": True,
        "include_session_docs": True,
        "query": "Apa isi kontrak yang saya unggah ini? Siapa para pihak dan apa pokok perjanjiannya?",
        "expected_keywords": ["kontrak", "perjanjian", "pihak"],
    },
    {
        "turn": 5,
        "description": "Follow-up on contract (memory + switched doc)",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Apa kewajiban dan hak masing-masing pihak dalam kontrak yang sama?",
        "expected_keywords": ["kewajiban", "hak", "pihak"],
    },
    {
        "turn": 6,
        "description": "Upload putusan + kontrak (multi-document context)",
        "upload_file": "putusan_mahkamah_agung_1.pdf",
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Sekarang saya punya dua dokumen. Jelaskan perbedaan sifat hukum antara kontrak dan putusan pengadilan yang saya unggah.",
        "expected_keywords": ["kontrak", "putusan", "perbedaan"],
    },
    {
        "turn": 7,
        "description": "Extract from URL (URL integration)",
        "upload_url": "https://www.cnbcindonesia.com/news/20251226155414-4-697445/kpk-setop-penyidikan-kasus-korupsi-izin-tambang-konawe-utara-rp27-t",
        "upload_file": None,
        "clear_docs": True,
        "include_session_docs": True,
        "query": "Apa isi berita yang saya berikan melalui URL tadi? Jelaskan kasusnya secara ringkas.",
        "expected_keywords": ["KPK", "korupsi", "tambang"],
    },
    {
        "turn": 8,
        "description": "Summary WITHOUT document (memory only)",
        "upload_file": None,
        "clear_docs": True,
        "include_session_docs": False,
        "query": "Berdasarkan seluruh percakapan kita, topik hukum apa saja yang sudah kita bahas? Sebutkan secara ringkas.",
        "expected_keywords": ["BPK", "kontrak", "korupsi"],
    }
]


# =============================================================================
# API CLIENT WITH STREAMING
# =============================================================================

class AuditAPIClient:
    """API client with streaming and full audit output"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.uploaded_docs: List[Dict] = []
        self.conversation_history: List[Dict] = []
    
    def check_api(self) -> bool:
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
            return resp.status_code == 200
        except:
            return False
    
    def cleanup_memory(self) -> Dict:
        """Force GPU/RAM cleanup between turns"""
        print(f"\n{Colors.DIM}🧹 Cleaning GPU/RAM memory...{Colors.RESET}")
        try:
            resp = requests.post(f"{API_BASE_URL}/memory/cleanup", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                before_gpu = data.get('before', {}).get('gpu', {}).get('allocated_mb', 0)
                after_gpu = data.get('after', {}).get('gpu', {}).get('allocated_mb', 0)
                freed = data.get('cleanup', {}).get('gpu_freed_mb', 0)
                
                print(f"{Colors.GREEN}✓ Memory cleaned: GPU {before_gpu:.0f}MB → {after_gpu:.0f}MB (freed {freed:.0f}MB){Colors.RESET}")
                return data
            else:
                print(f"{Colors.YELLOW}⚠ Cleanup endpoint returned {resp.status_code}{Colors.RESET}")
                return {}
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Memory cleanup error: {e}{Colors.RESET}")
            return {}
    
    def upload_document(self, filename: str) -> Tuple[bool, Dict]:
        """Upload document and return full details"""
        filepath = TEST_DOCS_DIR / filename
        if not filepath.exists():
            return False, {"error": f"File not found: {filepath}"}
        
        print(f"\n{Colors.YELLOW}📤 Uploading {filename}...{Colors.RESET}")
        start = time.time()
        
        try:
            with open(filepath, 'rb') as f:
                resp = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files={'file': (filename, f)},
                    data={'session_id': self.session_id},
                    timeout=UPLOAD_TIMEOUT
                )
            
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                self.uploaded_docs.append(data)
                print(f"{Colors.GREEN}✓ Upload complete ({elapsed:.1f}s){Colors.RESET}")
                print(f"  Document ID: {data.get('document_id', 'N/A')}")
                print(f"  Characters: {data.get('char_count', 0):,}")
                print(f"  Pages: {data.get('page_count', 0)}")
                return True, data
            else:
                return False, {"error": f"HTTP {resp.status_code}", "detail": resp.text}
                
        except requests.Timeout:
            return False, {"error": f"Upload timeout ({UPLOAD_TIMEOUT}s)"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def extract_url(self, url: str) -> Tuple[bool, Dict]:
        """Extract content from URL"""
        print(f"\n{Colors.YELLOW}🌐 Extracting URL...{Colors.RESET}")
        start = time.time()
        
        try:
            resp = requests.post(
                f"{API_BASE_URL}/documents/url",
                json={'url': url, 'session_id': self.session_id},
                timeout=URL_TIMEOUT
            )
            
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                self.uploaded_docs.append(data)
                print(f"{Colors.GREEN}✓ URL extracted ({elapsed:.1f}s){Colors.RESET}")
                print(f"  Characters: {data.get('char_count', 0):,}")
                return True, data
            else:
                return False, {"error": f"HTTP {resp.status_code}", "detail": resp.text}
                
        except requests.Timeout:
            return False, {"error": f"URL timeout ({URL_TIMEOUT}s)"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def clear_documents(self):
        """Clear session documents"""
        try:
            requests.delete(
                f"{API_BASE_URL}/documents",
                params={'session_id': self.session_id},
                timeout=30
            )
            self.uploaded_docs = []
            print(f"{Colors.DIM}🗑️  Documents cleared{Colors.RESET}")
        except:
            pass
    
    def list_documents(self) -> List[Dict]:
        """List current session documents"""
        try:
            resp = requests.get(
                f"{API_BASE_URL}/documents",
                params={'session_id': self.session_id},
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json().get('documents', [])
        except:
            pass
        return []
    
    def chat_streaming(self, query: str, include_docs: bool) -> Dict:
        """Send chat with streaming output - FULL AUDIT"""
        print(f"\n{Colors.BOLD}{'═' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}QUERY{Colors.RESET}")
        print(f"{'═' * 100}")
        print(f"{query}")
        
        # Show document context status
        print(f"\n{Colors.CYAN}Document Context: {'ENABLED' if include_docs else 'DISABLED'}{Colors.RESET}")
        if include_docs:
            docs = self.list_documents()
            if docs:
                print(f"  Active Documents: {len(docs)}")
                for d in docs:
                    print(f"    - {d.get('filename', 'N/A')} ({d.get('char_count', 0):,} chars)")
        
        print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}STREAMING RESPONSE{Colors.RESET}")
        print(f"{'─' * 100}")
        
        start = time.time()
        result = {
            'success': False,
            'answer': '',
            'thinking': '',
            'sources': [],
            'elapsed': 0,
            'error': None
        }
        
        try:
            payload = {
                'query': query,
                'session_id': self.session_id,
                'include_session_documents': include_docs,
                'thinking_level': 'low',
                'stream': True,
                'top_k': 5,
                'max_tokens': 1024
            }
            
            # Streaming request
            with requests.post(
                f"{API_BASE_URL}/rag/chat",
                json=payload,
                stream=True,
                timeout=CHAT_TIMEOUT
            ) as resp:
                if resp.status_code != 200:
                    result['error'] = f"HTTP {resp.status_code}"
                    return result
                
                full_text = ""
                chunk_count = 0
                
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                
                                if data.get('type') == 'chunk':
                                    token = data.get('content', '')
                                    full_text += token
                                    print(token, end='', flush=True)
                                    chunk_count += 1
                                    
                                elif data.get('type') == 'metadata':
                                    result['sources'] = data.get('sources', [])
                                    result['thinking'] = data.get('thinking', '')
                                    
                            except json.JSONDecodeError:
                                pass
                
                result['answer'] = full_text
                result['success'] = True
                
            elapsed = time.time() - start
            result['elapsed'] = elapsed
            
            print(f"\n\n{Colors.DIM}[Streamed in {elapsed:.1f}s]{Colors.RESET}")
            
            # Show sources/citations with FULL LEGAL REFERENCES
            if result['sources']:
                print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
                print(f"{Colors.BOLD}📚 LEGAL REFERENCES ({len(result['sources'])} sources retrieved){Colors.RESET}")
                print(f"{'─' * 100}")
                
                for idx, src in enumerate(result['sources'][:8], 1):
                    reg_type = src.get('regulation_type', src.get('type', 'N/A'))
                    reg_num = src.get('regulation_number', src.get('number', 'N/A'))
                    year = src.get('year', 'N/A')
                    title = src.get('title', src.get('regulation_title', ''))
                    article = src.get('article', src.get('article_number', ''))
                    score = src.get('score', src.get('relevance_score', 0))
                    snippet = src.get('content', src.get('text', src.get('snippet', '')))[:150]
                    
                    # Main reference line
                    ref_str = f"{reg_type} No. {reg_num}/{year}"
                    if article:
                        ref_str += f" Pasal {article}"
                    
                    print(f"\n  {Colors.CYAN}{idx}. {ref_str}{Colors.RESET}")
                    print(f"     Score: {score:.3f}")
                    
                    if title:
                        print(f"     Title: {title[:80]}{'...' if len(title) > 80 else ''}")
                    
                    if snippet:
                        # Clean up snippet
                        snippet = snippet.replace('\n', ' ').strip()
                        print(f"     Preview: \"{snippet}{'...' if len(snippet) >= 150 else ''}\"")
                
                print(f"\n{'─' * 100}")
            
            # Save to conversation history
            self.conversation_history.append({
                'turn': len(self.conversation_history) + 1,
                'query': query,
                'answer': full_text,
                'include_docs': include_docs,
                'sources': result['sources'],
                'elapsed': elapsed
            })
            
        except requests.Timeout:
            result['error'] = f"Chat timeout ({CHAT_TIMEOUT}s)"
            result['elapsed'] = time.time() - start
            print(f"\n\n{Colors.RED}⏱️ TIMEOUT after {result['elapsed']:.1f}s{Colors.RESET}")
        except Exception as e:
            result['error'] = str(e)
            result['elapsed'] = time.time() - start
            print(f"\n\n{Colors.RED}❌ ERROR: {e}{Colors.RESET}")
        
        return result


# =============================================================================
# TEST RUNNER
# =============================================================================

class MultiTurnAuditTestRunner:
    """Runs 8-turn test with FULL audit output"""
    
    def __init__(self):
        self.session_id = f"test-audit-{int(time.time())}"
        self.client = AuditAPIClient(self.session_id)
        self.results: List[Dict] = []
        self.start_time = None
    
    def run_turn(self, config: Dict) -> Dict:
        """Run a single turn with full audit output"""
        turn_num = config['turn']
        
        print(f"\n\n{'█' * 100}")
        print(f"  TURN {turn_num}: {config['description']}")
        print(f"{'█' * 100}")
        
        # MEMORY CLEANUP at start of each turn to prevent OOM
        self.client.cleanup_memory()
        
        result = {
            'turn': turn_num,
            'description': config['description'],
            'passed': False,
            'upload_success': True,
            'chat_result': None,
            'keywords_found': [],
            'keywords_expected': config['expected_keywords'],
            'error': None,
            'memory_cleaned': True
        }
        
        # Step 1: Clear docs if needed
        if config.get('clear_docs'):
            self.client.clear_documents()
        
        # Step 2: Upload file if specified
        if config.get('upload_file'):
            success, data = self.client.upload_document(config['upload_file'])
            result['upload_success'] = success
            if not success:
                result['error'] = data.get('error')
                print(f"{Colors.RED}✗ Upload failed: {result['error']}{Colors.RESET}")
                return result
        
        # Step 3: Extract URL if specified
        if config.get('upload_url'):
            success, data = self.client.extract_url(config['upload_url'])
            result['upload_success'] = success
            if not success:
                result['error'] = data.get('error')
                print(f"{Colors.RED}✗ URL extraction failed: {result['error']}{Colors.RESET}")
                return result
        
        # Step 4: Send chat
        chat_result = self.client.chat_streaming(
            config['query'],
            config['include_session_docs']
        )
        result['chat_result'] = chat_result
        
        if chat_result['error']:
            result['error'] = chat_result['error']
            return result
        
        # Step 5: Validate keywords
        answer = chat_result['answer'].lower()
        found = [kw for kw in config['expected_keywords'] if kw.lower() in answer]
        result['keywords_found'] = found
        
        # Pass if at least half the keywords found
        threshold = max(1, len(config['expected_keywords']) // 2)
        result['passed'] = len(found) >= threshold
        
        # Show validation result
        print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}KEYWORD VALIDATION{Colors.RESET}")
        print(f"{'─' * 100}")
        print(f"  Expected: {config['expected_keywords']}")
        print(f"  Found: {found}")
        status = f"{Colors.GREEN}✓ PASS" if result['passed'] else f"{Colors.RED}✗ FAIL"
        print(f"  Status: {status}{Colors.RESET}")
        
        return result
    
    def run_all(self) -> Dict:
        """Run all 8 turns"""
        self.start_time = datetime.now()
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("═" * 100)
        print("  MULTI-TURN DOCUMENT INTEGRATION TEST (AUDIT MODE)")
        print("═" * 100)
        print(f"{Colors.RESET}")
        
        print(f"Session ID: {self.session_id}")
        print(f"API URL: {API_BASE_URL}")
        print(f"Timeouts: Chat={CHAT_TIMEOUT}s, Upload={UPLOAD_TIMEOUT}s, URL={URL_TIMEOUT}s")
        print(f"Test Documents: {TEST_DOCS_DIR}")
        
        # Check API
        if not self.client.check_api():
            print(f"\n{Colors.RED}ERROR: API not running at {API_BASE_URL}{Colors.RESET}")
            return {'error': 'API not running', 'results': []}
        
        print(f"\n{Colors.GREEN}✓ API is running{Colors.RESET}")
        
        # Run all turns
        for config in TURN_CONFIG:
            result = self.run_turn(config)
            self.results.append(result)
            time.sleep(5)  # Brief pause between turns
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return self.generate_report(total_time)
    
    def generate_report(self, total_time: float) -> Dict:
        """Generate comprehensive report"""
        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed
        
        # Print summary
        print(f"\n\n{'═' * 100}")
        print(f"  TEST SUMMARY")
        print(f"{'═' * 100}")
        
        for r in self.results:
            status = f"{Colors.GREEN}✓ PASS" if r['passed'] else f"{Colors.RED}✗ FAIL"
            elapsed = r['chat_result']['elapsed'] if r['chat_result'] else 0
            print(f"{status}{Colors.RESET} Turn {r['turn']}: {r['description'][:50]}... ({elapsed:.1f}s)")
            if r['error']:
                print(f"      Error: {r['error']}")
        
        print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
        print(f"  Total:  {len(self.results)}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"  Pass Rate: {passed/len(self.results)*100:.1f}%")
        print(f"\nTotal Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        # Print conversation history
        print(f"\n{'═' * 100}")
        print(f"  CONVERSATION HISTORY AUDIT")
        print(f"{'═' * 100}")
        
        for turn in self.client.conversation_history:
            print(f"\n{Colors.CYAN}Turn {turn['turn']}:{Colors.RESET}")
            print(f"  Q: {turn['query'][:80]}...")
            print(f"  A: {turn['answer'][:200]}...")
            print(f"  Sources: {len(turn['sources'])} | Time: {turn['elapsed']:.1f}s")
        
        # Build report
        report = {
            'test_name': 'Multi-Turn Document Integration Test (Audit)',
            'session_id': self.session_id,
            'timestamp': self.start_time.isoformat(),
            'total_time_seconds': round(total_time, 2),
            'summary': {
                'total_turns': len(self.results),
                'passed': passed,
                'failed': failed,
                'pass_rate': f"{passed/len(self.results)*100:.1f}%"
            },
            'turns': self.results,
            'conversation_history': self.client.conversation_history
        }
        
        # Save report
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = REPORT_DIR / f"multi_turn_audit_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n{Colors.CYAN}Report saved: {report_file}{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.YELLOW}Could not save report: {e}{Colors.RESET}")
        
        return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    runner = MultiTurnAuditTestRunner()
    report = runner.run_all()
    
    if report.get('error'):
        print(f"\n{Colors.RED}Test failed: {report['error']}{Colors.RESET}")
        return 1
    
    if report['summary']['failed'] == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}Some tests failed - see report for details{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
