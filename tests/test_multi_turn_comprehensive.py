"""
Multi-Turn Comprehensive Document Integration Test

8-turn conversation test validating:
- Document upload and context injection
- Document switching between turns
- Conversation memory with document context
- Backwards compatibility (no document turns)
- URL extraction integration
- Keyword validation and timing metrics

This test is the FOUNDATION for:
- Pipeline validation
- API integration testing  
- UI feature development

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


# =============================================================================
# TEST CONFIGURATION - 8 TURNS
# =============================================================================

TURN_CONFIG = [
    {
        "turn": 1,
        "description": "Upload PDF #1 and ask about it",
        "upload_file": "peraturan_1.pdf",
        "clear_docs": True,  # Start fresh
        "include_session_docs": True,
        "query": "Apa yang diatur dalam peraturan BPK yang saya unggah ini? Jelaskan secara singkat.",
        "expected_keywords": ["BPK", "tata kerja", "peraturan", "badan pemeriksa"],
        "timeout": 180
    },
    {
        "turn": 2,
        "description": "Follow-up on same document (memory + doc)",
        "upload_file": None,  # Use existing
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Siapa saja yang disebutkan dalam peraturan tersebut? Sebutkan struktur organisasinya.",
        "expected_keywords": ["ketua", "anggota", "sekretariat", "BPK"],
        "timeout": 180
    },
    {
        "turn": 3,
        "description": "General question WITHOUT document (backwards compat)",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": False,  # NO DOCUMENT
        "query": "Jelaskan tentang UU Ketenagakerjaan No. 13 Tahun 2003 secara singkat.",
        "expected_keywords": ["ketenagakerjaan", "pekerja", "tenaga kerja"],
        "timeout": 180
    },
    {
        "turn": 4,
        "description": "Upload PDF #2 (contract) - switch document",
        "upload_file": "contract_sample_1.pdf",
        "clear_docs": True,  # Clear previous doc
        "include_session_docs": True,
        "query": "Apa isi kontrak yang saya unggah? Jelaskan para pihak dan pokok perjanjiannya.",
        "expected_keywords": ["kontrak", "perjanjian", "pihak"],
        "timeout": 180
    },
    {
        "turn": 5,
        "description": "Follow-up on contract (memory + switched doc)",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Apa kewajiban masing-masing pihak dalam kontrak tersebut?",
        "expected_keywords": ["kewajiban", "pihak"],
        "timeout": 180
    },
    {
        "turn": 6,
        "description": "Upload both docs (multi-document context)",
        "upload_file": "peraturan_1.pdf",  # Add back first doc
        "clear_docs": False,  # Keep contract
        "include_session_docs": True,
        "query": "Bandingkan kedua dokumen yang saya unggah. Apa perbedaan sifat hukumnya?",
        "expected_keywords": ["peraturan", "kontrak", "perbedaan", "hukum"],
        "timeout": 240  # Longer for multi-doc
    },
    {
        "turn": 7,
        "description": "Extract from URL (URL integration)",
        "upload_url": "https://www.cnbcindonesia.com/news/20251226155414-4-697445/kpk-setop-penyidikan-kasus-korupsi-izin-tambang-konawe-utara-rp27-t",
        "upload_file": None,
        "clear_docs": True,  # Clear PDFs
        "include_session_docs": True,
        "query": "Apa isi berita yang saya berikan? Jelaskan kasusnya.",
        "expected_keywords": ["KPK", "korupsi", "penyidikan", "tambang"],
        "timeout": 180
    },
    {
        "turn": 8,
        "description": "Summary WITHOUT document (memory only)",
        "upload_file": None,
        "clear_docs": True,
        "include_session_docs": False,  # NO DOCUMENT
        "query": "Berdasarkan percakapan kita sebelumnya, topik hukum apa saja yang sudah kita bahas?",
        "expected_keywords": ["peraturan", "kontrak", "korupsi"],  # From memory
        "timeout": 180
    }
]


# =============================================================================
# TEST RUNNER CLASS
# =============================================================================

class MultiTurnTestRunner:
    """Runs comprehensive 8-turn test with metrics"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"test-multi-{int(time.time())}"
        self.results: List[Dict] = []
        self.uploaded_docs: List[str] = []
        self.start_time = None
        self.total_time = 0
    
    def check_api(self) -> bool:
        """Check if API is running"""
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def upload_document(self, filename: str) -> Optional[Dict]:
        """Upload a document"""
        filepath = TEST_DOCS_DIR / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'rb') as f:
                resp = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files={'file': (filename, f)},
                    data={'session_id': self.session_id},
                    timeout=60
                )
            
            if resp.status_code == 200:
                data = resp.json()
                self.uploaded_docs.append(data.get('document_id'))
                return data
            return None
        except Exception as e:
            print(f"Upload error: {e}")
            return None
    
    def extract_url(self, url: str) -> Optional[Dict]:
        """Extract content from URL"""
        try:
            resp = requests.post(
                f"{API_BASE_URL}/documents/url",
                json={'url': url, 'session_id': self.session_id},
                timeout=60
            )
            
            if resp.status_code == 200:
                data = resp.json()
                self.uploaded_docs.append(data.get('document_id'))
                return data
            return None
        except Exception as e:
            print(f"URL extract error: {e}")
            return None
    
    def clear_documents(self):
        """Clear session documents"""
        try:
            requests.delete(
                f"{API_BASE_URL}/documents",
                params={'session_id': self.session_id},
                timeout=10
            )
            self.uploaded_docs = []
        except:
            pass
    
    def chat(self, query: str, include_docs: bool, timeout: int) -> Tuple[Optional[str], float]:
        """Send chat request and return answer + time"""
        start = time.time()
        
        try:
            payload = {
                'query': query,
                'session_id': self.session_id,
                'include_session_documents': include_docs,
                'thinking_level': 'low',
                'stream': False,
                'top_k': 5,
                'max_tokens': 1024
            }
            
            resp = requests.post(
                f"{API_BASE_URL}/rag/chat",
                json=payload,
                timeout=timeout
            )
            
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                return resp.json().get('answer', ''), elapsed
            return None, elapsed
            
        except requests.Timeout:
            return None, time.time() - start
        except Exception as e:
            return None, time.time() - start
    
    def validate_keywords(self, answer: str, expected: List[str]) -> Tuple[bool, List[str]]:
        """Check if answer contains expected keywords"""
        if not answer:
            return False, []
        
        answer_lower = answer.lower()
        found = [kw for kw in expected if kw.lower() in answer_lower]
        
        # Pass if at least 50% keywords found
        threshold = max(1, len(expected) // 2)
        return len(found) >= threshold, found
    
    def run_turn(self, config: Dict) -> Dict:
        """Run a single turn"""
        turn_num = config['turn']
        result = {
            'turn': turn_num,
            'description': config['description'],
            'passed': False,
            'answer': None,
            'time_seconds': 0,
            'keywords_found': [],
            'keywords_expected': config['expected_keywords'],
            'error': None
        }
        
        print(f"\n{Colors.BOLD}Turn {turn_num}: {config['description']}{Colors.RESET}")
        print(f"  Query: {config['query'][:60]}...")
        
        # Clear docs if needed
        if config.get('clear_docs'):
            self.clear_documents()
            print(f"  {Colors.YELLOW}Cleared documents{Colors.RESET}")
        
        # Upload file if specified
        if config.get('upload_file'):
            print(f"  {Colors.YELLOW}Uploading {config['upload_file']}...{Colors.RESET}")
            upload_result = self.upload_document(config['upload_file'])
            if upload_result:
                print(f"  {Colors.GREEN}✓ Uploaded: {upload_result.get('char_count')} chars{Colors.RESET}")
            else:
                result['error'] = "Upload failed"
                print(f"  {Colors.RED}✗ Upload failed{Colors.RESET}")
                return result
        
        # Extract URL if specified
        if config.get('upload_url'):
            print(f"  {Colors.YELLOW}Extracting URL...{Colors.RESET}")
            url_result = self.extract_url(config['upload_url'])
            if url_result:
                print(f"  {Colors.GREEN}✓ Extracted: {url_result.get('char_count')} chars{Colors.RESET}")
            else:
                result['error'] = "URL extraction failed"
                print(f"  {Colors.RED}✗ URL extraction failed{Colors.RESET}")
                return result
        
        # Send chat
        print(f"  {Colors.YELLOW}Generating response...{Colors.RESET}")
        answer, elapsed = self.chat(
            config['query'],
            config['include_session_docs'],
            config['timeout']
        )
        
        result['time_seconds'] = round(elapsed, 2)
        
        if answer:
            result['answer'] = answer
            
            # Validate keywords
            passed, found = self.validate_keywords(answer, config['expected_keywords'])
            result['passed'] = passed
            result['keywords_found'] = found
            
            status = f"{Colors.GREEN}✓ PASS" if passed else f"{Colors.RED}✗ FAIL"
            print(f"  {status}{Colors.RESET} ({elapsed:.1f}s)")
            print(f"  Keywords: {found} / {config['expected_keywords']}")
            print(f"  Answer: {answer[:150]}...")
        else:
            result['error'] = "Timeout or error"
            print(f"  {Colors.RED}✗ TIMEOUT ({elapsed:.1f}s){Colors.RESET}")
        
        return result
    
    def run_all(self) -> Dict:
        """Run all turns"""
        self.start_time = datetime.now()
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("=" * 70)
        print("  MULTI-TURN COMPREHENSIVE DOCUMENT INTEGRATION TEST")
        print("=" * 70)
        print(f"{Colors.RESET}")
        
        print(f"Session ID: {self.session_id}")
        print(f"API URL: {API_BASE_URL}")
        print(f"Test documents: {TEST_DOCS_DIR}")
        
        # Check API
        if not self.check_api():
            print(f"\n{Colors.RED}ERROR: API not running at {API_BASE_URL}{Colors.RESET}")
            return {'error': 'API not running', 'results': []}
        
        print(f"\n{Colors.GREEN}✓ API is running{Colors.RESET}")
        
        # Run all turns
        for config in TURN_CONFIG:
            result = self.run_turn(config)
            self.results.append(result)
            time.sleep(2)  # Brief pause between turns
        
        self.total_time = (datetime.now() - self.start_time).total_seconds()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate test report"""
        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed
        
        report = {
            'test_name': 'Multi-Turn Comprehensive Document Integration Test',
            'session_id': self.session_id,
            'timestamp': self.start_time.isoformat() if self.start_time else None,
            'total_time_seconds': round(self.total_time, 2),
            'summary': {
                'total_turns': len(self.results),
                'passed': passed,
                'failed': failed,
                'pass_rate': f"{passed/len(self.results)*100:.1f}%"
            },
            'turns': self.results,
            'timing': {
                'average_per_turn': round(sum(r['time_seconds'] for r in self.results) / len(self.results), 2),
                'slowest_turn': max(self.results, key=lambda r: r['time_seconds'])['turn'],
                'fastest_turn': min(self.results, key=lambda r: r['time_seconds'])['turn']
            }
        }
        
        # Print summary
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}TEST SUMMARY{Colors.RESET}")
        print(f"{'='*70}")
        
        for r in self.results:
            status = f"{Colors.GREEN}✓" if r['passed'] else f"{Colors.RED}✗"
            print(f"  {status} Turn {r['turn']}: {r['description'][:40]}... ({r['time_seconds']}s){Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
        print(f"  Total:  {len(self.results)}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"  Pass Rate: {report['summary']['pass_rate']}")
        print(f"\nTotal Time: {self.total_time:.1f}s")
        
        # Save report
        self.save_report(report)
        
        return report
    
    def save_report(self, report: Dict):
        """Save report to file"""
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = REPORT_DIR / f"multi_turn_test_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n{Colors.CYAN}Report saved: {report_file}{Colors.RESET}")
            
        except Exception as e:
            print(f"\n{Colors.YELLOW}Could not save report: {e}{Colors.RESET}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    runner = MultiTurnTestRunner()
    report = runner.run_all()
    
    if report.get('error'):
        print(f"\n{Colors.RED}Test failed: {report['error']}{Colors.RESET}")
        return 1
    
    # Exit with appropriate code
    if report['summary']['failed'] == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}Some tests failed - see report for details{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
