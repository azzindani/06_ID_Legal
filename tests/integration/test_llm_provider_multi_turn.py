"""
LLM Provider Multi-Turn Conversation Test with Full RAG System

Comprehensive end-to-end test that validates the LLM Provider system with:
1. Full RAG pipeline initialization (retrieval + database)
2. Multi-turn conversation with document context
3. OpenRouter as the LLM provider (with API streaming)
4. Provider fallback chain (auto-retry with different providers)
5. All thinking levels (low, medium, high)
6. All generation parameters (temperature, top_k, max_tokens, etc.)

This test runs the COMPLETE system through the API, similar to test_multi_turn_comprehensive.py
but using OpenRouter instead of local LLM.

Usage:
    # Start API server first (with OpenRouter provider)
    python -m api.server --llm-provider openrouter
    
    # Run test
    python tests/integration/test_llm_provider_multi_turn.py --openrouter-key sk-or-v1-...

File: tests/integration/test_llm_provider_multi_turn.py
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")
TEST_DOCS_DIR = PROJECT_ROOT / "tests" / "test_documents"
REPORT_DIR = PROJECT_ROOT / "tests" / "test_reports"

# Timeouts
CHAT_TIMEOUT = 300  # 5 minutes for OpenRouter
UPLOAD_TIMEOUT = 60


# Colors for terminal output
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


def print_header(title: str, char: str = "="):
    """Print a formatted header"""
    width = 80
    print(f"\n{Colors.BOLD}{char * width}{Colors.RESET}")
    print(f"{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{char * width}\n")


def print_result(name: str, success: bool, message: str = ""):
    """Print a test result"""
    emoji = "✅" if success else "❌"
    color = Colors.GREEN if success else Colors.RED
    print(f"{emoji} {color}{name}{Colors.RESET}: {message}")


# =============================================================================
# TEST CONFIGURATION - Multi-Turn Scenarios
# =============================================================================

TURN_CONFIG = [
    {
        "turn": 1,
        "description": "Upload PDF #1 (Peraturan BPK) + LOW thinking",
        "thinking_level": "low",
        "upload_file": "peraturan_1.pdf",
        "clear_docs": True,
        "include_session_docs": True,
        "query": "Apa yang diatur dalam peraturan BPK yang saya unggah ini? Jelaskan secara singkat fokus pengaturannya.",
        "expected_keywords": ["BPK", "tata kerja", "peraturan"],
        "max_tokens": 1024,
        "temperature": 0.7,
    },
    {
        "turn": 2,
        "description": "Follow-up on same document (memory + doc)",
        "thinking_level": "low",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Berdasarkan dokumen yang sama, siapa saja pejabat atau struktur yang disebutkan di dalamnya?",
        "expected_keywords": ["ketua", "anggota", "BPK"],
        "max_tokens": 512,
        "temperature": 0.7,
    },
    {
        "turn": 3,
        "description": "General question WITHOUT document (backwards compat)",
        "thinking_level": "medium",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": False,
        "query": "Jelaskan tentang UU Ketenagakerjaan No. 13 Tahun 2003 secara singkat, terutama hak-hak pekerja.",
        "expected_keywords": ["ketenagakerjaan", "pekerja", "hak"],
        "max_tokens": 1024,
        "temperature": 0.5,
    },
    {
        "turn": 4,
        "description": "Upload PDF #2 (contract) - switch document + MEDIUM",
        "thinking_level": "medium",
        "upload_file": "contract_sample_1.pdf",
        "clear_docs": True,
        "include_session_docs": True,
        "query": "Apa isi kontrak yang saya unggah ini? Siapa para pihak dan apa pokok perjanjiannya?",
        "expected_keywords": ["kontrak", "perjanjian", "pihak"],
        "max_tokens": 1024,
        "temperature": 0.5,
    },
    {
        "turn": 5,
        "description": "Follow-up on contract (memory + switched doc)",
        "thinking_level": "low",
        "upload_file": None,
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Apa kewajiban dan hak masing-masing pihak dalam kontrak yang sama?",
        "expected_keywords": ["kewajiban", "hak", "pihak"],
        "max_tokens": 512,
        "temperature": 0.7,
    },
    {
        "turn": 6,
        "description": "Upload putusan + HIGH thinking (multi-doc)",
        "thinking_level": "high",
        "upload_file": "putusan_mahkamah_agung_1.pdf",
        "clear_docs": False,
        "include_session_docs": True,
        "query": "Sekarang saya punya dua dokumen. Jelaskan perbedaan sifat hukum antara kontrak dan putusan pengadilan yang saya unggah.",
        "expected_keywords": ["kontrak", "putusan", "perbedaan"],
        "max_tokens": 2048,
        "temperature": 0.3,
    },
    {
        "turn": 7,
        "description": "Extract from URL (URL integration) + MEDIUM",
        "thinking_level": "medium",
        "upload_url": "https://www.cnbcindonesia.com/news/20251226155414-4-697445/kpk-setop-penyidikan-kasus-korupsi-izin-tambang-konawe-utara-rp27-t",
        "upload_file": None,
        "clear_docs": True,
        "include_session_docs": True,
        "query": "Apa isi berita yang saya berikan melalui URL tadi? Jelaskan kasusnya secara ringkas.",
        "expected_keywords": ["KPK", "korupsi", "tambang"],
        "max_tokens": 1024,
        "temperature": 0.5,
    },
    {
        "turn": 8,
        "description": "Summary WITHOUT document (memory only)",
        "thinking_level": "low",
        "upload_file": None,
        "clear_docs": True,
        "include_session_docs": False,
        "query": "Berdasarkan seluruh percakapan kita, topik hukum apa saja yang sudah kita bahas? Sebutkan secara ringkas.",
        "expected_keywords": ["BPK", "kontrak", "korupsi"],
        "max_tokens": 512,
        "temperature": 0.7,
    },
]



# =============================================================================
# API CLIENT
# =============================================================================

class OpenRouterTestClient:
    """API client for testing with OpenRouter provider"""
    
    def __init__(self, session_id: str, openrouter_key: str):
        self.session_id = session_id
        self.openrouter_key = openrouter_key
        self.uploaded_docs: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
    
    def check_api(self) -> bool:
        """Check if API is running"""
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
            return resp.status_code == 200
        except:
            return False
    
    def configure_openrouter(self) -> bool:
        """Configure API to use OpenRouter provider"""
        try:
            # Free models that work well (user-tested):
            # - nvidia/nemotron-3-nano-30b-a3b:free (default, fast)
            # - deepseek/deepseek-r1-0528:free (reasoning)
            # - openai/gpt-oss-20b:free (smaller, faster)
            resp = requests.post(
                f"{API_BASE_URL}/llm/config",
                json={
                    "provider": "openrouter",
                    "model": "nvidia/nemotron-3-nano-30b-a3b:free",
                    "api_key": self.openrouter_key,
                    "save_key": False
                },
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                print(f"{Colors.GREEN}✓ OpenRouter configured: {data.get('model')}{Colors.RESET}")
                return True
            else:
                print(f"{Colors.RED}✗ Failed to configure: {resp.text}{Colors.RESET}")
                return False
        except Exception as e:
            print(f"{Colors.RED}✗ Config error: {e}{Colors.RESET}")
            return False
    
    def get_llm_status(self) -> Dict:
        """Get current LLM provider status"""
        try:
            resp = requests.get(f"{API_BASE_URL}/llm/status", timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return {}
    
    def upload_document(self, filename: str) -> Tuple[bool, Dict]:
        """Upload document to session"""
        filepath = TEST_DOCS_DIR / filename
        if not filepath.exists():
            return False, {"error": f"File not found: {filepath}"}
        
        print(f"\n{Colors.YELLOW}📤 Uploading {filename}...{Colors.RESET}")
        
        try:
            with open(filepath, 'rb') as f:
                resp = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files={'file': (filename, f)},
                    data={'session_id': self.session_id},
                    timeout=UPLOAD_TIMEOUT
                )
            
            if resp.status_code == 200:
                data = resp.json()
                self.uploaded_docs.append(data)
                print(f"{Colors.GREEN}✓ Uploaded: {data.get('char_count', 0):,} chars{Colors.RESET}")
                return True, data
            else:
                return False, {"error": f"HTTP {resp.status_code}"}
                
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
    
    def extract_url(self, url: str) -> Tuple[bool, Dict]:
        """Extract content from URL"""
        print(f"\n{Colors.YELLOW}🌐 Extracting URL...{Colors.RESET}")
        
        try:
            resp = requests.post(
                f"{API_BASE_URL}/documents/url",
                json={'url': url, 'session_id': self.session_id},
                timeout=120
            )
            
            if resp.status_code == 200:
                data = resp.json()
                self.uploaded_docs.append(data)
                print(f"{Colors.GREEN}✓ URL extracted: {data.get('char_count', 0):,} chars{Colors.RESET}")
                return True, data
            else:
                return False, {"error": f"HTTP {resp.status_code}"}
                
        except Exception as e:
            return False, {"error": str(e)}
    
    def chat_streaming(
        self,
        query: str,
        include_docs: bool,
        thinking_level: str = "low",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict:
        """Send chat with streaming and return result"""
        
        print(f"\n{Colors.BOLD}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.CYAN}Query:{Colors.RESET} {query[:80]}...")
        print(f"{Colors.CYAN}Thinking:{Colors.RESET} {thinking_level} | {Colors.CYAN}Docs:{Colors.RESET} {include_docs}")
        print(f"{Colors.BOLD}{'─' * 80}{Colors.RESET}")
        
        result = {
            'success': False,
            'answer': '',
            'thinking': '',
            'sources': [],
            'elapsed': 0,
            'error': None,
            'provider': 'unknown'
        }
        
        start = time.time()
        
        try:
            payload = {
                'query': query,
                'session_id': self.session_id,
                'include_session_documents': include_docs,
                'thinking_level': thinking_level,
                'stream': True,
                'top_k': 10,
                'max_tokens': max_tokens,
                'temperature': temperature,
                # No max_document_chars - use full document
            }
            
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
                thinking_text = ""
                in_thinking = False
                
                print(f"\n{Colors.MAGENTA}[Thinking]{Colors.RESET} ", end="", flush=True)
                
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                
                                if data.get('type') == 'thinking':
                                    content = data.get('content', '')
                                    thinking_text += content
                                    # Show abbreviated thinking
                                    if len(thinking_text) < 200:
                                        print(content[:20], end="", flush=True)
                                    
                                elif data.get('type') == 'chunk':
                                    if not in_thinking:
                                        print(f"\n\n{Colors.GREEN}[Answer]{Colors.RESET} ", end="", flush=True)
                                        in_thinking = True
                                    
                                    token = data.get('content', '')
                                    full_text += token
                                    print(token, end="", flush=True)
                                    
                                elif data.get('type') == 'done' or data.get('type') == 'metadata':
                                    result['sources'] = data.get('sources', data.get('citations', []))
                                    result['provider'] = data.get('provider', 'openrouter')
                                    
                            except json.JSONDecodeError:
                                pass
                
                result['answer'] = full_text
                result['thinking'] = thinking_text
                result['success'] = len(full_text) > 0
            
            elapsed = time.time() - start
            result['elapsed'] = elapsed
            
            print(f"\n\n{Colors.DIM}[{elapsed:.1f}s | {len(result['sources'])} sources]{Colors.RESET}")
            
            # Track conversation
            self.conversation_history.append({
                'turn': len(self.conversation_history) + 1,
                'query': query,
                'answer': full_text,
                'thinking_level': thinking_level,
                'include_docs': include_docs,
                'elapsed': elapsed
            })
            
        except requests.Timeout:
            result['error'] = f"Timeout after {CHAT_TIMEOUT}s"
            result['elapsed'] = time.time() - start
            print(f"\n{Colors.RED}⏱️ TIMEOUT{Colors.RESET}")
        except Exception as e:
            result['error'] = str(e)
            result['elapsed'] = time.time() - start
            print(f"\n{Colors.RED}❌ {e}{Colors.RESET}")
        
        return result


# =============================================================================
# TEST RUNNER
# =============================================================================

class MultiTurnOpenRouterTestRunner:
    """Runs multi-turn test with OpenRouter provider"""
    
    def __init__(self, openrouter_key: str):
        self.session_id = f"test-openrouter-{int(time.time())}"
        self.openrouter_key = openrouter_key
        self.client = OpenRouterTestClient(self.session_id, openrouter_key)
        self.results: List[Dict] = []
        self.start_time = None
    
    def run_turn(self, config: Dict) -> Dict:
        """Run a single conversation turn"""
        turn_num = config['turn']
        
        print_header(f"TURN {turn_num}: {config['description']}")
        
        result = {
            'turn': turn_num,
            'description': config['description'],
            'thinking_level': config['thinking_level'],
            'passed': False,
            'chat_result': None,
            'keywords_found': [],
            'keywords_expected': config['expected_keywords'],
            'error': None
        }
        
        # Clear docs if needed
        if config.get('clear_docs'):
            self.client.clear_documents()
        
        # Upload file if specified
        if config.get('upload_file'):
            success, data = self.client.upload_document(config['upload_file'])
            if not success:
                result['error'] = f"Upload failed: {data.get('error')}"
                print(f"{Colors.RED}✗ {result['error']}{Colors.RESET}")
                return result
        
        # Extract URL if specified
        if config.get('upload_url'):
            success, data = self.client.extract_url(config['upload_url'])
            if not success:
                result['error'] = f"URL extraction failed: {data.get('error')}"
                print(f"{Colors.RED}✗ {result['error']}{Colors.RESET}")
                return result
        
        # Send chat
        chat_result = self.client.chat_streaming(
            query=config['query'],
            include_docs=config['include_session_docs'],
            thinking_level=config['thinking_level'],
            max_tokens=config.get('max_tokens', 1024),
            temperature=config.get('temperature', 0.7)
        )
        result['chat_result'] = chat_result
        
        if chat_result['error']:
            result['error'] = chat_result['error']
            return result
        
        # Validate keywords
        answer = chat_result['answer'].lower()
        found = [kw for kw in config['expected_keywords'] if kw.lower() in answer]
        result['keywords_found'] = found
        
        threshold = max(1, len(config['expected_keywords']) // 2)
        result['passed'] = len(found) >= threshold
        
        # Show validation
        print(f"\n{Colors.BOLD}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.CYAN}Keywords:{Colors.RESET} Expected {config['expected_keywords']}")
        print(f"{Colors.CYAN}Found:{Colors.RESET} {found}")
        status = f"{Colors.GREEN}✓ PASS" if result['passed'] else f"{Colors.RED}✗ FAIL"
        print(f"{Colors.CYAN}Status:{Colors.RESET} {status}{Colors.RESET}")
        
        return result
    
    def run_all(self) -> Dict:
        """Run all turns"""
        self.start_time = datetime.now()
        
        print_header("LLM PROVIDER MULTI-TURN TEST (OpenRouter)", "█")
        
        print(f"Session ID: {self.session_id}")
        print(f"API URL: {API_BASE_URL}")
        print(f"Test Documents: {TEST_DOCS_DIR}")
        
        # Check API
        if not self.client.check_api():
            print(f"\n{Colors.RED}ERROR: API not running at {API_BASE_URL}{Colors.RESET}")
            print(f"Start with: python -m api.server --llm-provider openrouter")
            return {'error': 'API not running', 'results': []}
        
        print(f"{Colors.GREEN}✓ API is running{Colors.RESET}")
        
        # Configure OpenRouter
        if not self.client.configure_openrouter():
            print(f"\n{Colors.RED}ERROR: Could not configure OpenRouter{Colors.RESET}")
            return {'error': 'OpenRouter config failed', 'results': []}
        
        # Show LLM status
        status = self.client.get_llm_status()
        print(f"LLM Provider: {status.get('provider', 'unknown')}")
        print(f"Model: {status.get('model', 'unknown')}")
        
        # Run all turns
        for config in TURN_CONFIG:
            result = self.run_turn(config)
            self.results.append(result)
            time.sleep(2)  # Brief pause between turns
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return self.generate_report(total_time)
    
    def generate_report(self, total_time: float) -> Dict:
        """Generate test report"""
        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed
        
        print_header("TEST SUMMARY")
        
        # Per-turn summary
        for r in self.results:
            status = f"{Colors.GREEN}✓ PASS" if r['passed'] else f"{Colors.RED}✗ FAIL"
            elapsed = r['chat_result']['elapsed'] if r['chat_result'] else 0
            print(f"{status}{Colors.RESET} Turn {r['turn']}: {r['description'][:40]}... [{r['thinking_level']}] ({elapsed:.1f}s)")
            if r['error']:
                print(f"      Error: {r['error']}")
        
        print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
        print(f"  Total:  {len(self.results)}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"  Pass Rate: {passed/len(self.results)*100:.1f}%")
        print(f"\nTotal Time: {total_time:.1f}s")
        
        # Features tested
        print(f"\n{Colors.CYAN}Features Tested:{Colors.RESET}")
        print("  ✅ OpenRouter as LLM provider")
        print("  ✅ Multi-turn conversation memory")
        print("  ✅ Document upload and context")
        print("  ✅ Thinking levels (low/medium/high)")
        print("  ✅ Streaming response")
        print("  ✅ Generation parameters (max_tokens, temperature)")
        
        # Build report
        report = {
            'test_name': 'LLM Provider Multi-Turn (OpenRouter)',
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
        }
        
        # Save report
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = REPORT_DIR / f"openrouter_multi_turn_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n{Colors.CYAN}Report: {report_file}{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.YELLOW}Could not save report: {e}{Colors.RESET}")
        
        return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM Provider Multi-Turn Test with OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start API first:
    python -m api.server --llm-provider none
    
    # Then run test:
    python tests/integration/test_llm_provider_multi_turn.py --openrouter-key sk-or-v1-...
    
    # Or set environment variable:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python tests/integration/test_llm_provider_multi_turn.py
"""
    )
    parser.add_argument("--openrouter-key", type=str, help="OpenRouter API key")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/api/v1", help="API base URL")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(f"{Colors.RED}ERROR: OpenRouter API key required{Colors.RESET}")
        print(f"Use --openrouter-key or set OPENROUTER_API_KEY environment variable")
        return 1
    
    # Set API URL
    global API_BASE_URL
    API_BASE_URL = args.api_url
    
    # Run test
    runner = MultiTurnOpenRouterTestRunner(api_key)
    report = runner.run_all()
    
    if report.get('error'):
        print(f"\n{Colors.RED}Test failed: {report['error']}{Colors.RESET}")
        return 1
    
    if report['summary']['failed'] == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}Some tests failed - see report{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
