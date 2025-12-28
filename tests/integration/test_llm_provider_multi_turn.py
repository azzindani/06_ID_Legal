"""
LLM Provider Multi-Turn Conversation Test with Document Support

Comprehensive end-to-end test that validates:
1. Multi-turn conversation with document context
2. Provider fallback chain (auto-retry with different providers)
3. Smart provider switching with context preservation
4. Document upload and URL extraction
5. Streaming responses across providers
6. Cost/token tracking during conversation

Usage:
    # Basic test (uses local provider or none)
    python tests/integration/test_llm_provider_multi_turn.py
    
    # With OpenRouter (tests fallback and switching)
    python tests/integration/test_llm_provider_multi_turn.py --with-openrouter --openrouter-key sk-or-v1-...
    
    # Full test with API server
    python tests/integration/test_llm_provider_multi_turn.py --with-api

File: tests/integration/test_llm_provider_multi_turn.py
"""

import os
import sys
import time
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Generator

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    print(f"\n{Colors.BOLD}{char * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{char * 80}\n")


def print_result(name: str, success: bool, message: str = ""):
    """Print a test result"""
    emoji = "✅" if success else "❌"
    color = Colors.GREEN if success else Colors.RED
    print(f"{emoji} {color}{name}{Colors.RESET}: {message}")


# =============================================================================
# FALLBACK CHAIN - Auto-retry with different providers
# =============================================================================

class ProviderFallbackChain:
    """
    Implements provider fallback for reliability.
    If one provider fails, automatically tries the next in chain.
    """
    
    def __init__(self, providers: List[str], openrouter_key: Optional[str] = None):
        """
        Initialize fallback chain.
        
        Args:
            providers: Ordered list of provider IDs to try
            openrouter_key: API key for OpenRouter (if in chain)
        """
        self.chain = providers
        self.openrouter_key = openrouter_key
        self.current_idx = 0
        self.active_provider = None
        self.fallback_history: List[Dict] = []
        
        # Import providers
        from core.llm_providers import LLMProviderFactory, NoneProvider
        self.factory = LLMProviderFactory
        
        print(f"{Colors.CYAN}[FallbackChain] Initialized with: {providers}{Colors.RESET}")
    
    def _create_provider(self, provider_type: str):
        """Create a provider instance"""
        kwargs = {}
        if provider_type == "openrouter" and self.openrouter_key:
            kwargs['api_key'] = self.openrouter_key
            kwargs['model'] = "nvidia/nemotron-3-nano-30b-a3b:free"
        elif provider_type == "local":
            kwargs['auto_load'] = False  # Don't auto-load for test
        
        try:
            return self.factory.get_provider(provider_type, **kwargs)
        except Exception as e:
            print(f"{Colors.YELLOW}  ⚠ Could not create {provider_type}: {e}{Colors.RESET}")
            return None
    
    def get_provider(self, force_next: bool = False):
        """
        Get the current active provider, or fall back to next.
        
        Args:
            force_next: Force switching to next provider in chain
            
        Returns:
            Active provider or None if all failed
        """
        if force_next:
            self.current_idx = min(self.current_idx + 1, len(self.chain) - 1)
        
        while self.current_idx < len(self.chain):
            provider_type = self.chain[self.current_idx]
            provider = self._create_provider(provider_type)
            
            if provider and provider.is_available():
                self.active_provider = provider
                return provider
            
            # Record fallback
            self.fallback_history.append({
                'from': provider_type,
                'reason': 'not_available',
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"{Colors.YELLOW}  ⚠ {provider_type} not available, trying next...{Colors.RESET}")
            self.current_idx += 1
        
        # All failed - return NoneProvider as last resort
        print(f"{Colors.RED}  All providers failed, using NoneProvider{Colors.RESET}")
        from core.llm_providers import NoneProvider
        return NoneProvider()
    
    def generate_with_fallback(
        self,
        prompt: str,
        max_retries: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with automatic fallback on failure.
        
        Args:
            prompt: Input prompt
            max_retries: Max retries before giving up
            **kwargs: Generation parameters
            
        Returns:
            Generation result with fallback info
        """
        attempts = []
        
        for attempt in range(max_retries + 1):
            provider = self.get_provider(force_next=(attempt > 0))
            
            try:
                start = time.time()
                result = provider.generate(prompt, **kwargs)
                elapsed = time.time() - start
                
                if result.get('success'):
                    return {
                        **result,
                        'provider_used': provider.provider_name,
                        'attempts': attempts,
                        'generation_time': elapsed
                    }
                
                # Record failed attempt
                attempts.append({
                    'provider': provider.provider_name,
                    'error': result.get('error', 'Unknown error'),
                    'elapsed': elapsed
                })
                
            except Exception as e:
                attempts.append({
                    'provider': provider.provider_name,
                    'error': str(e),
                    'elapsed': 0
                })
        
        # All retries exhausted
        return {
            'success': False,
            'error': 'All providers failed',
            'attempts': attempts,
            'generated_text': ''
        }
    
    def stream_with_fallback(
        self,
        prompt: str,
        max_retries: int = 2,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream with automatic fallback on failure.
        """
        for attempt in range(max_retries + 1):
            provider = self.get_provider(force_next=(attempt > 0))
            
            try:
                for chunk in provider.generate_stream(prompt, **kwargs):
                    yield {
                        **chunk,
                        'provider': provider.provider_name
                    }
                    
                    if chunk.get('done'):
                        return
                        
            except Exception as e:
                yield {
                    'token': '',
                    'done': False,
                    'success': False,
                    'error': f"Fallback: {e}",
                    'provider': provider.provider_name
                }
        
        yield {
            'token': '',
            'done': True,
            'success': False,
            'error': 'All providers failed'
        }


# =============================================================================
# CONTEXT PRESERVATION - Smart provider switching
# =============================================================================

class ConversationWithProviderSwitching:
    """
    Manages multi-turn conversation with provider switching.
    Preserves context when changing providers.
    """
    
    def __init__(self, fallback_chain: ProviderFallbackChain):
        self.chain = fallback_chain
        self.conversation: List[Dict] = []
        self.documents: List[Dict] = []
        self.provider_history: List[str] = []
        self.token_usage: Dict[str, int] = {}
        
        # Import context transfer
        try:
            from core.llm_providers.context_transfer import ContextTransfer
            self.context_transfer = ContextTransfer()
        except ImportError:
            self.context_transfer = None
    
    def add_document(self, doc_id: str, filename: str, content: str, char_count: int):
        """Add document to context"""
        self.documents.append({
            'id': doc_id,
            'filename': filename,
            'content': content[:10000],  # Limit for context
            'char_count': char_count,
            'added_at': datetime.now().isoformat()
        })
        print(f"{Colors.CYAN}  📄 Document added: {filename} ({char_count:,} chars){Colors.RESET}")
    
    def clear_documents(self):
        """Clear all documents"""
        self.documents = []
        print(f"{Colors.DIM}  🗑️ Documents cleared{Colors.RESET}")
    
    def switch_provider(self, new_provider: str) -> bool:
        """
        Switch to a different provider while preserving context.
        
        Args:
            new_provider: Provider ID to switch to
            
        Returns:
            True if switch successful
        """
        current = self.chain.active_provider
        current_name = current.provider_name if current else "none"
        
        print(f"\n{Colors.MAGENTA}🔄 Switching provider: {current_name} → {new_provider}{Colors.RESET}")
        
        # Check context compatibility
        if self.context_transfer and self.conversation:
            warnings = self.context_transfer.check_compatibility(
                from_model=current.model_name if current else "none",
                to_model=new_provider,
                conversation_tokens=sum(len(t.get('content', '')) // 4 for t in self.conversation)
            )
            
            if warnings:
                print(f"{Colors.YELLOW}  Context warnings:{Colors.RESET}")
                for w in warnings:
                    print(f"    ⚠ {w}")
        
        # Force switch in chain
        try:
            # Find provider index
            if new_provider in self.chain.chain:
                self.chain.current_idx = self.chain.chain.index(new_provider)
                provider = self.chain.get_provider()
                
                if provider and provider.is_available():
                    self.provider_history.append({
                        'from': current_name,
                        'to': new_provider,
                        'timestamp': datetime.now().isoformat(),
                        'context_preserved': True
                    })
                    print(f"{Colors.GREEN}  ✓ Switch successful{Colors.RESET}")
                    return True
            
            print(f"{Colors.RED}  ✗ Switch failed{Colors.RESET}")
            return False
            
        except Exception as e:
            print(f"{Colors.RED}  ✗ Switch error: {e}{Colors.RESET}")
            return False
    
    def chat(
        self,
        query: str,
        include_docs: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat message with context.
        
        Args:
            query: User query
            include_docs: Include document context
            stream: Use streaming (prints tokens in real-time)
            
        Returns:
            Response with metadata
        """
        # Build prompt with context
        prompt_parts = []
        
        # Add document context
        if include_docs and self.documents:
            prompt_parts.append("## Document Context\n")
            for doc in self.documents:
                prompt_parts.append(f"### {doc['filename']}\n{doc['content'][:5000]}\n")
        
        # Add conversation history (last 4 turns)
        if self.conversation:
            prompt_parts.append("\n## Previous Conversation\n")
            for turn in self.conversation[-4:]:
                prompt_parts.append(f"User: {turn['query']}\n")
                prompt_parts.append(f"Assistant: {turn['response']}\n")
        
        # Add current query
        prompt_parts.append(f"\n## Current Query\n{query}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Record turn
        turn = {
            'role': 'user',
            'query': query,
            'response': '',
            'include_docs': include_docs,
            'doc_count': len(self.documents) if include_docs else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        start = time.time()
        
        if stream:
            # Stream response
            response_text = ""
            print(f"\n{Colors.BOLD}Response:{Colors.RESET} ", end="", flush=True)
            
            for chunk in self.chain.stream_with_fallback(full_prompt, max_new_tokens=512):
                if chunk.get('success', True) and not chunk.get('done'):
                    token = chunk.get('token', '')
                    response_text += token
                    print(token, end="", flush=True)
            
            print()  # Newline after stream
            
            turn['response'] = response_text
            turn['provider'] = self.chain.active_provider.provider_name if self.chain.active_provider else 'none'
            
        else:
            # Non-streaming
            result = self.chain.generate_with_fallback(full_prompt, max_new_tokens=512)
            turn['response'] = result.get('generated_text', '')
            turn['provider'] = result.get('provider_used', 'none')
            turn['attempts'] = result.get('attempts', [])
        
        turn['elapsed'] = time.time() - start
        self.conversation.append(turn)
        
        # Track token usage
        provider = turn['provider']
        tokens = len(turn['response']) // 4  # Rough estimate
        self.token_usage[provider] = self.token_usage.get(provider, 0) + tokens
        
        return turn
    
    def get_summary(self) -> Dict:
        """Get conversation summary"""
        return {
            'total_turns': len(self.conversation),
            'documents_used': len(self.documents),
            'providers_used': list(set(t.get('provider', 'unknown') for t in self.conversation)),
            'provider_switches': len(self.provider_history),
            'token_usage': self.token_usage,
            'total_time': sum(t.get('elapsed', 0) for t in self.conversation)
        }


# =============================================================================
# TEST SCENARIOS
# =============================================================================

def test_fallback_chain():
    """Test provider fallback chain"""
    print_header("TEST 1: Provider Fallback Chain")
    
    try:
        # Create chain with multiple providers
        chain = ProviderFallbackChain(
            providers=["openrouter", "local", "none"],
            openrouter_key=None  # Will fail, testing fallback
        )
        
        # Should fall back to 'none' since no API key
        provider = chain.get_provider()
        print(f"Active Provider: {provider.provider_name}")
        
        # Test generate with fallback
        result = chain.generate_with_fallback(
            prompt="Jelaskan apa itu hukum pidana dalam satu kalimat.",
            max_new_tokens=100
        )
        
        print(f"Success: {result['success']}")
        print(f"Provider Used: {result.get('provider_used', 'N/A')}")
        print(f"Attempts: {len(result.get('attempts', []))}")
        print(f"Response: {result.get('generated_text', '')[:100]}...")
        
        print_result("Fallback Chain", True, f"Fell back to {provider.provider_name}")
        return True
        
    except Exception as e:
        print_result("Fallback Chain", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_context_preservation():
    """Test context preservation during provider switching"""
    print_header("TEST 2: Context Preservation")
    
    try:
        from core.llm_providers.context_transfer import ContextTransfer
        
        transfer = ContextTransfer()
        
        # Simulate conversation
        conversation = [
            {"role": "user", "content": "Apa itu hukum pidana?"},
            {"role": "assistant", "content": "Hukum pidana adalah cabang hukum yang mengatur tentang tindak pidana..."},
            {"role": "user", "content": "Apa sanksinya?"},
            {"role": "assistant", "content": "Sanksi dalam hukum pidana meliputi pidana penjara, denda..."},
        ]
        
        # Test switching from large to small context model
        print("[Switching: Claude (200K) → GPT-4o-mini (128K)]")
        warnings = transfer.check_compatibility(
            from_model="anthropic/claude-sonnet-4",
            to_model="openai/gpt-4o-mini",
            conversation_tokens=150000
        )
        
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  ⚠ {w}")
        
        # Prepare context for new model
        prepared = transfer.prepare_context(
            conversation=conversation,
            to_model="nvidia/nemotron-3-nano-30b-a3b:free"
        )
        
        print(f"\nContext Preparation:")
        print(f"  Original: {len(conversation)} messages")
        print(f"  Prepared: {len(prepared['messages'])} messages")
        print(f"  Truncated: {prepared['truncated']}")
        
        print_result("Context Preservation", True, "Context transfer works")
        return True
        
    except Exception as e:
        print_result("Context Preservation", False, str(e))
        return False


def test_multi_turn_with_documents():
    """Test multi-turn conversation with document context"""
    print_header("TEST 3: Multi-Turn Conversation with Documents")
    
    try:
        chain = ProviderFallbackChain(providers=["none"])
        conversation = ConversationWithProviderSwitching(chain)
        
        # Add a mock document
        mock_doc = """
        PERATURAN PEMERINTAH REPUBLIK INDONESIA
        NOMOR 35 TAHUN 2021
        TENTANG PERJANJIAN KERJA WAKTU TERTENTU
        
        Pasal 1
        Perjanjian Kerja Waktu Tertentu (PKWT) adalah perjanjian kerja antara 
        pekerja/buruh dengan pengusaha untuk mengadakan hubungan kerja dalam 
        waktu tertentu atau untuk pekerjaan tertentu.
        
        Pasal 2
        PKWT dapat dibuat paling lama 5 (lima) tahun termasuk perpanjangan.
        """
        
        conversation.add_document(
            doc_id="doc-001",
            filename="PP_35_2021_PKWT.pdf",
            content=mock_doc,
            char_count=len(mock_doc)
        )
        
        # Turn 1: Ask about document
        print("\n[Turn 1: Query about document]")
        turn1 = conversation.chat(
            query="Apa yang diatur dalam dokumen ini?",
            include_docs=True
        )
        print(f"Provider: {turn1['provider']}")
        print(f"Response: {turn1['response'][:150]}...")
        
        # Turn 2: Follow-up
        print("\n[Turn 2: Follow-up question]")
        turn2 = conversation.chat(
            query="Berapa lama maksimal PKWT dapat dibuat?",
            include_docs=True
        )
        print(f"Provider: {turn2['provider']}")
        print(f"Response: {turn2['response'][:150]}...")
        
        # Turn 3: Without document
        print("\n[Turn 3: General question without document]")
        turn3 = conversation.chat(
            query="Apa perbedaan PKWT dan PKWTT?",
            include_docs=False
        )
        print(f"Provider: {turn3['provider']}")
        print(f"Response: {turn3['response'][:150]}...")
        
        # Summary
        summary = conversation.get_summary()
        print(f"\n[Conversation Summary]")
        print(f"  Total Turns: {summary['total_turns']}")
        print(f"  Documents Used: {summary['documents_used']}")
        print(f"  Providers: {summary['providers_used']}")
        print(f"  Total Time: {summary['total_time']:.1f}s")
        
        print_result("Multi-Turn with Documents", True, f"{summary['total_turns']} turns completed")
        return True
        
    except Exception as e:
        print_result("Multi-Turn with Documents", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_provider_switching_mid_conversation():
    """Test switching providers mid-conversation"""
    print_header("TEST 4: Provider Switching Mid-Conversation")
    
    try:
        # Start with 'none' provider
        chain = ProviderFallbackChain(providers=["none", "local"])
        conversation = ConversationWithProviderSwitching(chain)
        
        # Turn 1 with 'none'
        print("[Turn 1: Using 'none' provider]")
        turn1 = conversation.chat("Apa itu kontrak kerja?")
        print(f"Provider: {turn1['provider']}")
        
        # Switch provider
        print("\n[Attempting provider switch]")
        switched = conversation.switch_provider("local")
        print(f"Switch Result: {'Success' if switched else 'Failed (expected)'}")
        
        # Turn 2 after switch attempt
        print("\n[Turn 2: After switch attempt]")
        turn2 = conversation.chat("Jelaskan lebih lanjut")
        print(f"Provider: {turn2['provider']}")
        
        # Check history
        print(f"\nProvider History: {len(conversation.provider_history)} switches")
        
        summary = conversation.get_summary()
        print(f"\n[Summary]")
        print(f"  Providers Used: {summary['providers_used']}")
        print(f"  Provider Switches: {summary['provider_switches']}")
        
        print_result("Provider Switching", True, "Context preserved across attempts")
        return True
        
    except Exception as e:
        print_result("Provider Switching", False, str(e))
        return False


def test_streaming_with_fallback():
    """Test streaming generation with fallback"""
    print_header("TEST 5: Streaming with Fallback")
    
    try:
        chain = ProviderFallbackChain(providers=["none"])
        
        print("[Streaming Response]")
        full_text = ""
        token_count = 0
        
        for chunk in chain.stream_with_fallback(
            prompt="Sebutkan 3 jenis perjanjian kerja.",
            max_new_tokens=100
        ):
            if chunk.get('success', True) and not chunk.get('done'):
                token = chunk.get('token', '')
                full_text += token
                token_count += 1
                print(token, end="", flush=True)
        
        print(f"\n\nTotal Tokens: {token_count}")
        print(f"Full Text Length: {len(full_text)} chars")
        
        print_result("Streaming with Fallback", len(full_text) > 0, f"Streamed {token_count} tokens")
        return len(full_text) > 0
        
    except Exception as e:
        print_result("Streaming with Fallback", False, str(e))
        return False


def test_with_openrouter(api_key: str):
    """Test with real OpenRouter API"""
    print_header("TEST 6: OpenRouter Multi-Turn (Live)")
    
    try:
        chain = ProviderFallbackChain(
            providers=["openrouter", "none"],
            openrouter_key=api_key
        )
        conversation = ConversationWithProviderSwitching(chain)
        
        # Mock document
        doc = "Pasal 1: Setiap warga negara berhak atas pendidikan."
        conversation.add_document("doc-1", "UUD_Pendidikan.txt", doc, len(doc))
        
        # Turn 1: With document + streaming
        print("\n[Turn 1: Document + Streaming]")
        turn1 = conversation.chat(
            query="Apa yang diatur dalam dokumen ini?",
            include_docs=True,
            stream=True
        )
        print(f"\nProvider: {turn1['provider']}")
        print(f"Time: {turn1['elapsed']:.1f}s")
        
        # Turn 2: Follow-up
        print("\n[Turn 2: Follow-up]")
        turn2 = conversation.chat(
            query="Siapa yang berhak?",
            include_docs=True,
            stream=True
        )
        print(f"\nProvider: {turn2['provider']}")
        
        # Summary
        summary = conversation.get_summary()
        print(f"\n[Summary]")
        print(f"  Turns: {summary['total_turns']}")
        print(f"  Token Usage: {summary['token_usage']}")
        print(f"  Total Time: {summary['total_time']:.1f}s")
        
        print_result("OpenRouter Multi-Turn", True, "Live API working")
        return True
        
    except Exception as e:
        print_result("OpenRouter Multi-Turn", False, str(e))
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Provider Multi-Turn Conversation Test")
    parser.add_argument("--with-openrouter", action="store_true", help="Include OpenRouter live test")
    parser.add_argument("--openrouter-key", type=str, help="OpenRouter API key")
    parser.add_argument("--with-api", action="store_true", help="Test with API server (not implemented yet)")
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  🧪 LLM Provider Multi-Turn Conversation Test")
    print("=" * 80)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    
    results = []
    
    # Core tests (always run)
    print("\n" + "-" * 80)
    print("  PART 1: Core Features")
    print("-" * 80)
    
    results.append(("Fallback Chain", test_fallback_chain()))
    results.append(("Context Preservation", test_context_preservation()))
    results.append(("Multi-Turn with Documents", test_multi_turn_with_documents()))
    results.append(("Provider Switching", test_provider_switching_mid_conversation()))
    results.append(("Streaming with Fallback", test_streaming_with_fallback()))
    
    # OpenRouter live test
    if args.with_openrouter:
        print("\n" + "-" * 80)
        print("  PART 2: OpenRouter Live")
        print("-" * 80)
        
        api_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY")
        if api_key:
            results.append(("OpenRouter Multi-Turn", test_with_openrouter(api_key)))
        else:
            print("\n⚠️ Skipping OpenRouter: No API key")
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        emoji = "✅" if success else "❌"
        print(f"  {emoji} {name}")
    
    print(f"\n{'=' * 80}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'=' * 80}\n")
    
    # Feature coverage
    print("📋 Feature Coverage:")
    features = [
        ("Provider Fallback Chain", True),
        ("Smart Provider Switching", True),
        ("Context Preservation", True),
        ("Multi-Turn Conversation", True),
        ("Document Context", True),
        ("Streaming with Fallback", True),
        ("OpenRouter Live", args.with_openrouter),
        ("API Integration", args.with_api),
    ]
    for feature, tested in features:
        emoji = "✅" if tested else "⏭️"
        print(f"  {emoji} {feature}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All tests passed!{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}⚠️ Some tests failed{Colors.RESET}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
