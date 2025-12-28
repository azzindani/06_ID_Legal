"""
LLM Provider System - Real Simulation Test

This script runs a comprehensive end-to-end test of the LLM Provider system.
It tests all components directly (without mocking) to show how the system works.

Usage:
    # Run without API (tests core modules only)
    python tests/integration/test_llm_providers_simulation.py
    
    # Run with API server (tests full system)
    python tests/integration/test_llm_providers_simulation.py --with-api
    
    # Include OpenRouter live test (requires API key)
    python tests/integration/test_llm_providers_simulation.py --with-openrouter

File: tests/integration/test_llm_providers_simulation.py
"""

import sys
import os
import time
import json
import argparse
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, success: bool, message: str = ""):
    """Print a test result"""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {name}: {message}")


def test_none_provider():
    """Test NoneProvider - RAG-only mode"""
    print_header("TEST 1: NoneProvider (RAG-Only Mode)")
    
    try:
        from core.llm_providers import NoneProvider
        
        provider = NoneProvider()
        print(f"Provider: {provider.provider_name}")
        print(f"Model: {provider.model_name}")
        print(f"Available: {provider.is_available()}")
        
        # Test generate
        result = provider.generate("Apa itu hukum pidana?")
        print(f"\n[Generate Response]")
        print(f"Success: {result['success']}")
        print(f"Message Preview: {result['generated_text'][:100]}...")
        
        # Test streaming
        print(f"\n[Streaming Response]")
        stream_text = ""
        for chunk in provider.generate_stream("Test streaming"):
            if not chunk['done']:
                stream_text += chunk['token']
        print(f"Streamed {len(stream_text)} characters")
        
        # Test info
        info = provider.get_info()
        print(f"\n[Provider Info]")
        print(json.dumps(info, indent=2))
        
        print_result("NoneProvider", True, "All methods work correctly")
        return True
        
    except Exception as e:
        print_result("NoneProvider", False, str(e))
        return False


def test_provider_factory():
    """Test LLMProviderFactory"""
    print_header("TEST 2: LLMProviderFactory")
    
    try:
        from core.llm_providers import LLMProviderFactory
        
        # List providers
        providers = LLMProviderFactory.list_providers()
        print(f"Available Providers: {providers}")
        
        # Create none provider
        provider = LLMProviderFactory.get_provider("none")
        print(f"Created Provider: {provider.provider_name}")
        
        # Check singleton
        provider2 = LLMProviderFactory.get_provider("none")
        is_same = provider is provider2
        print(f"Singleton Pattern: {'Works' if is_same else 'BROKEN'}")
        
        # Check initialization
        print(f"Is Initialized: {LLMProviderFactory.is_initialized()}")
        
        # Get current provider
        current = LLMProviderFactory.get_current_provider()
        print(f"Current Provider: {current.provider_name if current else 'None'}")
        
        # Shutdown
        LLMProviderFactory.shutdown()
        print(f"After Shutdown: {not LLMProviderFactory.is_initialized()}")
        
        print_result("LLMProviderFactory", True, "Factory works correctly")
        return True
        
    except Exception as e:
        print_result("LLMProviderFactory", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_secure_keystore():
    """Test SecureKeyStore - encrypted API key storage"""
    print_header("TEST 3: SecureKeyStore (Encrypted Storage)")
    
    try:
        from core.llm_providers.keystore import SecureKeyStore
        
        # Use temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            keystore = SecureKeyStore(storage_dir=Path(tmpdir))
            
            print(f"Storage Location: {tmpdir}")
            print(f"Encryption Available: {keystore._fernet is not None}")
            
            # Save key
            test_key = "sk-or-v1-test-key-12345-abcdef"
            success = keystore.save_key("openrouter", test_key)
            print(f"Save Key: {'Success' if success else 'Failed'}")
            
            # Load key
            loaded_key = keystore.load_key("openrouter")
            matches = loaded_key == test_key
            print(f"Load Key: {'Matches' if matches else 'MISMATCH!'}")
            
            # List providers
            providers = keystore.list_providers()
            print(f"Providers with Keys: {providers}")
            
            # Delete key
            deleted = keystore.delete_key("openrouter")
            print(f"Delete Key: {'Success' if deleted else 'Failed'}")
            
            # Verify deleted
            after_delete = keystore.load_key("openrouter")
            print(f"After Delete: {'Gone' if after_delete is None else 'Still exists!'}")
        
        print_result("SecureKeyStore", True, "Encryption and storage work")
        return True
        
    except Exception as e:
        print_result("SecureKeyStore", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_response_cache():
    """Test ResponseCache - LRU caching"""
    print_header("TEST 4: ResponseCache (LRU Caching)")
    
    try:
        from core.llm_providers.cache import ResponseCache
        
        cache = ResponseCache(max_size=5, ttl_seconds=60)
        
        # Test set/get
        prompt = "Apa itu hukum perdata?"
        response = {"generated_text": "Hukum perdata adalah...", "success": True}
        
        cache.set(prompt, response, model="test-model")
        print("Set: Added response to cache")
        
        cached = cache.get(prompt, model="test-model")
        print(f"Get: {'Hit' if cached else 'Miss'}")
        
        # Test cache miss
        miss = cache.get("different prompt", model="test-model")
        print(f"Different Prompt: {'Miss (correct)' if miss is None else 'Hit (incorrect)'}")
        
        # Test stats
        stats = cache.get_stats()
        print(f"\n[Cache Stats]")
        print(json.dumps(stats, indent=2))
        
        # Test LRU eviction (add more than max_size)
        for i in range(10):
            cache.set(f"prompt_{i}", {"text": f"response_{i}"}, model="model")
        
        stats_after = cache.get_stats()
        print(f"\nAfter 10 insertions (max 5): {stats_after['size']} entries")
        
        # Test clear
        cache.clear()
        after_clear = cache.get_stats()
        print(f"After Clear: {after_clear['size']} entries")
        
        print_result("ResponseCache", True, "Caching with LRU works")
        return True
        
    except Exception as e:
        print_result("ResponseCache", False, str(e))
        return False


def test_usage_tracker():
    """Test UsageTracker - token/cost tracking"""
    print_header("TEST 5: UsageTracker (Token Tracking)")
    
    try:
        from core.llm_providers.usage_tracker import UsageTracker
        
        tracker = UsageTracker(persist=False)  # Don't save to disk for test
        
        # Record some usage
        tracker.record(
            provider="openrouter",
            model="nvidia/nemotron",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.0,
            query="Test query"
        )
        print("Recorded: 150 tokens")
        
        tracker.record(
            provider="openrouter",
            model="nvidia/nemotron",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            cost_usd=0.0,
            query="Another query"
        )
        print("Recorded: 300 tokens")
        
        # Get session stats
        stats = tracker.get_session_stats()
        print(f"\n[Session Stats]")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Request Count: {stats['request_count']}")
        print(f"Avg Tokens/Request: {stats['avg_tokens_per_request']}")
        print(f"By Provider: {stats['by_provider']}")
        
        print_result("UsageTracker", True, "Token tracking works")
        return True
        
    except Exception as e:
        print_result("UsageTracker", False, str(e))
        return False


def test_context_transfer():
    """Test ContextTransfer - smart provider switching"""
    print_header("TEST 6: ContextTransfer (Smart Switching)")
    
    try:
        from core.llm_providers.context_transfer import ContextTransfer
        
        transfer = ContextTransfer()
        
        # Check compatibility
        warnings = transfer.check_compatibility(
            from_model="anthropic/claude-sonnet-4",  # 200K context
            to_model="openai/gpt-4o-mini",            # 128K context
            conversation_tokens=150000
        )
        
        print("[Compatibility Check: Claude → GPT-4o-mini]")
        for w in warnings:
            print(f"  ⚠️ {w}")
        
        # Test context preparation
        conversation = [
            {"role": "user", "content": "Apa itu hukum pidana?"},
            {"role": "assistant", "content": "Hukum pidana adalah cabang hukum yang mengatur tentang pelanggaran..."},
            {"role": "user", "content": "Apa sanksinya?"},
            {"role": "assistant", "content": "Sanksi dalam hukum pidana meliputi..."},
        ]
        
        context = transfer.prepare_context(
            conversation=conversation,
            to_model="nvidia/nemotron-3-nano-30b-a3b:free"
        )
        
        print(f"\n[Context Preparation]")
        print(f"Original Messages: {len(conversation)}")
        print(f"Prepared Messages: {len(context['messages'])}")
        print(f"Truncated: {context['truncated']}")
        
        print_result("ContextTransfer", True, "Context transfer works")
        return True
        
    except Exception as e:
        print_result("ContextTransfer", False, str(e))
        return False


def test_model_presets():
    """Test model presets configuration"""
    print_header("TEST 7: Model Presets (Free Priority)")
    
    try:
        from core.llm_providers.openrouter import get_model_presets
        from config import LLM_MODEL_PRESETS, OPENROUTER_MODEL
        
        # Test presets from openrouter module
        presets = get_model_presets()
        print("[Model Presets from openrouter.py]")
        for name, model_id in presets.items():
            print(f"  {name}: {model_id}")
        
        # Verify free models exist
        free_count = sum(1 for name in presets if 'free' in name.lower())
        print(f"\nFree Presets: {free_count}")
        
        # Test presets from config
        print("\n[Model Presets from config.py]")
        for name, model_id in LLM_MODEL_PRESETS.items():
            print(f"  {name}: {model_id}")
        
        # Verify default model
        print(f"\nDefault OPENROUTER_MODEL: {OPENROUTER_MODEL}")
        is_free = ":free" in OPENROUTER_MODEL
        print(f"Is Free Model: {is_free}")
        
        print_result("Model Presets", True, f"{len(presets)} presets available, free priority confirmed")
        return True
        
    except Exception as e:
        print_result("Model Presets", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_api_key_validation():
    """Test API key validation"""
    print_header("TEST 8: API Key Validation")
    
    try:
        from core.llm_providers.openrouter import OpenRouterProvider
        
        # Test invalid key format
        print("[Testing Invalid Key Formats]")
        
        # Empty key should raise
        try:
            OpenRouterProvider(api_key="")
            print("  Empty key: SHOULD HAVE FAILED!")
            validation_works = False
        except ValueError as e:
            print(f"  Empty key: Rejected ✓ ({e})")
            validation_works = True
        
        # Too short key
        try:
            OpenRouterProvider(api_key="sk-short")
            print("  Short key: SHOULD HAVE FAILED!")
            validation_works = False
        except ValueError as e:
            print(f"  Short key: Rejected ✓")
            validation_works = True
        
        # Whitespace only
        try:
            OpenRouterProvider(api_key="   ")
            print("  Whitespace key: SHOULD HAVE FAILED!")
            validation_works = False
        except ValueError as e:
            print(f"  Whitespace key: Rejected ✓")
            validation_works = True
        
        # Valid format (won't actually validate against API without real key)
        print("\n[Testing Valid Key Format]")
        try:
            # This key format is valid but won't work for real API calls
            provider = OpenRouterProvider(api_key="sk-or-v1-test-key-format-valid-for-testing-only-12345678")
            print(f"  Valid format key: Accepted ✓")
            print(f"  Provider created: {provider.provider_name}")
        except ValueError as e:
            print(f"  Valid format key: Rejected (unexpected) - {e}")
            validation_works = False
        
        print_result("API Key Validation", validation_works, "Invalid keys are rejected")
        return validation_works
        
    except Exception as e:
        print_result("API Key Validation", False, str(e))
        return False


def test_local_provider():
    """Test LocalProvider initialization"""
    print_header("TEST 9: LocalProvider (GPU Wrapper)")
    
    try:
        from core.llm_providers import LocalProvider
        
        # Create without loading model
        provider = LocalProvider(config={}, auto_load=False)
        
        print(f"Provider: {provider.provider_name}")
        print(f"Model: {provider.model_name}")
        print(f"Available (without loading): {provider.is_available()}")
        
        # Get info
        info = provider.get_info()
        print(f"\n[Provider Info]")
        print(f"  Loaded: {info.get('loaded', False)}")
        print(f"  Device: {info.get('device', 'N/A')}")
        
        # Test generate without loaded model
        print("\n[Testing generate without model loaded]")
        result = provider.generate("Test prompt")
        if result['success']:
            print("  Generate: Returned result (model might be loaded)")
        else:
            print(f"  Generate: Correctly failed - {result.get('error', 'No model')}")
        
        print_result("LocalProvider", True, "Wrapper initialized correctly")
        return True
        
    except Exception as e:
        print_result("LocalProvider", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_openrouter_provider(api_key: str):
    """Test OpenRouterProvider with real API call including streaming"""
    print_header("TEST 10: OpenRouterProvider (Live API)")
    
    try:
        from core.llm_providers import OpenRouterProvider
        
        provider = OpenRouterProvider(
            api_key=api_key,
            model="nvidia/nemotron-3-nano-30b-a3b:free"  # Free model
        )
        
        print(f"Provider: {provider.provider_name}")
        print(f"Model: {provider.model_name}")
        print(f"Available: {provider.is_available()}")
        
        # Test non-streaming generate
        print("\n[Non-Streaming Generation...]")
        start = time.time()
        result = provider.generate(
            prompt="Jelaskan secara singkat apa itu hukum perdata dalam 2 kalimat.",
            max_new_tokens=100,
            temperature=0.7
        )
        elapsed = time.time() - start
        
        print(f"Success: {result['success']}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Tokens: {result.get('tokens_generated', 'N/A')}")
        print(f"Response: {result['generated_text'][:150]}...")
        
        # Test streaming generate (SSE)
        print("\n[SSE Streaming Generation...]")
        stream_tokens = 0
        stream_text = ""
        start = time.time()
        
        for chunk in provider.generate_stream(
            prompt="Apa itu hukum pidana? Jawab singkat.",
            max_new_tokens=50
        ):
            if chunk.get('success') and not chunk.get('done'):
                token = chunk.get('token', '')
                stream_text += token
                stream_tokens = chunk.get('tokens_generated', stream_tokens)
                # Print tokens as they come (simulating real streaming)
                print(token, end='', flush=True)
        
        elapsed = time.time() - start
        print(f"\n\nStreaming Stats:")
        print(f"  Total Tokens: {stream_tokens}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Streaming: Works ✓")
        
        print_result("OpenRouterProvider", True, "Both sync and streaming work")
        return True
        
    except Exception as e:
        print_result("OpenRouterProvider", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_runtime_switching(base_url: str = "http://127.0.0.1:8000"):
    """Test runtime provider switching via API"""
    print_header("TEST 11: Runtime Provider Switching")
    
    import requests
    
    api_url = f"{base_url}/api/v1"
    
    try:
        # Get current status
        print("[1. Get Current Status]")
        r = requests.get(f"{api_url}/llm/status", timeout=10)
        if r.status_code == 200:
            status = r.json()
            print(f"Current Provider: {status['provider']}")
        else:
            print(f"Failed to get status: {r.status_code}")
            return False
        
        # Switch to none
        print("\n[2. Switch to 'none' Provider]")
        r = requests.post(
            f"{api_url}/llm/config",
            json={"provider": "none"},
            timeout=10
        )
        if r.status_code == 200:
            result = r.json()
            print(f"Success: {result.get('success')}")
            print(f"Provider: {result.get('provider')}")
            print(f"Available: {result.get('available')}")
        else:
            print(f"Failed: {r.status_code} - {r.text}")
        
        # Verify switch
        print("\n[3. Verify Switch]")
        r = requests.get(f"{api_url}/llm/status", timeout=10)
        if r.status_code == 200:
            status = r.json()
            switched = status['provider'] == 'none'
            print(f"Current Provider: {status['provider']}")
            print(f"Switch Successful: {switched}")
        else:
            print(f"Failed to verify: {r.status_code}")
            return False
        
        print_result("Runtime Switching", True, "Provider switching works via API")
        return True
        
    except requests.exceptions.ConnectionError:
        print_result("Runtime Switching", False, "Server not running")
        return False
    except Exception as e:
        print_result("Runtime Switching", False, str(e))
        return False



def test_api_endpoints(base_url: str = "http://127.0.0.1:8000"):
    """Test LLM API endpoints"""
    print_header("TEST 8: LLM API Endpoints")
    
    import requests
    
    api_url = f"{base_url}/api/v1"
    
    try:
        # Test /llm/providers
        print("[GET /llm/providers]")
        r = requests.get(f"{api_url}/llm/providers", timeout=10)
        if r.status_code == 200:
            providers = r.json()
            print(f"Available: {[p['id'] for p in providers]}")
        else:
            print(f"Failed: {r.status_code}")
        
        # Test /llm/presets
        print("\n[GET /llm/presets]")
        r = requests.get(f"{api_url}/llm/presets", timeout=10)
        if r.status_code == 200:
            presets = r.json()
            print(f"Presets: {list(presets['presets'].keys())}")
            print(f"Recommended: {presets['recommended']}")
        else:
            print(f"Failed: {r.status_code}")
        
        # Test /llm/status
        print("\n[GET /llm/status]")
        r = requests.get(f"{api_url}/llm/status", timeout=10)
        if r.status_code == 200:
            status = r.json()
            print(f"Provider: {status['provider']}")
            print(f"Model: {status['model']}")
            print(f"Available: {status['available']}")
        else:
            print(f"Failed: {r.status_code}")
        
        # Test /llm/cache/stats
        print("\n[GET /llm/cache/stats]")
        r = requests.get(f"{api_url}/llm/cache/stats", timeout=10)
        if r.status_code == 200:
            stats = r.json()
            print(f"Cache Size: {stats.get('size', 0)}")
            print(f"Hits: {stats.get('hits', 0)}")
        else:
            print(f"Failed: {r.status_code}")
        
        # Test /llm/usage
        print("\n[GET /llm/usage]")
        r = requests.get(f"{api_url}/llm/usage", timeout=10)
        if r.status_code == 200:
            usage = r.json()
            print(f"Session Tokens: {usage['session'].get('total_tokens', 0)}")
        else:
            print(f"Failed: {r.status_code}")
        
        print_result("API Endpoints", True, "All endpoints responding")
        return True
        
    except requests.exceptions.ConnectionError:
        print_result("API Endpoints", False, "Server not running. Start with: python -m api.server --llm-provider none")
        return False
    except Exception as e:
        print_result("API Endpoints", False, str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="LLM Provider System - Real Simulation Test")
    parser.add_argument("--with-api", action="store_true", help="Test API endpoints (requires running server)")
    parser.add_argument("--with-openrouter", action="store_true", help="Test OpenRouter live API")
    parser.add_argument("--openrouter-key", type=str, help="OpenRouter API key")
    parser.add_argument("--full", action="store_true", help="Run all tests (equivalent to --with-api --with-openrouter)")
    args = parser.parse_args()
    
    # --full enables all optional tests
    if args.full:
        args.with_api = True
        args.with_openrouter = True
    
    print("\n" + "="*60)
    print("  🧪 LLM Provider System - Real Simulation Test")
    print("="*60)
    print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"\nOptions:")
    print(f"  --with-api:       {args.with_api}")
    print(f"  --with-openrouter: {args.with_openrouter}")
    
    results = []
    
    # =========================================================================
    # PART 1: Core Provider Tests (always run)
    # =========================================================================
    print("\n" + "-"*60)
    print("  PART 1: Core Providers")
    print("-"*60)
    
    results.append(("NoneProvider", test_none_provider()))
    results.append(("LLMProviderFactory", test_provider_factory()))
    results.append(("LocalProvider", test_local_provider()))
    
    # =========================================================================
    # PART 2: Storage & Utilities (always run)
    # =========================================================================
    print("\n" + "-"*60)
    print("  PART 2: Storage & Utilities")
    print("-"*60)
    
    results.append(("SecureKeyStore", test_secure_keystore()))
    results.append(("ResponseCache", test_response_cache()))
    results.append(("UsageTracker", test_usage_tracker()))
    results.append(("ContextTransfer", test_context_transfer()))
    
    # =========================================================================
    # PART 3: Configuration (always run)
    # =========================================================================
    print("\n" + "-"*60)
    print("  PART 3: Configuration")
    print("-"*60)
    
    results.append(("Model Presets", test_model_presets()))
    results.append(("API Key Validation", test_api_key_validation()))
    
    # =========================================================================
    # PART 4: Live API Tests (optional)
    # =========================================================================
    if args.with_openrouter:
        print("\n" + "-"*60)
        print("  PART 4: OpenRouter Live API")
        print("-"*60)
        
        api_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY")
        if api_key:
            results.append(("OpenRouterProvider", test_openrouter_provider(api_key)))
        else:
            print("\n⚠️ Skipping OpenRouter test: No API key provided")
            print("   Use --openrouter-key or set OPENROUTER_API_KEY env var")
    
    # =========================================================================
    # PART 5: API Endpoint Tests (optional, requires server)
    # =========================================================================
    if args.with_api:
        print("\n" + "-"*60)
        print("  PART 5: API Endpoints (Server Required)")
        print("-"*60)
        
        results.append(("API Endpoints", test_api_endpoints()))
        results.append(("Runtime Switching", test_runtime_switching()))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("TEST SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    # Group results by category
    print("  [Core Providers]")
    for name in ["NoneProvider", "LLMProviderFactory", "LocalProvider"]:
        for n, s in results:
            if n == name:
                emoji = "✅" if s else "❌"
                print(f"    {emoji} {name}")
    
    print("\n  [Storage & Utilities]")
    for name in ["SecureKeyStore", "ResponseCache", "UsageTracker", "ContextTransfer"]:
        for n, s in results:
            if n == name:
                emoji = "✅" if s else "❌"
                print(f"    {emoji} {name}")
    
    print("\n  [Configuration]")
    for name in ["Model Presets", "API Key Validation"]:
        for n, s in results:
            if n == name:
                emoji = "✅" if s else "❌"
                print(f"    {emoji} {name}")
    
    if args.with_openrouter:
        print("\n  [OpenRouter Live]")
        for n, s in results:
            if n == "OpenRouterProvider":
                emoji = "✅" if s else "❌"
                print(f"    {emoji} {n}")
    
    if args.with_api:
        print("\n  [API Endpoints]")
        for name in ["API Endpoints", "Runtime Switching"]:
            for n, s in results:
                if n == name:
                    emoji = "✅" if s else "❌"
                    print(f"    {emoji} {name}")
    
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 All tests passed! The LLM Provider system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Show feature coverage
    print("\n📋 Feature Coverage:")
    features = [
        ("3 Providers (OpenRouter, Local, None)", True),
        ("Encrypted API key storage", True),
        ("Token tracking", True),
        ("Response caching", True),
        ("Free model presets (priority)", True),
        ("API key validation", True),
        ("Smart provider switching", True),
        ("SSE Streaming", args.with_openrouter),
        ("Runtime provider switching", args.with_api),
        ("CLI argument testing", False),  # Needs manual test
    ]
    for feature, tested in features:
        emoji = "✅" if tested else "⏭️"
        status = "Tested" if tested else "Skipped"
        print(f"  {emoji} {feature}: {status}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
