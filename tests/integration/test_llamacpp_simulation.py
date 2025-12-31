"""
LlamaCpp Provider - Simulation Test

End-to-end test of the LlamaCpp provider system.
Tests model loading, generation, streaming, and valve switching.

Usage:
    # Basic test (CPU mode)
    python tests/integration/test_llamacpp_simulation.py
    
    # GPU test
    python tests/integration/test_llamacpp_simulation.py --test-gpu
    
    # Valve switching test
    python tests/integration/test_llamacpp_simulation.py --test-valve
    
    # Full test
    python tests/integration/test_llamacpp_simulation.py --full

File: tests/integration/test_llamacpp_simulation.py
"""

import sys
import os
import time
import json
import argparse

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


def test_llamacpp_import():
    """Test that LlamaCppProvider can be imported"""
    print_header("TEST 1: LlamaCppProvider Import")
    
    try:
        from core.llm_providers import LlamaCppProvider
        print(f"Import successful: {LlamaCppProvider}")
        
        # Check it inherits from base
        from core.llm_providers import LLMProviderBase
        is_subclass = issubclass(LlamaCppProvider, LLMProviderBase)
        print(f"Is LLMProviderBase subclass: {is_subclass}")
        
        print_result("Import", True, "LlamaCppProvider imported successfully")
        return True
        
    except ImportError as e:
        print_result("Import", False, str(e))
        return False


def test_llamacpp_init():
    """Test LlamaCppProvider initialization (without loading model)"""
    print_header("TEST 2: LlamaCppProvider Initialization")
    
    try:
        from core.llm_providers import LlamaCppProvider
        
        # Initialize with defaults
        provider = LlamaCppProvider()
        
        print(f"Provider Name: {provider.provider_name}")
        print(f"Model Name: {provider.model_name}")
        print(f"Is Available: {provider.is_available()}")
        print(f"Context Window: {provider.get_context_window()}")
        
        # Get info
        info = provider.get_info()
        print(f"\n[Provider Info]")
        print(json.dumps(info, indent=2))
        
        print_result("Initialization", True, "Provider initialized without loading model")
        return True
        
    except Exception as e:
        print_result("Initialization", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_llamacpp_config():
    """Test LlamaCpp configuration from config.py"""
    print_header("TEST 3: LlamaCpp Configuration")
    
    try:
        from config import (
            LLAMACPP_REPO_ID,
            LLAMACPP_FILENAME,
            LLAMACPP_N_CTX,
            LLAMACPP_N_GPU_LAYERS,
            LLAMACPP_N_THREADS,
            LLAMACPP_USE_MMAP,
            LLAMACPP_FLASH_ATTN,
        )
        
        print(f"[Config Values]")
        print(f"  LLAMACPP_REPO_ID: {LLAMACPP_REPO_ID}")
        print(f"  LLAMACPP_FILENAME: {LLAMACPP_FILENAME}")
        print(f"  LLAMACPP_N_CTX: {LLAMACPP_N_CTX}")
        print(f"  LLAMACPP_N_GPU_LAYERS: {LLAMACPP_N_GPU_LAYERS}")
        print(f"  LLAMACPP_N_THREADS: {LLAMACPP_N_THREADS}")
        print(f"  LLAMACPP_USE_MMAP: {LLAMACPP_USE_MMAP}")
        print(f"  LLAMACPP_FLASH_ATTN: {LLAMACPP_FLASH_ATTN}")
        
        print_result("Configuration", True, "All config values loaded")
        return True
        
    except ImportError as e:
        print_result("Configuration", False, f"Config import failed: {e}")
        return False


def test_llamacpp_factory():
    """Test LlamaCppProvider in factory"""
    print_header("TEST 4: Factory Registration")
    
    try:
        from core.llm_providers import LLMProviderFactory
        
        # Check if llamacpp is in registry
        providers = LLMProviderFactory.list_providers()
        print(f"[Available Providers]")
        for name, meta in providers.items():
            print(f"  {name}: {meta['description']}")
        
        has_llamacpp = "llamacpp" in providers
        print(f"\nLlamaCpp in registry: {has_llamacpp}")
        
        if has_llamacpp:
            print(f"LlamaCpp metadata: {providers['llamacpp']}")
        
        print_result("Factory", has_llamacpp, "LlamaCpp registered in factory")
        return has_llamacpp
        
    except Exception as e:
        print_result("Factory", False, str(e))
        return False


def test_llamacpp_model_load(n_gpu_layers: int = 0):
    """Test model loading (downloads if needed)"""
    print_header(f"TEST 5: Model Loading (n_gpu_layers={n_gpu_layers})")
    
    try:
        from core.llm_providers import LlamaCppProvider
        
        print(f"Initializing LlamaCppProvider...")
        provider = LlamaCppProvider(n_gpu_layers=n_gpu_layers)
        
        print(f"Loading model (this may download ~4GB on first run)...")
        start_time = time.time()
        success = provider.load_model()
        load_time = time.time() - start_time
        
        if success:
            print(f"Model loaded in {load_time:.2f}s")
            print(f"Available: {provider.is_available()}")
            
            info = provider.get_info()
            print(f"Model path: {info.get('model_path', 'N/A')}")
        else:
            print("Model load failed")
        
        print_result("Model Load", success, f"Loaded in {load_time:.2f}s" if success else "Failed")
        return success, provider
        
    except Exception as e:
        print_result("Model Load", False, str(e))
        import traceback
        traceback.print_exc()
        return False, None


def test_llamacpp_generate(provider):
    """Test synchronous generation"""
    print_header("TEST 6: Synchronous Generation")
    
    if provider is None or not provider.is_available():
        print_result("Generate", False, "Provider not available")
        return False
    
    try:
        prompt = "Apa itu hukum pidana? Jelaskan dalam 2 kalimat."
        print(f"Prompt: {prompt[:50]}...")
        
        print("\n[Generating...]")
        start_time = time.time()
        result = provider.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        elapsed = time.time() - start_time
        
        print(f"\n[Result]")
        print(f"Success: {result['success']}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Tokens: {result.get('tokens_generated', 'N/A')}")
        print(f"Tokens/s: {result.get('tokens_per_second', 0):.2f}")
        print(f"\nResponse:\n{result['generated_text'][:300]}...")
        
        print_result("Generate", result['success'], f"{result.get('tokens_generated', 0)} tokens in {elapsed:.2f}s")
        return result['success']
        
    except Exception as e:
        print_result("Generate", False, str(e))
        return False


def test_llamacpp_stream(provider):
    """Test streaming generation"""
    print_header("TEST 7: Streaming Generation")
    
    if provider is None or not provider.is_available():
        print_result("Stream", False, "Provider not available")
        return False
    
    try:
        prompt = "Apa itu hukum perdata? Jawab singkat."
        print(f"Prompt: {prompt}")
        
        print("\n[Streaming...]")
        start_time = time.time()
        full_text = ""
        token_count = 0
        
        for chunk in provider.generate_stream(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.7
        ):
            if chunk['success'] and not chunk['done']:
                token = chunk.get('token', '')
                full_text += token
                token_count = chunk.get('tokens_generated', token_count)
                print(token, end='', flush=True)
        
        elapsed = time.time() - start_time
        
        print(f"\n\n[Streaming Stats]")
        print(f"Total Tokens: {token_count}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Tokens/s: {token_count/elapsed:.2f}" if elapsed > 0 else "N/A")
        
        print_result("Stream", True, f"{token_count} tokens streamed")
        return True
        
    except Exception as e:
        print_result("Stream", False, str(e))
        return False


def test_valve_switching():
    """Test valve switching between providers"""
    print_header("TEST 8: Valve Switching (local ↔ llamacpp ↔ openrouter)")
    
    try:
        from core.llm_providers import LLMProviderFactory
        
        # Start with none
        print("[1. Switch to 'none' provider]")
        LLMProviderFactory.shutdown()
        provider = LLMProviderFactory.get_provider("none")
        print(f"Current: {provider.provider_name}")
        
        # Switch to llamacpp (don't auto-load to save time)
        print("\n[2. Switch to 'llamacpp' provider]")
        provider = LLMProviderFactory.get_provider("llamacpp", auto_load=False)
        print(f"Current: {provider.provider_name}")
        print(f"Available: {provider.is_available()}")
        
        # Switch back to none
        print("\n[3. Switch back to 'none' provider]")
        provider = LLMProviderFactory.get_provider("none")
        print(f"Current: {provider.provider_name}")
        
        # Cleanup
        LLMProviderFactory.shutdown()
        print("\n[4. Shutdown complete]")
        
        print_result("Valve Switching", True, "Provider switching works")
        return True
        
    except Exception as e:
        print_result("Valve Switching", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_cpu_mode():
    """Test CPU-only inference"""
    print_header("TEST 9: CPU-Only Mode (n_gpu_layers=0)")
    
    success, provider = test_llamacpp_model_load(n_gpu_layers=0)
    
    if success and provider:
        test_llamacpp_generate(provider)
        provider.unload_model()
        return True
    
    return False


def test_gpu_mode():
    """Test GPU inference"""
    print_header("TEST 10: GPU Mode (n_gpu_layers=-1)")
    
    success, provider = test_llamacpp_model_load(n_gpu_layers=-1)
    
    if success and provider:
        test_llamacpp_generate(provider)
        provider.unload_model()
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="LlamaCpp Provider - Simulation Test")
    parser.add_argument("--test-gpu", action="store_true", help="Test GPU mode")
    parser.add_argument("--test-valve", action="store_true", help="Test valve switching")
    parser.add_argument("--test-generate", action="store_true", help="Test generation (requires model)")
    parser.add_argument("--full", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  🧪 LlamaCpp Provider - Simulation Test")
    print("="*60)
    print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    
    results = []
    
    # =========================================================================
    # PART 1: Basic Tests (always run)
    # =========================================================================
    print("\n" + "-"*60)
    print("  PART 1: Basic Tests")
    print("-"*60)
    
    results.append(("Import", test_llamacpp_import()))
    results.append(("Initialization", test_llamacpp_init()))
    results.append(("Configuration", test_llamacpp_config()))
    results.append(("Factory", test_llamacpp_factory()))
    
    # =========================================================================
    # PART 2: Valve Switching (optional)
    # =========================================================================
    if args.test_valve or args.full:
        print("\n" + "-"*60)
        print("  PART 2: Valve Switching")
        print("-"*60)
        
        results.append(("Valve Switching", test_valve_switching()))
    
    # =========================================================================
    # PART 3: Generation Tests (optional, requires model download)
    # =========================================================================
    if args.test_generate or args.full:
        print("\n" + "-"*60)
        print("  PART 3: Generation Tests")
        print("-"*60)
        
        success, provider = test_llamacpp_model_load(n_gpu_layers=-1)
        results.append(("Model Load", success))
        
        if success and provider:
            results.append(("Generate", test_llamacpp_generate(provider)))
            results.append(("Stream", test_llamacpp_stream(provider)))
            provider.unload_model()
    
    # =========================================================================
    # PART 4: GPU-specific Tests (optional)
    # =========================================================================
    if args.test_gpu and not args.full:
        print("\n" + "-"*60)
        print("  PART 4: GPU Mode Test")
        print("-"*60)
        
        results.append(("GPU Mode", test_gpu_mode()))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("TEST SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        emoji = "✅" if success else "❌"
        print(f"  {emoji} {name}")
    
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
