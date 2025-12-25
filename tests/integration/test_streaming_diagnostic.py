"""
Streaming Diagnostic Test
Tests streaming at each layer to identify where issues occur

Run with:
    python tests/integration/test_streaming_diagnostic.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import LOG_DIR, ENABLE_FILE_LOGGING, LOG_VERBOSITY
from utils.logger_utils import get_logger, initialize_logging

initialize_logging(
    enable_file_logging=ENABLE_FILE_LOGGING,
    log_dir=LOG_DIR,
    verbosity_mode=LOG_VERBOSITY
)
logger = get_logger("StreamingDiagnostic")


def test_llm_engine_streaming():
    """Test 1: Direct LLMEngine streaming (lowest level)"""
    print("\n" + "=" * 80)
    print("TEST 1: LLMEngine Direct Streaming")
    print("=" * 80)
    
    from core.generation.llm_engine import LLMEngine
    from config import get_default_config
    
    config = get_default_config()
    engine = LLMEngine(config)
    
    print("Loading LLM model...")
    if not engine.load_model():
        print("❌ Failed to load LLM model")
        return False
    
    print("✅ Model loaded")
    
    # Simple test prompt
    prompt = "Write a short poem about the moon in 4 lines."
    
    print(f"\nPrompt: {prompt}")
    print("\nStreaming output from LLMEngine:")
    print("-" * 40)
    
    token_count = 0
    full_output = ""
    
    try:
        for chunk in engine.generate_stream(prompt, max_new_tokens=100):
            if chunk['success']:
                if not chunk['done']:
                    token = chunk['token']
                    print(token, end='', flush=True)
                    full_output += token
                    token_count += 1
                else:
                    print(f"\n\n----- Done -----")
                    print(f"Tokens: {token_count}")
            else:
                print(f"\n❌ Error: {chunk.get('error')}")
                return False
        
        print("-" * 40)
        
        if token_count > 0:
            print(f"✅ LLMEngine streaming WORKS! ({token_count} tokens)")
            engine.unload_model()
            return True
        else:
            print("❌ No tokens received from LLMEngine")
            engine.unload_model()
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        engine.unload_model()
        return False


def test_generation_engine_streaming():
    """Test 2: GenerationEngine streaming (thinking detection layer)"""
    print("\n" + "=" * 80)
    print("TEST 2: GenerationEngine Streaming (with thinking detection)")
    print("=" * 80)
    
    from core.generation.generation_engine import GenerationEngine
    from config import get_default_config
    
    config = get_default_config()
    engine = GenerationEngine(config)
    
    print("Initializing GenerationEngine...")
    if not engine.initialize():
        print("❌ Failed to initialize GenerationEngine")
        return False
    
    print("✅ GenerationEngine initialized")
    
    # Create mock retrieved results
    mock_results = [{
        'record': {
            'regulation_type': 'Undang-Undang',
            'regulation_number': '13',
            'year': '2003',
            'about': 'Ketenagakerjaan',
            'content': 'Pasal 1: Ketenagakerjaan adalah segala hal yang berhubungan dengan tenaga kerja.'
        },
        'scores': {'final': 0.8, 'semantic': 0.7}
    }]
    
    query = "Jelaskan UU Ketenagakerjaan"
    
    print(f"\nQuery: {query}")
    print("\nStreaming output from GenerationEngine:")
    print("-" * 40)
    
    token_count = 0
    thinking_count = 0
    answer_count = 0
    
    try:
        for chunk in engine.generate_answer(
            query=query,
            retrieved_results=mock_results,
            stream=True,
            thinking_mode='low'
        ):
            chunk_type = chunk.get('type', '')
            
            if chunk_type == 'thinking':
                token = chunk.get('token', '')
                print(f"[T]{token}", end='', flush=True)  # [T] = thinking
                thinking_count += 1
                token_count += 1
                
            elif chunk_type == 'token':
                token = chunk.get('token', '')
                print(token, end='', flush=True)
                answer_count += 1
                token_count += 1
                
            elif chunk_type == 'complete':
                print(f"\n\n----- Complete -----")
                print(f"Answer preview: {chunk.get('answer', '')[:100]}...")
                
            elif chunk_type == 'error':
                print(f"\n❌ Error: {chunk.get('error')}")
                return False
        
        print("-" * 40)
        print(f"Total tokens: {token_count}")
        print(f"  - Thinking: {thinking_count}")
        print(f"  - Answer: {answer_count}")
        
        if token_count > 0:
            print(f"✅ GenerationEngine streaming WORKS!")
            engine.shutdown()
            return True
        else:
            print("❌ No tokens received from GenerationEngine")
            engine.shutdown()
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        engine.shutdown()
        return False


def main():
    """Run all diagnostic tests"""
    print("\n" + "🔍 STREAMING DIAGNOSTIC TESTS".center(80))
    print("=" * 80)
    print("This will test streaming at each layer to identify issues\n")
    
    results = []
    
    # Test 1: LLMEngine (lowest level)
    result1 = test_llm_engine_streaming()
    results.append(("LLMEngine Streaming", result1))
    
    if not result1:
        print("\n⚠️ Stopping early - LLMEngine failed")
        print("Fix LLMEngine before testing higher layers")
        return 1
    
    # Test 2: GenerationEngine (thinking detection)
    result2 = test_generation_engine_streaming()
    results.append(("GenerationEngine Streaming", result2))
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("🎉 All streaming layers working correctly!")
        return 0
    else:
        print("⚠️ Some layers have issues - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
