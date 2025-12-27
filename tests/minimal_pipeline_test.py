"""
Minimal API Test - Direct Pipeline Testing

This bypasses Gradio/HTTP completely to test if the pipeline itself works
for multiple sequential requests.

Usage:
    python tests/minimal_pipeline_test.py

File: tests/minimal_pipeline_test.py
"""

import sys
import os
import time
import gc

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(msg, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def clear_gpu():
    """Aggressively clear GPU memory"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB allocated")
    except:
        pass

def test_pipeline_multi_query():
    """Test pipeline with multiple sequential queries"""
    log("=" * 60)
    log("MINIMAL PIPELINE TEST (No HTTP, No Gradio)")
    log("=" * 60)
    
    # Step 1: Initialize pipeline
    log("\n--- STEP 1: Initialize Pipeline ---")
    from pipeline import RAGPipeline
    
    pipeline = RAGPipeline()
    log("RAGPipeline instance created")
    
    if not pipeline.initialize():
        log("❌ Failed to initialize pipeline", "ERROR")
        return False
    log("✅ Pipeline initialized")
    clear_gpu()
    
    # Test queries
    queries = [
        "Apa sanksi pelanggaran UU Ketenagakerjaan?",
        "Bagaimana prosedur PHK?",
        "Apa hak pekerja yang di-PHK?",
    ]
    
    # Step 2: Test retrieve_documents (no LLM)
    log("\n--- STEP 2: Test retrieve_documents (3x) ---")
    for i, query in enumerate(queries, 1):
        log(f"[{i}] Retrieving: '{query[:40]}...'")
        start = time.time()
        try:
            result = pipeline.retrieve_documents(query, top_k=3)
            elapsed = time.time() - start
            sources = result.get('sources', [])
            log(f"[{i}] ✅ Retrieved {len(sources)} docs in {elapsed:.2f}s", "SUCCESS")
        except Exception as e:
            log(f"[{i}] ❌ Retrieve failed: {e}", "ERROR")
            return False
        clear_gpu()
    
    # Step 3: Test full query with streaming (LLM generation)
    log("\n--- STEP 3: Test query with streaming (3x) ---")
    for i, query in enumerate(queries, 1):
        log(f"[{i}] Query (streaming): '{query[:40]}...'")
        start = time.time()
        chunks_received = 0
        final_answer = ""
        
        try:
            for chunk in pipeline.query(query, stream=True, thinking_mode='low'):
                if not isinstance(chunk, dict):
                    continue
                    
                chunk_type = chunk.get('type', '')
                if chunk_type in ['token', 'thinking']:
                    chunks_received += 1
                    if chunks_received == 1:
                        log(f"[{i}] 📡 First token at {time.time()-start:.2f}s")
                elif chunk_type == 'complete':
                    final_answer = chunk.get('answer', '')[:100]
                    break
            
            elapsed = time.time() - start
            log(f"[{i}] ✅ Generated in {elapsed:.2f}s, {chunks_received} tokens", "SUCCESS")
            log(f"[{i}] Answer preview: {final_answer}...")
                
        except Exception as e:
            log(f"[{i}] ❌ Query failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
        
        # Critical: Clear GPU between queries
        log(f"[{i}] Clearing GPU...")
        clear_gpu()
        time.sleep(1)  # Small delay
    
    # Step 4: Test non-streaming query
    log("\n--- STEP 4: Test query without streaming (2x) ---")
    for i in range(2):
        query = f"Cari informasi tentang kontrak kerja - test {i+1}"
        log(f"[{i+1}] Query (non-stream): '{query[:40]}...'")
        start = time.time()
        
        try:
            result = pipeline.query(query, stream=False, thinking_mode='low')
            elapsed = time.time() - start
            
            if result.get('success'):
                answer = result.get('answer', '')[:100]
                log(f"[{i+1}] ✅ Generated in {elapsed:.2f}s", "SUCCESS")
                log(f"[{i+1}] Answer preview: {answer}...")
            else:
                log(f"[{i+1}] ❌ Query failed: {result.get('error')}", "ERROR")
                return False
                
        except Exception as e:
            log(f"[{i+1}] ❌ Exception: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
        
        clear_gpu()
        time.sleep(1)
    
    # Cleanup
    log("\n--- CLEANUP ---")
    pipeline.shutdown()
    clear_gpu()
    
    log("\n" + "=" * 60)
    log("✅ ALL TESTS PASSED - Pipeline works for multiple queries!", "SUCCESS")
    log("=" * 60)
    return True


def test_llm_engine_direct():
    """Test LLM Engine directly for multiple generations"""
    log("\n" + "=" * 60)
    log("LLM ENGINE DIRECT TEST (Lowest level)")
    log("=" * 60)
    
    from core.generation.llm_engine import LLMEngine
    from config import get_default_config
    
    config = get_default_config()
    engine = LLMEngine(config)
    
    log("Loading model...")
    if not engine.load_model():
        log("❌ Failed to load model", "ERROR")
        return False
    log("✅ Model loaded")
    clear_gpu()
    
    prompts = [
        "Jelaskan tentang hukum ketenagakerjaan di Indonesia.",
        "Apa itu kontrak kerja waktu tertentu (PKWT)?",
        "Bagaimana prosedur pemutusan hubungan kerja yang sah?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        log(f"\n[{i}] Testing generate_stream: '{prompt[:40]}...'")
        start = time.time()
        tokens = 0
        
        try:
            for chunk in engine.generate_stream(prompt, max_new_tokens=100):
                if chunk.get('done'):
                    break
                if chunk.get('success') and chunk.get('token'):
                    tokens += 1
                    if tokens == 1:
                        log(f"[{i}] 📡 First token at {time.time()-start:.2f}s")
            
            elapsed = time.time() - start
            log(f"[{i}] ✅ Generated {tokens} tokens in {elapsed:.2f}s", "SUCCESS")
            
        except Exception as e:
            log(f"[{i}] ❌ Generation failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
        
        clear_gpu()
        time.sleep(1)
    
    engine.unload_model()
    clear_gpu()
    
    log("\n✅ LLM ENGINE TESTS PASSED", "SUCCESS")
    return True


if __name__ == "__main__":
    # Run lowest-level test first
    log("Starting tests...\n")
    
    try:
        # Test 1: LLM Engine (lowest level)
        if test_llm_engine_direct():
            clear_gpu()
            time.sleep(2)
            
            # Test 2: Full pipeline
            test_pipeline_multi_query()
        else:
            log("LLM Engine test failed, skipping pipeline test", "ERROR")
            
    except KeyboardInterrupt:
        log("\nTests interrupted", "WARN")
    except Exception as e:
        log(f"Test suite failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
