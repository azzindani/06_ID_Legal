"""
Minimal Pipeline Test - Direct Pipeline Testing with Full Cache Management

Tests the core pipeline directly (without HTTP/Gradio) to isolate blocking issues.
Includes aggressive RAM/VRAM clearing and diagnostic output.

Usage:
    python tests/minimal_pipeline_test.py

File: tests/minimal_pipeline_test.py
"""

import sys
import os
import time
import gc
import json
from datetime import datetime
from typing import Dict, List, Any

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    stats = {
        'ram_used_mb': 0,
        'vram_used_mb': 0,
        'vram_total_mb': 0,
        'vram_percent': 0
    }
    
    # RAM stats
    try:
        import psutil
        process = psutil.Process()
        stats['ram_used_mb'] = process.memory_info().rss / 1024 / 1024
    except:
        pass
    
    # VRAM stats
    try:
        import torch
        if torch.cuda.is_available():
            stats['vram_used_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['vram_total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            stats['vram_percent'] = (stats['vram_used_mb'] / stats['vram_total_mb']) * 100 if stats['vram_total_mb'] > 0 else 0
    except:
        pass
    
    return stats


def clear_all_cache(context: str = "") -> Dict[str, Any]:
    """
    Aggressively clear both RAM and VRAM caches.
    Returns memory stats before and after clearing.
    """
    before = get_memory_stats()
    
    # Clear Python garbage
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Also clear any cached allocations
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
    except:
        pass
    
    # Force garbage collection again after CUDA clear
    gc.collect()
    
    after = get_memory_stats()
    
    freed_ram = before['ram_used_mb'] - after['ram_used_mb']
    freed_vram = before['vram_used_mb'] - after['vram_used_mb']
    
    return {
        'context': context,
        'before': before,
        'after': after,
        'freed_ram_mb': freed_ram,
        'freed_vram_mb': freed_vram
    }


# =============================================================================
# LOGGING / DIAGNOSTICS
# =============================================================================

class DiagnosticLogger:
    """Collects diagnostic information for analysis"""
    
    def __init__(self):
        self.start_time = time.time()
        self.events: List[Dict[str, Any]] = []
        self.test_results: List[Dict[str, Any]] = []
        
    def log(self, msg: str, level: str = "INFO", **extra):
        timestamp = time.strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time
        
        event = {
            'timestamp': timestamp,
            'elapsed_s': round(elapsed, 2),
            'level': level,
            'message': msg,
            **extra
        }
        self.events.append(event)
        
        # Console output
        prefix = {"SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(level, "")
        print(f"[{timestamp}] [{level}] {prefix} {msg}", flush=True)
        if extra:
            for k, v in extra.items():
                print(f"    {k}: {v}", flush=True)
    
    def record_test(self, test_name: str, success: bool, elapsed_s: float, 
                    error: str = None, memory: Dict = None, **extra):
        result = {
            'test_name': test_name,
            'success': success,
            'elapsed_s': round(elapsed_s, 2),
            'error': error,
            'memory': memory,
            **extra
        }
        self.test_results.append(result)
        return result
    
    def generate_summary(self) -> str:
        """Generate diagnostic summary for AI analysis"""
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['success'])
        failed = total_tests - passed
        
        summary_lines = [
            "\n" + "=" * 70,
            "DIAGNOSTIC SUMMARY",
            "=" * 70,
            f"Total Tests: {total_tests}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Total Time: {time.time() - self.start_time:.1f}s",
            "",
            "--- TEST RESULTS ---"
        ]
        
        for i, r in enumerate(self.test_results, 1):
            status = "✅ PASS" if r['success'] else "❌ FAIL"
            summary_lines.append(f"[{i}] {r['test_name']}: {status} ({r['elapsed_s']:.2f}s)")
            if r.get('memory'):
                m = r['memory']['after']
                summary_lines.append(f"    Memory: RAM={m['ram_used_mb']:.0f}MB, VRAM={m['vram_used_mb']:.0f}MB ({m['vram_percent']:.1f}%)")
            if r.get('error'):
                summary_lines.append(f"    Error: {r['error'][:200]}")
            if r.get('tokens_generated'):
                summary_lines.append(f"    Tokens: {r['tokens_generated']}")
        
        # Failure analysis
        if failed > 0:
            summary_lines.append("")
            summary_lines.append("--- FAILURE ANALYSIS ---")
            
            # Find pattern
            fail_indices = [i+1 for i, r in enumerate(self.test_results) if not r['success']]
            summary_lines.append(f"Failed at test(s): {fail_indices}")
            
            # Check if always fails at same point
            if len(fail_indices) > 0 and all(f == fail_indices[0] for f in fail_indices):
                summary_lines.append(f"⚠️ PATTERN: Always fails at test #{fail_indices[0]}")
                summary_lines.append("   LIKELY CAUSE: Resource exhaustion or lock deadlock")
            
            # Check memory at failure
            for r in self.test_results:
                if not r['success'] and r.get('memory'):
                    m = r['memory']['after']
                    if m['vram_percent'] > 90:
                        summary_lines.append("⚠️ LIKELY CAUSE: VRAM exhaustion (>90% used)")
        else:
            summary_lines.append("")
            summary_lines.append("--- ALL TESTS PASSED ---")
            summary_lines.append("The pipeline works correctly for multiple sequential queries.")
            summary_lines.append("If blocking occurs with Gradio/API, the issue is in HTTP/UI layer.")
        
        summary_lines.append("=" * 70)
        
        return "\n".join(summary_lines)


# =============================================================================
# TEST CASES
# =============================================================================

# Realistic UI query sequence (simulates user behavior)
UI_TEST_QUERIES = [
    # First query - initial question
    {
        'query': "Apa sanksi pelanggaran UU Ketenagakerjaan?",
        'description': "Initial query about labor law sanctions",
        'expected_action': 'retrieve + generate'
    },
    # Second query - follow-up
    {
        'query': "Bagaimana prosedur PHK menurut hukum Indonesia?",
        'description': "Follow-up about termination procedures",
        'expected_action': 'retrieve + generate'
    },
    # Third query - different topic (simulates topic switch)
    {
        'query': "Apa itu kontrak kerja waktu tertentu?",
        'description': "Topic switch to PKWT contracts",
        'expected_action': 'retrieve + generate'
    },
    # Fourth query - rapid follow-up
    {
        'query': "Jelaskan perbedaan PKWT dan PKWTT",
        'description': "Quick follow-up on contracts",
        'expected_action': 'retrieve + generate'
    },
]


def run_pipeline_tests(logger: DiagnosticLogger):
    """Run full pipeline tests simulating UI usage"""
    
    logger.log("=" * 60)
    logger.log("PIPELINE MULTI-QUERY TEST (Simulating UI Usage)")
    logger.log("=" * 60)
    
    # Step 1: Initialize
    logger.log("Initializing pipeline...")
    cache_result = clear_all_cache("before_init")
    
    from pipeline import RAGPipeline
    pipeline = RAGPipeline()
    
    init_start = time.time()
    if not pipeline.initialize():
        logger.log("Pipeline initialization failed!", "ERROR")
        return False
    
    init_time = time.time() - init_start
    cache_result = clear_all_cache("after_init")
    logger.log(f"Pipeline initialized in {init_time:.1f}s", "SUCCESS", 
               vram_mb=cache_result['after']['vram_used_mb'])
    
    # Step 2: Run test queries
    logger.log("")
    logger.log("--- Running UI Query Simulation ---")
    
    all_passed = True
    
    for i, test_case in enumerate(UI_TEST_QUERIES, 1):
        query = test_case['query']
        desc = test_case['description']
        
        logger.log(f"")
        logger.log(f"[TEST {i}/{len(UI_TEST_QUERIES)}] {desc}")
        logger.log(f"Query: \"{query[:50]}...\"")
        
        # Clear cache before each query
        pre_cache = clear_all_cache(f"before_query_{i}")
        logger.log(f"Pre-query VRAM: {pre_cache['after']['vram_used_mb']:.0f}MB")
        
        start_time = time.time()
        tokens_generated = 0
        first_token_time = None
        error_msg = None
        success = False
        
        try:
            # Test streaming (as UI does)
            for chunk in pipeline.query(query, stream=True, thinking_mode='low'):
                if not isinstance(chunk, dict):
                    continue
                
                chunk_type = chunk.get('type', '')
                
                if chunk_type in ['token', 'thinking']:
                    tokens_generated += 1
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                        logger.log(f"First token at {first_token_time:.2f}s")
                
                elif chunk_type == 'complete':
                    success = True
                    break
                    
                elif chunk_type == 'error':
                    error_msg = chunk.get('error', 'Unknown error')
                    break
            
            if not success and not error_msg:
                error_msg = "Stream ended without 'complete' event"
                
        except Exception as e:
            error_msg = str(e)
            import traceback
            logger.log(f"Exception: {traceback.format_exc()[:500]}", "ERROR")
        
        elapsed = time.time() - start_time
        post_cache = clear_all_cache(f"after_query_{i}")
        
        # Record result
        result = logger.record_test(
            test_name=f"Query_{i}_{desc[:20]}",
            success=success,
            elapsed_s=elapsed,
            error=error_msg,
            memory=post_cache,
            tokens_generated=tokens_generated,
            first_token_s=first_token_time
        )
        
        if success:
            logger.log(f"PASSED: {tokens_generated} tokens in {elapsed:.2f}s", "SUCCESS")
        else:
            logger.log(f"FAILED: {error_msg}", "ERROR")
            all_passed = False
        
        # Small delay between queries (realistic)
        time.sleep(0.5)
    
    # Cleanup
    logger.log("")
    logger.log("--- Cleanup ---")
    pipeline.shutdown()
    final_cache = clear_all_cache("final_cleanup")
    logger.log(f"Final VRAM: {final_cache['after']['vram_used_mb']:.0f}MB")
    
    return all_passed


def run_llm_engine_tests(logger: DiagnosticLogger):
    """Test LLM Engine directly (lowest level)"""
    
    logger.log("")
    logger.log("=" * 60)
    logger.log("LLM ENGINE DIRECT TEST (Lowest Level)")
    logger.log("=" * 60)
    
    from core.generation.llm_engine import LLMEngine
    from config import get_default_config
    
    cache_result = clear_all_cache("before_llm_load")
    
    config = get_default_config()
    engine = LLMEngine(config)
    
    logger.log("Loading LLM model...")
    load_start = time.time()
    if not engine.load_model():
        logger.log("Failed to load model!", "ERROR")
        logger.record_test("LLM_Load", False, time.time() - load_start, "Model load failed")
        return False
    
    load_time = time.time() - load_start
    cache_result = clear_all_cache("after_llm_load")
    logger.log(f"Model loaded in {load_time:.1f}s", "SUCCESS",
               vram_mb=cache_result['after']['vram_used_mb'])
    
    # Test prompts
    test_prompts = [
        "Jelaskan tentang UU Ketenagakerjaan Indonesia.",
        "Apa saja hak-hak pekerja menurut hukum?",
        "Bagaimana cara mengajukan gugatan ke pengadilan?",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.log(f"")
        logger.log(f"[LLM TEST {i}] Prompt: \"{prompt[:40]}...\"")
        
        pre_cache = clear_all_cache(f"before_llm_gen_{i}")
        
        start_time = time.time()
        tokens = 0
        first_token_time = None
        error_msg = None
        success = False
        
        try:
            for chunk in engine.generate_stream(prompt, max_new_tokens=100):
                if chunk.get('done'):
                    success = True
                    break
                if chunk.get('success') and chunk.get('token'):
                    tokens += 1
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                if chunk.get('error'):
                    error_msg = chunk.get('error')
                    break
                    
        except Exception as e:
            error_msg = str(e)
        
        elapsed = time.time() - start_time
        post_cache = clear_all_cache(f"after_llm_gen_{i}")
        
        logger.record_test(
            test_name=f"LLM_Gen_{i}",
            success=success,
            elapsed_s=elapsed,
            error=error_msg,
            memory=post_cache,
            tokens_generated=tokens
        )
        
        if success:
            logger.log(f"PASSED: {tokens} tokens in {elapsed:.2f}s", "SUCCESS")
        else:
            logger.log(f"FAILED: {error_msg}", "ERROR")
        
        time.sleep(0.5)
    
    engine.unload_model()
    clear_all_cache("after_llm_unload")
    
    return all(r['success'] for r in logger.test_results if r['test_name'].startswith('LLM_'))


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = DiagnosticLogger()
    
    logger.log("=" * 70)
    logger.log("MULTI-TURN PIPELINE DIAGNOSTIC TEST")
    logger.log(f"Started: {datetime.now().isoformat()}")
    logger.log("=" * 70)
    
    initial_memory = get_memory_stats()
    logger.log(f"Initial Memory: RAM={initial_memory['ram_used_mb']:.0f}MB, VRAM={initial_memory['vram_used_mb']:.0f}MB")
    
    try:
        # Test 1: LLM Engine (lowest level)
        llm_passed = run_llm_engine_tests(logger)
        
        if llm_passed:
            # Clear everything before pipeline test
            clear_all_cache("between_tests")
            time.sleep(2)
            
            # Test 2: Full pipeline
            run_pipeline_tests(logger)
        else:
            logger.log("Skipping pipeline tests since LLM tests failed", "WARN")
            
    except KeyboardInterrupt:
        logger.log("Tests interrupted by user", "WARN")
    except Exception as e:
        logger.log(f"Test suite crashed: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc()[:1000], "ERROR")
    
    # Print diagnostic summary
    summary = logger.generate_summary()
    print(summary)
    
    # Save results to file
    try:
        output_file = os.path.join(os.path.dirname(__file__), "test_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'events': logger.events,
                'test_results': logger.test_results,
                'summary': summary
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    except:
        pass


if __name__ == "__main__":
    main()
