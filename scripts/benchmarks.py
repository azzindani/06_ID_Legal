#!/usr/bin/env python3
"""
Performance Benchmarks Script

Measures search and generation performance.
"""

import sys
from config import LOG_DIR, ENABLE_FILE_LOGGING, LOG_VERBOSITY
import os
import time
import json
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_utils import get_logger, initialize_logging

logger = get_logger("Benchmarks")


def benchmark_queries() -> List[str]:
    """Sample queries for benchmarking"""
    return [
        "Apa sanksi pelanggaran UU Ketenagakerjaan?",
        "Bagaimana prosedur PHK menurut hukum?",
        "Apa definisi konsumen menurut UU?",
        "Apa hak karyawan kontrak?",
        "Bagaimana cara mendirikan PT?",
    ]


def run_search_benchmark(pipeline, queries: List[str]) -> Dict:
    """Benchmark search performance"""
    logger.info("Running search benchmark...")

    times = []
    for query in queries:
        start = time.time()
        # Just run search phase
        pipeline.query(query, stream=False)
        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"Query: {query[:50]}... Time: {elapsed:.2f}s")

    return {
        "total_queries": len(queries),
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


def run_generation_benchmark(pipeline, queries: List[str]) -> Dict:
    """Benchmark generation performance with detailed metrics"""
    logger.info("Running generation benchmark...")
    
    import psutil
    import statistics

    times = []
    tokens = []
    memory_samples = []
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

    for query in queries:
        # Sample memory before query
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        start = time.time()
        result = pipeline.query(query, stream=False)
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Sample memory after query
        mem_after = process.memory_info().rss / (1024 * 1024)
        memory_samples.append(mem_after - mem_before)

        # Estimate tokens
        answer_tokens = len(result.get('answer', '').split())
        tokens.append(answer_tokens)

        logger.info(f"Query: {query[:50]}... Time: {elapsed:.2f}s, Tokens: ~{answer_tokens}, Memory delta: {mem_after - mem_before:.1f}MB")
    
    # Calculate percentiles
    sorted_times = sorted(times)
    n = len(sorted_times)
    
    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f < len(data) - 1 else f
        return data[f] + (k - f) * (data[c] - data[f]) if f != c else data[f]
    
    p50 = percentile(sorted_times, 50)
    p95 = percentile(sorted_times, 95)
    p99 = percentile(sorted_times, 99)
    
    # Final memory
    final_memory = process.memory_info().rss / (1024 * 1024)

    return {
        "total_queries": len(queries),
        "total_time": sum(times),
        "avg_time": statistics.mean(times),
        "min_time": min(times),
        "max_time": max(times),
        "p50_latency": p50,
        "p95_latency": p95,
        "p99_latency": p99,
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
        "total_tokens": sum(tokens),
        "avg_tokens": statistics.mean(tokens),
        "tokens_per_second": sum(tokens) / sum(times) if sum(times) > 0 else 0,
        "queries_per_minute": 60 / statistics.mean(times) if times else 0,
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "memory_growth_mb": final_memory - initial_memory,
        "avg_memory_delta_mb": statistics.mean(memory_samples) if memory_samples else 0,
    }


def main():
    initialize_logging(
        enable_file_logging=ENABLE_FILE_LOGGING,
        log_dir=LOG_DIR,
        verbosity_mode=LOG_VERBOSITY
    )
    logger.info("=" * 60)
    logger.info("Performance Benchmarks")
    logger.info("=" * 60)

    from pipeline import RAGPipeline

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = RAGPipeline()
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return 1

    queries = benchmark_queries()

    # Run benchmarks
    results = {
        "generation": run_generation_benchmark(pipeline, queries)
    }

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for name, data in results.items():
        print(f"\n{name.upper()}:")
        for key, value in data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    # Save results
    output_path = "benchmark_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    pipeline.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
