#!/usr/bin/env python3
"""
Performance Benchmarks Script

Measures search and generation performance.
"""

import sys
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
    """Benchmark generation performance"""
    logger.info("Running generation benchmark...")

    times = []
    tokens = []

    for query in queries:
        start = time.time()
        result = pipeline.query(query, stream=False)
        elapsed = time.time() - start
        times.append(elapsed)

        # Estimate tokens
        answer_tokens = len(result.get('answer', '').split())
        tokens.append(answer_tokens)

        logger.info(f"Query: {query[:50]}... Time: {elapsed:.2f}s, Tokens: ~{answer_tokens}")

    return {
        "total_queries": len(queries),
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_tokens": sum(tokens),
        "avg_tokens": sum(tokens) / len(tokens),
        "tokens_per_second": sum(tokens) / sum(times) if sum(times) > 0 else 0,
    }


def main():
    initialize_logging()
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
