"""
Performance & Load Testing for Indonesian Legal RAG System

Tests system performance characteristics:
1. Query response time benchmarks by query type
2. Concurrent request handling
3. Memory usage monitoring
4. Throughput testing
5. Component timing breakdowns

Run with:
    python tests/integration/test_performance.py              # Standard performance test
    python tests/integration/test_performance.py --full       # Full benchmark suite
    python tests/integration/test_performance.py --concurrent # Concurrent load test
    python tests/integration/test_performance.py --memory     # Memory profiling

This helps establish performance baselines and identify bottlenecks.
"""

import sys
import os
import time
import threading
import queue
import gc
import traceback
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging


@dataclass
class QueryResult:
    """Stores result of a single query execution"""
    query: str
    query_type: str
    response_time: float
    success: bool
    answer_length: int = 0
    source_count: int = 0
    error: str = ""


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    p50_response_time: float = 0.0
    p90_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput_qps: float = 0.0
    results: List[QueryResult] = field(default_factory=list)


class PerformanceTester:
    """
    Performance and load testing for the RAG system
    """

    # Test queries organized by type for comprehensive coverage
    TEST_QUERIES = {
        'simple': [
            "Apa itu UU Ketenagakerjaan?",
            "Apa itu hak cipta?",
            "Definisi perjanjian kerja",
        ],
        'sanction': [
            "Apa sanksi pelanggaran UU ITE?",
            "Hukuman untuk korupsi menurut UU?",
            "Sanksi pidana pencucian uang",
        ],
        'procedural': [
            "Bagaimana prosedur pendaftaran merek?",
            "Cara mengajukan gugatan perdata?",
            "Proses pendirian PT menurut UU",
        ],
        'complex': [
            "Jelaskan hubungan antara UU Cipta Kerja dan UU Ketenagakerjaan terkait PHK",
            "Analisis perubahan regulasi perlindungan data pribadi di Indonesia",
            "Bandingkan sanksi UU ITE dengan KUHP untuk pencemaran nama baik",
        ]
    }

    def __init__(self, verbose: bool = True):
        initialize_logging()
        self.logger = get_logger("PerformanceTest")
        self.pipeline = None
        self.verbose = verbose
        self.results: List[QueryResult] = []

    def initialize_system(self) -> bool:
        """Initialize the RAG pipeline"""
        self.logger.info("=" * 70)
        self.logger.info("PERFORMANCE TEST - SYSTEM INITIALIZATION")
        self.logger.info("=" * 70)

        try:
            from pipeline import RAGPipeline

            self.logger.info("Creating RAG Pipeline...")
            self.pipeline = RAGPipeline()

            self.logger.info("Initializing components (may take 1-2 minutes)...")
            start_time = time.time()

            if not self.pipeline.initialize():
                self.logger.error("Pipeline initialization failed")
                return False

            elapsed = time.time() - start_time
            self.logger.success(f"System initialized in {elapsed:.1f}s")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            traceback.print_exc()
            return False

    def execute_query(self, query: str, query_type: str = "unknown") -> QueryResult:
        """Execute a single query and measure performance"""
        start_time = time.time()

        try:
            result = self.pipeline.query(query)
            elapsed = time.time() - start_time

            if result and result.get('answer'):
                return QueryResult(
                    query=query,
                    query_type=query_type,
                    response_time=elapsed,
                    success=True,
                    answer_length=len(result['answer']),
                    source_count=len(result.get('sources', []))
                )
            else:
                return QueryResult(
                    query=query,
                    query_type=query_type,
                    response_time=elapsed,
                    success=False,
                    error="No answer generated"
                )

        except Exception as e:
            elapsed = time.time() - start_time
            return QueryResult(
                query=query,
                query_type=query_type,
                response_time=elapsed,
                success=False,
                error=str(e)
            )

    def calculate_percentile(self, times: List[float], percentile: float) -> float:
        """Calculate percentile from a list of times"""
        if not times:
            return 0.0
        sorted_times = sorted(times)
        idx = int(len(sorted_times) * percentile / 100)
        idx = min(idx, len(sorted_times) - 1)
        return sorted_times[idx]

    def compute_metrics(self, results: List[QueryResult]) -> PerformanceMetrics:
        """Compute aggregated metrics from query results"""
        if not results:
            return PerformanceMetrics()

        successful = [r for r in results if r.success]
        times = [r.response_time for r in successful]

        metrics = PerformanceMetrics(
            total_queries=len(results),
            successful_queries=len(successful),
            failed_queries=len(results) - len(successful),
            total_time=sum(r.response_time for r in results),
            results=results
        )

        if times:
            metrics.avg_response_time = sum(times) / len(times)
            metrics.min_response_time = min(times)
            metrics.max_response_time = max(times)
            metrics.p50_response_time = self.calculate_percentile(times, 50)
            metrics.p90_response_time = self.calculate_percentile(times, 90)
            metrics.p99_response_time = self.calculate_percentile(times, 99)

        if metrics.total_time > 0:
            metrics.throughput_qps = metrics.total_queries / metrics.total_time

        return metrics

    def display_metrics(self, metrics: PerformanceMetrics, title: str = "Performance Metrics"):
        """Display performance metrics in a formatted table"""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

        print(f"""
  Queries:
    Total:      {metrics.total_queries}
    Successful: {metrics.successful_queries} ({100*metrics.successful_queries/max(1,metrics.total_queries):.1f}%)
    Failed:     {metrics.failed_queries}

  Response Times:
    Average:    {metrics.avg_response_time:.2f}s
    Min:        {metrics.min_response_time:.2f}s
    Max:        {metrics.max_response_time:.2f}s
    P50:        {metrics.p50_response_time:.2f}s
    P90:        {metrics.p90_response_time:.2f}s
    P99:        {metrics.p99_response_time:.2f}s

  Throughput:
    QPS:        {metrics.throughput_qps:.3f} queries/second
    Total Time: {metrics.total_time:.2f}s
""")

    def run_sequential_benchmark(self, query_types: List[str] = None) -> Dict[str, PerformanceMetrics]:
        """Run sequential benchmark across query types"""
        if query_types is None:
            query_types = list(self.TEST_QUERIES.keys())

        self.logger.info("\n" + "=" * 70)
        self.logger.info("SEQUENTIAL BENCHMARK TEST")
        self.logger.info("=" * 70)

        all_results: Dict[str, List[QueryResult]] = {}

        for qtype in query_types:
            queries = self.TEST_QUERIES.get(qtype, [])
            if not queries:
                continue

            self.logger.info(f"\nTesting {qtype.upper()} queries ({len(queries)} queries)...")
            all_results[qtype] = []

            for i, query in enumerate(queries, 1):
                if self.verbose:
                    print(f"  [{i}/{len(queries)}] {query[:50]}...", end=" ", flush=True)

                result = self.execute_query(query, qtype)
                all_results[qtype].append(result)

                if self.verbose:
                    status = "OK" if result.success else "FAIL"
                    print(f"[{status}] {result.response_time:.2f}s")

        # Compute and display metrics for each type
        metrics_by_type: Dict[str, PerformanceMetrics] = {}
        for qtype, results in all_results.items():
            metrics = self.compute_metrics(results)
            metrics_by_type[qtype] = metrics
            self.display_metrics(metrics, f"{qtype.upper()} Queries")

        # Overall metrics
        all_results_flat = []
        for results in all_results.values():
            all_results_flat.extend(results)
        overall_metrics = self.compute_metrics(all_results_flat)
        self.display_metrics(overall_metrics, "OVERALL PERFORMANCE")

        return metrics_by_type

    def run_concurrent_test(self, num_threads: int = 3, queries_per_thread: int = 2) -> PerformanceMetrics:
        """Run concurrent load test with multiple threads"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"CONCURRENT LOAD TEST ({num_threads} threads, {queries_per_thread} queries each)")
        self.logger.info("=" * 70)

        # Prepare queries for all threads
        all_queries = []
        for qtype, queries in self.TEST_QUERIES.items():
            for q in queries:
                all_queries.append((q, qtype))

        # Cycle through queries if needed
        test_queries = []
        idx = 0
        for _ in range(num_threads * queries_per_thread):
            test_queries.append(all_queries[idx % len(all_queries)])
            idx += 1

        results_queue = queue.Queue()
        start_time = time.time()

        def worker(worker_id: int, queries: List[tuple]):
            """Worker thread that executes queries"""
            for query, qtype in queries:
                result = self.execute_query(query, qtype)
                results_queue.put((worker_id, result))

        # Split queries among workers
        chunks = [[] for _ in range(num_threads)]
        for i, q in enumerate(test_queries):
            chunks[i % num_threads].append(q)

        # Start workers
        threads = []
        for i, chunk in enumerate(chunks):
            t = threading.Thread(target=worker, args=(i, chunk))
            t.start()
            threads.append(t)
            self.logger.info(f"Started worker thread {i} with {len(chunk)} queries")

        # Wait for completion
        for t in threads:
            t.join()

        total_time = time.time() - start_time

        # Collect results
        results = []
        while not results_queue.empty():
            worker_id, result = results_queue.get()
            results.append(result)
            if self.verbose:
                status = "OK" if result.success else "FAIL"
                print(f"  Worker {worker_id}: [{status}] {result.query[:40]}... {result.response_time:.2f}s")

        metrics = self.compute_metrics(results)
        metrics.total_time = total_time  # Use wall clock time for throughput
        metrics.throughput_qps = len(results) / total_time if total_time > 0 else 0

        self.display_metrics(metrics, "CONCURRENT TEST RESULTS")

        return metrics

    def run_memory_profile(self) -> Dict[str, Any]:
        """Profile memory usage during query execution"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("MEMORY USAGE PROFILE")
        self.logger.info("=" * 70)

        try:
            import tracemalloc
            tracemalloc.start()
        except ImportError:
            self.logger.warning("tracemalloc not available, using basic memory tracking")
            tracemalloc = None

        # Get baseline memory
        gc.collect()
        if tracemalloc:
            baseline_snapshot = tracemalloc.take_snapshot()
            baseline_current, baseline_peak = tracemalloc.get_traced_memory()

        memory_samples = []
        results = []

        # Run sample queries
        queries = [
            ("Apa itu UU ITE?", "simple"),
            ("Sanksi pelanggaran hak cipta", "sanction"),
            ("Prosedur pendaftaran merek dagang di Indonesia", "procedural"),
        ]

        for query, qtype in queries:
            gc.collect()

            if tracemalloc:
                before_current, _ = tracemalloc.get_traced_memory()

            result = self.execute_query(query, qtype)
            results.append(result)

            if tracemalloc:
                after_current, after_peak = tracemalloc.get_traced_memory()
                memory_diff = after_current - before_current

                memory_samples.append({
                    'query': query[:40],
                    'memory_delta_kb': memory_diff / 1024,
                    'current_mb': after_current / (1024 * 1024),
                    'peak_mb': after_peak / (1024 * 1024),
                    'response_time': result.response_time
                })

        # Display memory profile
        print(f"""
  Baseline Memory:
    Current:    {baseline_current / (1024*1024):.2f} MB
    Peak:       {baseline_peak / (1024*1024):.2f} MB

  Memory During Queries:
""")
        for sample in memory_samples:
            print(f"    {sample['query']}")
            print(f"      Delta: {sample['memory_delta_kb']:+.1f} KB, Current: {sample['current_mb']:.2f} MB")
            print()

        if tracemalloc:
            final_current, final_peak = tracemalloc.get_traced_memory()
            print(f"""
  Final Memory:
    Current:    {final_current / (1024*1024):.2f} MB
    Peak:       {final_peak / (1024*1024):.2f} MB
    Growth:     {(final_current - baseline_current) / (1024*1024):.2f} MB
""")
            tracemalloc.stop()

        return {
            'samples': memory_samples,
            'results': results
        }

    def run_warmup(self, num_queries: int = 2):
        """Run warmup queries to prime caches"""
        self.logger.info("\nRunning warmup queries...")
        warmup_queries = [
            "Apa itu hukum?",
            "Definisi peraturan"
        ]
        for q in warmup_queries[:num_queries]:
            self.execute_query(q, "warmup")
        self.logger.info("Warmup complete")

    def shutdown(self):
        """Clean up resources"""
        if self.pipeline:
            try:
                self.pipeline.shutdown()
            except Exception:
                pass
        self.logger.info("Performance tester shutdown complete")


def main():
    """Main entry point for performance testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Performance Testing for Legal RAG System')
    parser.add_argument('--full', action='store_true', help='Run full benchmark suite')
    parser.add_argument('--concurrent', action='store_true', help='Run concurrent load test')
    parser.add_argument('--memory', action='store_true', help='Run memory profiling')
    parser.add_argument('--threads', type=int, default=3, help='Number of concurrent threads')
    parser.add_argument('--queries', type=int, default=2, help='Queries per thread')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          INDONESIAN LEGAL RAG - PERFORMANCE TEST SUITE               ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    tester = PerformanceTester(verbose=not args.quiet)

    try:
        # Initialize system
        if not tester.initialize_system():
            print("\n❌ Failed to initialize system")
            return 1

        # Run warmup
        tester.run_warmup()

        # Determine what tests to run
        run_sequential = args.full or not (args.concurrent or args.memory)
        run_concurrent = args.full or args.concurrent
        run_memory = args.full or args.memory

        results_summary = {}

        # Sequential benchmark
        if run_sequential:
            if args.full:
                metrics = tester.run_sequential_benchmark()
            else:
                # Quick test with just simple and sanction queries
                metrics = tester.run_sequential_benchmark(['simple', 'sanction'])
            results_summary['sequential'] = metrics

        # Concurrent test
        if run_concurrent:
            metrics = tester.run_concurrent_test(
                num_threads=args.threads,
                queries_per_thread=args.queries
            )
            results_summary['concurrent'] = metrics

        # Memory profiling
        if run_memory:
            mem_data = tester.run_memory_profile()
            results_summary['memory'] = mem_data

        # Final summary
        print("\n" + "=" * 70)
        print("  PERFORMANCE TEST COMPLETE")
        print("=" * 70)

        if 'sequential' in results_summary:
            seq_metrics = list(results_summary['sequential'].values())
            if seq_metrics:
                total_queries = sum(m.total_queries for m in seq_metrics)
                total_success = sum(m.successful_queries for m in seq_metrics)
                print(f"  Sequential: {total_success}/{total_queries} queries successful")

        if 'concurrent' in results_summary:
            conc = results_summary['concurrent']
            print(f"  Concurrent: {conc.throughput_qps:.3f} QPS with {args.threads} threads")

        print("\n✅ Performance testing completed successfully")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        traceback.print_exc()
        return 1

    finally:
        tester.shutdown()


if __name__ == "__main__":
    sys.exit(main())
