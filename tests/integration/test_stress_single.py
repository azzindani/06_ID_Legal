"""
Stress Test - Single Query with Maximum Settings

This test runs a single comprehensive legal query with ALL settings maxed out:
- ALL 5 search phases enabled (including expert_review)
- Maximum candidates per phase (1000)
- Maximum research team size (5 personas)
- Maximum final_top_k (20 documents)
- Maximum max_new_tokens (8192)
- All validation features enabled

Purpose:
- Verify system stability under maximum load
- Measure peak resource usage (memory, time)
- Identify bottlenecks in the pipeline
- Validate that maxed settings don't cause crashes

Run with:
    python tests/integration/test_stress_single.py

Options:
    --quick      Use moderate settings instead of maximum
    --verbose    Show detailed output during processing
    --memory     Enable detailed memory profiling
"""

import sys
import os
import time
import tracemalloc
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging


# Maximum stress test configuration
STRESS_CONFIG_MAX = {
    'final_top_k': 20,                    # Maximum documents to return
    'max_rounds': 5,                       # Maximum search rounds
    'research_team_size': 5,               # All 5 personas
    'max_new_tokens': 8192,                # Maximum generation tokens
    'temperature': 0.7,
    'top_p': 1.0,
    'top_k': 100,                          # Maximum top_k
    'min_p': 0.05,
    'enable_cross_validation': True,
    'enable_devil_advocate': True,
    'consensus_threshold': 0.5,
    'parallel_research': True,
}

# Maximum search phases - all enabled with high candidates
STRESS_SEARCH_PHASES = {
    'initial_scan': {
        'candidates': 800,                 # High but not 1000 to avoid OOM
        'semantic_threshold': 0.15,        # Lower threshold = more results
        'keyword_threshold': 0.04,
        'description': 'Maximum broad scan',
        'time_limit': 120,
        'focus_areas': ['regulation_type', 'enacting_body'],
        'enabled': True
    },
    'focused_review': {
        'candidates': 400,
        'semantic_threshold': 0.25,
        'keyword_threshold': 0.08,
        'description': 'Maximum focused review',
        'time_limit': 120,
        'focus_areas': ['content', 'chapter', 'article'],
        'enabled': True
    },
    'deep_analysis': {
        'candidates': 200,
        'semantic_threshold': 0.35,
        'keyword_threshold': 0.12,
        'description': 'Maximum deep analysis',
        'time_limit': 120,
        'focus_areas': ['kg_entities', 'cross_references'],
        'enabled': True
    },
    'verification': {
        'candidates': 100,
        'semantic_threshold': 0.45,
        'keyword_threshold': 0.16,
        'description': 'Maximum verification',
        'time_limit': 90,
        'focus_areas': ['authority_score', 'temporal_score'],
        'enabled': True
    },
    'expert_review': {
        'candidates': 80,
        'semantic_threshold': 0.40,
        'keyword_threshold': 0.14,
        'description': 'Expert review phase - ENABLED for stress test',
        'time_limit': 90,
        'focus_areas': ['legal_richness', 'completeness_score'],
        'enabled': True                    # Enable for stress test
    }
}

# Moderate settings for quick mode
MODERATE_CONFIG = {
    'final_top_k': 10,
    'max_rounds': 3,
    'research_team_size': 3,
    'max_new_tokens': 4096,
    'temperature': 0.7,
    'top_p': 1.0,
    'top_k': 50,
    'enable_cross_validation': True,
    'enable_devil_advocate': False,
    'consensus_threshold': 0.6,
}


class StressTester:
    """Single query stress test with maximum settings"""

    def __init__(self, quick_mode: bool = False, verbose: bool = False, memory_profile: bool = False):
        initialize_logging()
        self.logger = get_logger("StressTest")
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.memory_profile = memory_profile

        # Select config based on mode
        self.config = MODERATE_CONFIG.copy() if quick_mode else STRESS_CONFIG_MAX.copy()
        if not quick_mode:
            self.config['search_phases'] = STRESS_SEARCH_PHASES

        self.pipeline = None
        self.results: Dict[str, Any] = {}

    def print_header(self):
        """Print test header"""
        mode = "QUICK MODE (Moderate Settings)" if self.quick_mode else "MAXIMUM STRESS MODE"
        print("\n" + "=" * 100)
        print(f"STRESS TEST - SINGLE QUERY - {mode}")
        print("=" * 100)
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        print("Configuration:")
        print("-" * 50)
        for key, value in self.config.items():
            if key != 'search_phases':
                print(f"  {key}: {value}")

        if 'search_phases' in self.config:
            print("\nSearch Phases:")
            for phase, settings in self.config['search_phases'].items():
                status = "ENABLED" if settings.get('enabled', False) else "disabled"
                print(f"  {phase}: {settings.get('candidates', 0)} candidates ({status})")
        print()

    def initialize(self) -> bool:
        """Initialize pipeline with stress config"""
        self.logger.info("Initializing pipeline with stress configuration...")
        print("Initializing RAG Pipeline...")

        try:
            from pipeline import RAGPipeline

            self.pipeline = RAGPipeline(config=self.config)
            success = self.pipeline.initialize()

            if success:
                print("Pipeline initialized successfully")
                self.logger.success("Pipeline ready for stress test")
            else:
                print("Pipeline initialization FAILED")
                self.logger.error("Pipeline initialization failed")

            return success

        except Exception as e:
            print(f"Initialization error: {e}")
            self.logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_stress_query(self) -> Dict[str, Any]:
        """Run a single complex query with maximum settings"""

        # Complex query that should trigger multiple personas and phases
        query = """
        Jelaskan secara lengkap dan komprehensif tentang:
        1. Prosedur pengajuan keberatan pajak menurut UU KUP beserta persyaratan dan jangka waktunya
        2. Sanksi administratif dan pidana yang dapat dikenakan jika terlambat mengajukan keberatan
        3. Hubungan antara keberatan pajak dengan banding di Pengadilan Pajak
        4. Hak-hak wajib pajak selama proses keberatan berlangsung
        5. Contoh kasus dan yurisprudensi terkait keberatan pajak
        """

        print("\n" + "=" * 100)
        print("EXECUTING STRESS QUERY")
        print("=" * 100)
        print(f"\nQuery ({len(query)} chars):")
        print("-" * 80)
        print(query.strip())
        print("-" * 80)

        # Start memory tracking if enabled
        if self.memory_profile:
            tracemalloc.start()
            print("\nMemory profiling enabled")

        # Execute query with timing
        start_time = time.time()

        try:
            # Use streaming for real-time output
            print("\n" + "=" * 100)
            print("GENERATING ANSWER (Streaming)")
            print("=" * 100)
            print()

            full_answer = ""
            chunk_count = 0
            stream_start = time.time()
            result = None

            # Use query() with stream=True
            for chunk in self.pipeline.query(query, stream=True):
                if chunk.get('type') == 'token':
                    token = chunk.get('token', '')
                    print(token, end='', flush=True)
                    full_answer += token
                    chunk_count += 1
                elif chunk.get('type') == 'complete':
                    result = chunk
                    break

            stream_time = time.time() - stream_start
            print(f"\n\n[Streamed: {chunk_count} tokens in {stream_time:.2f}s]")

            total_time = time.time() - start_time

            # Get memory stats
            memory_stats = {}
            if self.memory_profile:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_stats = {
                    'current_mb': current / (1024 * 1024),
                    'peak_mb': peak / (1024 * 1024)
                }
                print(f"\nMemory: Current={memory_stats['current_mb']:.2f}MB, Peak={memory_stats['peak_mb']:.2f}MB")

            # Compile results
            self.results = {
                'success': True,
                'query': query,
                'answer': full_answer,
                'answer_length': len(full_answer),
                'chunk_count': chunk_count,
                'total_time': total_time,
                'stream_time': stream_time,
                'memory': memory_stats,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            # Extract additional metadata if available
            if result:
                self.results['sources'] = result.get('sources', [])
                self.results['citations'] = result.get('citations', [])
                self.results['thinking'] = result.get('thinking', '')
                self.results['phase_metadata'] = result.get('phase_metadata', {})
                self.results['research_log'] = result.get('research_log', {})

            return self.results

        except Exception as e:
            total_time = time.time() - start_time
            print(f"\n\nERROR: {e}")
            self.logger.error(f"Stress query failed: {e}")
            import traceback
            traceback.print_exc()

            self.results = {
                'success': False,
                'query': query,
                'error': str(e),
                'total_time': total_time,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            if self.memory_profile:
                tracemalloc.stop()

            return self.results

    def print_results(self):
        """Print detailed results summary"""
        print("\n" + "=" * 100)
        print("STRESS TEST RESULTS")
        print("=" * 100)

        if self.results.get('success'):
            print("\nStatus: SUCCESS")
            print(f"Total Time: {self.results.get('total_time', 0):.2f}s")
            print(f"Answer Length: {self.results.get('answer_length', 0)} chars")
            print(f"Tokens Streamed: {self.results.get('chunk_count', 0)}")

            if self.results.get('sources'):
                print(f"\nSources Retrieved: {len(self.results['sources'])}")
                for i, source in enumerate(self.results['sources'][:5], 1):
                    title = source.get('title', 'Unknown')[:60]
                    score = source.get('score', 0)
                    print(f"  {i}. {title}... (score: {score:.4f})")
                if len(self.results['sources']) > 5:
                    print(f"  ... and {len(self.results['sources']) - 5} more")

            if self.results.get('research_log'):
                log = self.results['research_log']
                print(f"\nResearch Process:")
                print(f"  Team Members: {log.get('team_members', [])}")
                print(f"  Total Documents: {log.get('total_documents_retrieved', 0)}")

            if self.results.get('memory'):
                mem = self.results['memory']
                print(f"\nMemory Usage:")
                print(f"  Current: {mem.get('current_mb', 0):.2f} MB")
                print(f"  Peak: {mem.get('peak_mb', 0):.2f} MB")

        else:
            print("\nStatus: FAILED")
            print(f"Error: {self.results.get('error', 'Unknown')}")
            print(f"Time Before Failure: {self.results.get('total_time', 0):.2f}s")

        # Performance summary
        print("\n" + "-" * 50)
        print("PERFORMANCE SUMMARY")
        print("-" * 50)

        mode = "Quick" if self.quick_mode else "Maximum"
        print(f"Mode: {mode}")
        print(f"final_top_k: {self.config.get('final_top_k', 'N/A')}")
        print(f"research_team_size: {self.config.get('research_team_size', 'N/A')}")
        print(f"max_new_tokens: {self.config.get('max_new_tokens', 'N/A')}")

        if 'search_phases' in self.config:
            enabled = sum(1 for p in self.config['search_phases'].values() if p.get('enabled'))
            print(f"Search Phases Enabled: {enabled}/5")

        print("\n" + "=" * 100)

    def shutdown(self):
        """Clean up resources"""
        if self.pipeline:
            try:
                self.pipeline.shutdown()
                print("Pipeline shutdown complete")
            except Exception as e:
                print(f"Shutdown warning: {e}")

    def export_results(self, filepath: str):
        """Export results to JSON"""
        import json
        try:
            # Make config serializable
            export_data = self.results.copy()
            if 'config' in export_data and 'search_phases' in export_data['config']:
                # Already serializable
                pass

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"Results exported to: {filepath}")
        except Exception as e:
            print(f"Export error: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Stress Test - Single Query with Maximum Settings")
    parser.add_argument('--quick', action='store_true', help='Use moderate settings instead of maximum')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--memory', action='store_true', help='Enable memory profiling')
    parser.add_argument('--export', action='store_true', help='Export results to JSON')
    parser.add_argument('--output', type=str, help='Output file path for export')
    args = parser.parse_args()

    # Create tester
    tester = StressTester(
        quick_mode=args.quick,
        verbose=args.verbose,
        memory_profile=args.memory
    )

    # Print header
    tester.print_header()

    # Initialize
    if not tester.initialize():
        print("\nFailed to initialize. Exiting.")
        sys.exit(1)

    try:
        # Run stress query
        results = tester.run_stress_query()

        # Print results
        tester.print_results()

        # Export if requested
        if args.export:
            output_path = args.output or f"stress_single_results_{int(time.time())}.json"
            tester.export_results(output_path)

        # Return appropriate exit code
        sys.exit(0 if results.get('success') else 1)

    finally:
        tester.shutdown()


if __name__ == "__main__":
    main()
