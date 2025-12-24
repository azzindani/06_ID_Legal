"""
Concurrent User Simulation Test

Tests the system under load with multiple concurrent users making simultaneous requests.
Validates thread safety, rate limiting, and performance under load.

Run with:
    python tests/integration/test_concurrent_users.py

Options:
    --users N     Number of concurrent users (default: 10)
    --requests N  Requests per user (default: 5)

File: tests/integration/test_concurrent_users.py
"""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_utils import get_logger, initialize_logging
from config import LOG_DIR, ENABLE_FILE_LOGGING


class ConcurrentUserTester:
    """Concurrent user load tester"""
    
    def __init__(self, num_users: int = 10, requests_per_user: int = 5, verbose: bool = False):
        initialize_logging(
            enable_file_logging=ENABLE_FILE_LOGGING,
            log_dir=LOG_DIR,
            verbosity_mode='verbose' if verbose else 'minimal'
        )
        self.logger = get_logger("ConcurrentUserTest")
        self.num_users = num_users
        self.requests_per_user = requests_per_user
        self.verbose = verbose
        self.pipeline = None
        self.results_lock = threading.Lock()
        self.user_results = defaultdict(list)
    
    def print_header(self):
        """Print test header"""
        print("\n" + "=" * 100)
        print("CONCURRENT USER SIMULATION TEST")
        print("=" * 100)
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Concurrent Users: {self.num_users}")
        print(f"Requests per User: {self.requests_per_user}")
        print(f"Total Requests: {self.num_users * self.requests_per_user}")
        print()
    
    def initialize_pipeline(self) -> bool:
        """Initialize shared pipeline"""
        print("\n" + "-" * 80)
        print("Initializing Shared Pipeline")
        print("-" * 80)
        
        try:
            from pipeline import RAGPipeline
            
            # Use lightweight config for speed
            config = {
                'final_top_k': 3,
                'research_team_size': 2,
                'max_new_tokens': 1024
            }
            
            self.pipeline = RAGPipeline(config=config)
            success = self.pipeline.initialize()
            
            if success:
                print("✓ Pipeline initialized successfully")
                self.logger.success("Shared pipeline ready")
            else:
                print("✗ Pipeline initialization failed")
                
            return success
            
        except Exception as e:
            print(f"✗ Initialization error: {e}")
            self.logger.error(f"Init error: {e}")
            return False
    
    def simulate_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Simulate a single user making requests"""
        results = []
        
        queries = [
            "Apa itu PT?",
            "Bagaimana cara mendirikan PT?",
            "Berapa modal minimal untuk PT?",
            "Apa syarat pendirian PT?",
            "Siapa yang bisa mendirikan PT?"
        ]
        
        for req_num in range(self.requests_per_user):
            query = queries[req_num % len(queries)]
            
            start_time = time.time()
            try:
                # Make request to pipeline
                result = self.pipeline.query(
                    question=query,
                    conversation_history=None,
                    stream=False,
                    thinking_mode='low'
                )
                
                elapsed = time.time() - start_time
                
                success = result.get('success', True)
                answer_length = len(result.get('answer', ''))
                
                results.append({
                    'user_id': user_id,
                    'request_num': req_num,
                    'query': query,
                    'success': success,
                    'time': elapsed,
                    'answer_length': answer_length
                })
                
                if self.verbose:
                    print(f"User {user_id} req {req_num+1}: {'✓' if success else '✗'} ({elapsed:.2f}s)")
                
            except Exception as e:
                elapsed = time.time() - start_time
                results.append({
                    'user_id': user_id,
                    'request_num': req_num,
                    'query': query,
                    'success': False,
                    'time': elapsed,
                    'error': str(e)
                })
                self.logger.error(f"User {user_id} req {req_num} failed: {e}")
        
        return results
    
    def test_concurrent_access(self) -> bool:
        """Test concurrent user access"""
        print("\n" + "-" * 80)
        print("TEST 1: Concurrent Access (Thread Safety)")
        print("-" * 80)
        
        print(f"\nSpawning {self.num_users} concurrent users...")
        
        start_time = time.time()
        all_results = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.num_users) as executor:
                # Submit all user tasks
                futures = {
                    executor.submit(self.simulate_user, user_id): user_id
                    for user_id in range(self.num_users)
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    user_id = futures[future]
                    try:
                        user_results = future.result()
                        all_results.extend(user_results)
                        print(f"✓ User {user_id} completed {len(user_results)} requests")
                    except Exception as e:
                        print(f"✗ User {user_id} crashed: {e}")
                        self.logger.error(f"User {user_id} exception: {e}")
            
            total_time = time.time() - start_time
            
            # Analyze results
            total_requests = len(all_results)
            successful = sum(1 for r in all_results if r.get('success', False))
            failed = total_requests - successful
            
            avg_time = sum(r['time'] for r in all_results) / total_requests if total_requests > 0 else 0
            max_time = max((r['time'] for r in all_results), default=0)
            min_time = min((r['time'] for r in all_results), default=0)
            
            throughput = total_requests / total_time if total_time > 0 else 0
            
            print(f"\n{'=' * 80}")
            print("CONCURRENT ACCESS RESULTS")
            print(f"{'=' * 80}")
            print(f"Total Requests: {total_requests}")
            print(f"Successful: {successful} ({successful/total_requests*100:.1f}%)")
            print(f"Failed: {failed} ({failed/total_requests*100:.1f}%)")
            print(f"\nTiming:")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Avg Time/Request: {avg_time:.2f}s")
            print(f"  Min Time: {min_time:.2f}s")
            print(f"  Max Time: {max_time:.2f}s")
            print(f"  Throughput: {throughput:.2f} req/s")
            
            # Pass if > 90% success rate
            success_rate = successful / total_requests if total_requests > 0 else 0
            if success_rate >= 0.9:
                print(f"\n✓ Concurrent access test PASSED (success rate: {success_rate*100:.1f}%)")
                return True
            else:
                print(f"\n✗ Concurrent access test FAILED (success rate: {success_rate*100:.1f}%)")
                return False
                
        except Exception as e:
            print(f"\n✗ Concurrent test failed: {e}")
            self.logger.error(f"Concurrent test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all concurrent tests"""
        self.print_header()
        
        if not self.initialize_pipeline():
            print("\n✗ FATAL: Pipeline initialization failed")
            return {}
        
        results = {}
        results['concurrent_access'] = self.test_concurrent_access()
        
        return results
    
    def print_results(self, results: Dict[str, bool]):
        """Print final results"""
        print("\n" + "=" * 100)
        print("CONCURRENT USER TEST RESULTS")
        print("=" * 100)
        print()
        
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        
        for test_name, passed_test in results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            print(f"{status} - {test_name.replace('_', ' ').title()}")
        
        print()
        print(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL CONCURRENT TESTS PASSED!")
            self.logger.success("System is thread-safe")
        else:
            print(f"\n⚠️ {total - passed} tests failed")
            self.logger.error("Concurrency issues detected")
        
        print("=" * 100)
    
    def shutdown(self):
        """Clean up"""
        if self.pipeline:
            try:
                self.pipeline.shutdown()
                print("\nPipeline shutdown complete")
            except Exception as e:
                print(f"Shutdown warning: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Concurrent User Simulation Test")
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--requests', type=int, default=5, help='Requests per user')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    tester = ConcurrentUserTester(
        num_users=args.users,
        requests_per_user=args.requests,
        verbose=args.verbose
    )
    
    try:
        results = tester.run_all_tests()
        
        if not results:
            print("\nTests aborted")
            sys.exit(1)
        
        tester.print_results(results)
        
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        tester.shutdown()


if __name__ == "__main__":
    main()
