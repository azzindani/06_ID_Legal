"""
Edge Case and Error Handling Test

Tests unusual inputs, error scenarios, and boundary conditions.
Validates system robustness and graceful failure handling.

Run with:
    python tests/integration/test_edge_cases.py

File: tests/integration/test_edge_cases.py
"""

import sys
import os
from typing import Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_utils import get_logger, initialize_logging
from config import LOG_DIR, ENABLE_FILE_LOGGING


class EdgeCaseTester:
    """Edge case and error handling tester"""
    
    def __init__(self, verbose: bool = False):
        initialize_logging(
            enable_file_logging=ENABLE_FILE_LOGGING,
            log_dir=LOG_DIR,
            verbosity_mode='verbose' if verbose else 'minimal'
        )
        self.logger = get_logger("EdgeCaseTest")
        self.verbose = verbose
        self.pipeline = None
    
    def print_header(self):
        """Print test header"""
        print("\n" + "=" * 100)
        print("EDGE CASE & ERROR HANDLING TEST")
        print("=" * 100)
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def initialize_pipeline(self) -> bool:
        """Initialize pipeline"""
        print("\n" + "-" * 80)
        print("Initializing Pipeline")
        print("-" * 80)
        
        try:
            from pipeline import RAGPipeline
            
            config = {'final_top_k': 3, 'max_new_tokens': 1024}
            self.pipeline = RAGPipeline(config=config)
            success = self.pipeline.initialize()
            
            if success:
                print("✓ Pipeline initialized")
                self.logger.success("Pipeline ready")
            
            return success
            
        except Exception as e:
            print(f"✗ Init error: {e}")
            self.logger.error(f"Init error: {e}")
            return False
    
    def test_unusual_inputs(self) -> bool:
        """Test unusual but valid inputs"""
        print("\n" + "-" * 80)
        print("TEST 1: Unusual Inputs")
        print("-" * 80)
        
        test_cases = [
            # Very long query (near max)
            {
                'name': 'Very long query (1900 chars)',
                'query': 'A' * 1900,  # Exactly 1900 chars
                'should_work': True
            },
            # Unicode and special Indonesian characters
            {
                'name': 'Unicode characters',
                'query': 'Apa itu "Perseroan Terbatas" (PT)? ©️ ®️ ™️',
                'should_work': True
            },
            # Emoji
            {
                'name': 'Emoji in query',
                'query': 'Apa itu PT? 🏢 🏛️ ⚖️',
                'should_work': True
            },
            # Mixed case
            {
                'name': 'Mixed case',
                'query': 'ApA iTu Uu PeRdAtA?',
                'should_work': True
            },
            # Excessive whitespace
            {
                'name': 'Excessive whitespace',
                'query': 'Apa    itu       UU    Perdata?',
                'should_work': True
            },
            # Numbers only
            {
                'name': 'Numbers in query',
                'query': 'UU No 40 Tahun 2007',
                'should_work': True
            },
        ]
        
        passed = 0
        failed = 0
        
        for test in test_cases:
            try:
                from security import sanitize_query
                sanitized = sanitize_query(test['query'])
                
                if test['should_work']:
                    print(f"✓ {test['name']}: Accepted")
                    passed += 1
                else:
                    print(f"✗ {test['name']}: Should have been rejected")
                    print(f"   Query length: {len(test['query'])} chars")
                    print(f"   Query: {test['query'][:50]}...")
                    failed += 1
                    
            except ValueError as e:
                if not test['should_work']:
                    print(f"✓ {test['name']}: Correctly rejected")
                    passed += 1
                else:
                    print(f"✗ {test['name']}: Incorrectly rejected")
                    print(f"   Query length: {len(test['query'])} chars (max 2000)")
                    print(f"   Error: {e}")
                    print(f"   Query preview: {test['query'][:50]}...")
                    failed += 1
            except Exception as e:
                print(f"✗ {test['name']}: Unexpected error - {e}")
                print(f"   Query length: {len(test['query'])} chars")
                import traceback
                traceback.print_exc()
                failed += 1
        
        print(f"\nPassed: {passed}/{len(test_cases)}")
        return failed == 0
    
    def test_boundary_conditions(self) -> bool:
        """Test boundary conditions"""
        print("\n" + "-" * 80)
        print("TEST 2: Boundary Conditions")
        print("-" * 80)
        
        from security import sanitize_query
        
        try:
            # Test 1: Exactly at max length (2000 chars)
            max_query = 'A' * 2000
            try:
                sanitize_query(max_query)
                print("✓ Max length (2000 chars) accepted")
            except ValueError:
                print("✗ Max length rejected")
                return False
            
            # Test 2: One over max length
            over_max = 'A' * 2001
            try:
                sanitize_query(over_max)
                print("✗ Over-length (2001 chars) should be rejected")
                return False
            except ValueError:
                print("✓ Over-length (2001 chars) correctly rejected")
            
            # Test 3: Minimum valid query
            try:
                sanitize_query("A")
                print("✓ Single character query accepted")
            except ValueError:
                print("✗ Single character rejected")
                return False
            
            # Test 4: Empty string (should fail)
            try:
                sanitize_query("")
                print("✗ Empty string should be rejected")
                return False
            except ValueError:
                print("✓ Empty string correctly rejected")
            
            # Test 5: Whitespace only (should fail)
            try:
                sanitize_query("   ")
                print("✗ Whitespace-only should be rejected")
                return False
            except ValueError:
                print("✓ Whitespace-only correctly rejected")
            
            return True
            
        except Exception as e:
            print(f"✗ Boundary test failed: {e}")
            return False
    
    def test_error_recovery(self) -> bool:
        """Test error recovery"""
        print("\n" + "-" * 80)
        print("TEST 3: Error Recovery")
        print("-" * 80)
        
        try:
            # Test 1: Invalid thinking_mode
            try:
                result = self.pipeline.query(
                    question="Apa itu PT?",
                    thinking_mode='invalid_mode'
                )
                # Should default to 'low' instead of crashing
                print("✓ Invalid thinking_mode handled gracefully")
            except Exception as e:
                print(f"✗ System crashed on invalid thinking_mode: {e}")
                return False
            
            # Test 2: Very simple query (should still work)
            result = self.pipeline.query(
                question="PT",
                stream=False,
                thinking_mode='low'
            )
            if result.get('success', True):
                print("✓ Very short query handled")
            else:
                print("✗ Short query failed")
                return False
            
            # Test 3: Query after query (no crash from state)
            for i in range(3):
                result = self.pipeline.query(
                    question=f"Test query {i+1}",
                    stream=False,
                    thinking_mode='low'
                )
                if not result.get('success', True):
                    print(f"✗ Failed on query {i+1}")
                    return False
            print("✓ Multiple sequential queries handled")
            
            return True
            
        except Exception as e:
            print(f"✗ Error recovery test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all edge case tests"""
        self.print_header()
        
        if not self.initialize_pipeline():
            print("\n✗ FATAL: Pipeline init failed")
            return {}
        
        results = {}
        results['unusual_inputs'] = self.test_unusual_inputs()
        results['boundary_conditions'] = self.test_boundary_conditions()
        results['error_recovery'] = self.test_error_recovery()
        
        return results
    
    def print_results(self, results: Dict[str, bool]):
        """Print final results"""
        print("\n" + "=" * 100)
        print("EDGE CASE TEST RESULTS")
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
            print("\n🎉 ALL EDGE CASE TESTS PASSED!")
            self.logger.success("System handles edge cases well")
        else:
            print(f"\n⚠️ {total - passed} tests failed")
            self.logger.error("Edge case issues detected")
        
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
    
    parser = argparse.ArgumentParser(description="Edge Case & Error Handling Test")
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    tester = EdgeCaseTester(verbose=args.verbose)
    
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
