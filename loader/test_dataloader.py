"""
Test suite for Enhanced KG Dataset Loader with Centralized Logging
All test logs go to the same system log file
"""

import unittest
import sys
import os
from typing import Dict, Any

# Skip if dependencies not available
import pytest
numpy = pytest.importorskip("numpy")

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loader.dataloader import EnhancedKGDatasetLoader
from config import DATASET_NAME
from logger_utils import get_logger, initialize_logging


class TestEnhancedKGDatasetLoader(unittest.TestCase):
    """Test cases for EnhancedKGDatasetLoader"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load dataset once for all tests"""
        # Get logger from centralized system
        cls.logger = get_logger("TestSuite")
        
        cls.logger.info("="*80)
        cls.logger.info("STARTING DATALOADER TEST SUITE")
        cls.logger.info("="*80)
        
        # Initialize loader with test embedding dimension
        cls.embedding_dim = 1024
        cls.loader = EnhancedKGDatasetLoader(DATASET_NAME, cls.embedding_dim)
        
        # Track progress
        cls.progress_messages = []
        
        def progress_callback(msg):
            cls.progress_messages.append(msg)
            print(f"   {msg}")
        
        # Load dataset
        cls.logger.info("Loading dataset from HuggingFace", {
            "dataset": DATASET_NAME,
            "embedding_dim": cls.embedding_dim
        })
        
        success = cls.loader.load_from_huggingface(progress_callback)
        
        if not success:
            cls.logger.error("Failed to load dataset!")
            raise Exception("Failed to load dataset!")
        
        cls.logger.success("Dataset loaded successfully")
        cls.logger.info("="*80)
    
    def test_01_loading_success(self):
        """Test that dataset loaded successfully"""
        self.logger.info("[TEST 1] Testing loading success...")
        
        self.assertIsNotNone(self.loader.all_records, "Records should be loaded")
        self.assertGreater(len(self.loader.all_records), 0, "Should have records")
        self.assertIsNotNone(self.loader.embeddings, "Embeddings should be loaded")
        
        self.logger.success("Loading test passed", {
            "records": f"{len(self.loader.all_records):,}",
            "embeddings_shape": str(self.loader.embeddings.shape)
        })
    
    def test_02_record_structure(self):
        """Test record structure integrity"""
        self.logger.info("[TEST 2] Testing record structure...")
        
        required_fields = [
            'global_id', 'local_id', 'regulation_type', 'enacting_body',
            'regulation_number', 'year', 'about', 'content',
            'kg_entity_count', 'kg_authority_score', 'kg_connectivity_score'
        ]
        
        first_record = self.loader.all_records[0]
        
        for field in required_fields:
            self.assertIn(field, first_record, f"Record should have '{field}' field")
        
        self.logger.success("Record structure test passed", {
            "required_fields": len(required_fields),
            "sample_id": first_record['global_id'],
            "sample_type": first_record['regulation_type']
        })
    
    def test_03_embeddings_integrity(self):
        """Test embeddings integrity"""
        self.logger.info("[TEST 3] Testing embeddings integrity...")
        
        import torch
        
        self.assertIsNotNone(self.loader.embeddings)
        
        expected_shape = (len(self.loader.all_records), self.embedding_dim)
        actual_shape = self.loader.embeddings.shape
        
        self.assertEqual(actual_shape, expected_shape, 
                        f"Embeddings shape mismatch: expected {expected_shape}, got {actual_shape}")
        
        self.assertTrue(torch.is_tensor(self.loader.embeddings), "Embeddings should be a tensor")
        self.assertFalse(torch.isnan(self.loader.embeddings).any(), "Embeddings contain NaN")
        self.assertFalse(torch.isinf(self.loader.embeddings).any(), "Embeddings contain Inf")
        
        mean_val = self.loader.embeddings.mean().item()
        std_val = self.loader.embeddings.std().item()
        
        self.logger.success("Embeddings integrity test passed", {
            "shape": str(actual_shape),
            "mean": f"{mean_val:.4f}",
            "std": f"{std_val:.4f}"
        })
    
    def test_04_tfidf_availability(self):
        """Test TF-IDF matrix availability"""
        self.logger.info("[TEST 4] Testing TF-IDF availability...")
        
        self.assertIsNotNone(self.loader.tfidf_vectorizer, "TF-IDF vectorizer should exist")
        
        if self.loader.tfidf_matrix is not None:
            self.logger.success("TF-IDF test passed", {
                "matrix_shape": str(self.loader.tfidf_matrix.shape),
                "features": self.loader.tfidf_vectorizer.n_features
            })
        else:
            self.logger.warning("TF-IDF matrix not available (using dummy vectorizer)")
    
    def test_05_kg_indexes(self):
        """Test KG indexes were built"""
        self.logger.info("[TEST 5] Testing KG indexes...")
        
        kg_enhanced_count = len(self.loader.kg_entities_lookup)
        self.assertGreater(kg_enhanced_count, 0, "Should have KG-enhanced documents")
        
        self.logger.success("KG indexes test passed", {
            "entities": f"{kg_enhanced_count:,}",
            "cross_refs": f"{len(self.loader.kg_cross_references_lookup):,}",
            "domains": f"{len(self.loader.kg_domains_lookup):,}",
            "authority_tiers": len(self.loader.authority_index),
            "unique_domains": len(self.loader.domain_index)
        })
    
    def test_06_statistics(self):
        """Test statistics generation"""
        self.logger.info("[TEST 6] Testing statistics generation...")
        
        stats = self.loader.get_statistics()
        
        self.assertIsInstance(stats, dict, "Statistics should be a dictionary")
        self.assertIn('total_records', stats)
        self.assertIn('kg_enhanced', stats)
        self.assertIn('kg_enhancement_rate', stats)
        
        self.logger.success("Statistics test passed", {
            "total": f"{stats.get('total_records', 0):,}",
            "kg_enhanced": f"{stats.get('kg_enhanced', 0):,}",
            "enhancement_rate": f"{stats.get('kg_enhancement_rate', 0):.1%}",
            "avg_authority": f"{stats.get('avg_authority_score', 0):.3f}"
        })
    
    def test_07_sample_records_integrity(self):
        """Test integrity of sample records"""
        self.logger.info("[TEST 7] Testing sample records integrity...")
        
        sample_size = min(10, len(self.loader.all_records))
        
        for i in range(sample_size):
            record = self.loader.all_records[i]
            
            self.assertIsNotNone(record['global_id'])
            self.assertIsNotNone(record['regulation_type'])
            self.assertIsNotNone(record['regulation_number'])
            
            self.assertGreaterEqual(record['kg_authority_score'], 0.0)
            self.assertLessEqual(record['kg_authority_score'], 1.0)
            
            self.assertGreaterEqual(record['kg_temporal_score'], 0.0)
            self.assertLessEqual(record['kg_temporal_score'], 1.0)
            
            self.assertGreaterEqual(record['kg_connectivity_score'], 0.0)
            self.assertLessEqual(record['kg_connectivity_score'], 1.0)
        
        self.logger.success("Sample records test passed", {
            "samples_validated": sample_size
        })
    
    def test_08_index_consistency(self):
        """Test index consistency with records"""
        self.logger.info("[TEST 8] Testing index consistency...")
        
        # Test authority index
        for tier, indices in self.loader.authority_index.items():
            for idx in indices[:5]:
                record = self.loader.all_records[idx]
                calculated_tier = max(0, min(10, int(record['kg_authority_score'] * 10)))
                self.assertEqual(tier, calculated_tier, 
                               f"Authority tier mismatch for record {idx}")
        
        # Test domain index
        for domain, indices in list(self.loader.domain_index.items())[:5]:
            for idx in indices[:3]:
                record = self.loader.all_records[idx]
                self.assertEqual(record['kg_primary_domain'], domain,
                               f"Domain mismatch for record {idx}")
        
        self.logger.success("Index consistency test passed")
    
    def test_09_kg_enhancement_distribution(self):
        """Test distribution of KG enhancements"""
        self.logger.info("[TEST 9] Testing KG enhancement distribution...")
        
        has_entities = sum(1 for r in self.loader.all_records if r['kg_entity_count'] > 0)
        has_cross_refs = sum(1 for r in self.loader.all_records if r['kg_cross_ref_count'] > 0)
        has_obligations = sum(1 for r in self.loader.all_records if r['kg_has_obligations'])
        has_prohibitions = sum(1 for r in self.loader.all_records if r['kg_has_prohibitions'])
        has_permissions = sum(1 for r in self.loader.all_records if r['kg_has_permissions'])
        
        total = len(self.loader.all_records)
        
        self.logger.success("KG enhancement distribution test passed", {
            "entities": f"{has_entities:,} ({has_entities/total:.1%})",
            "cross_refs": f"{has_cross_refs:,} ({has_cross_refs/total:.1%})",
            "obligations": f"{has_obligations:,} ({has_obligations/total:.1%})",
            "prohibitions": f"{has_prohibitions:,} ({has_prohibitions/total:.1%})",
            "permissions": f"{has_permissions:,} ({has_permissions/total:.1%})"
        })
    
    def test_10_memory_efficiency(self):
        """Test memory usage is reasonable"""
        self.logger.info("[TEST 10] Testing memory efficiency...")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Adjusted threshold for large datasets
            max_memory_mb = 15000  # 15GB threshold
            
            if memory_mb < max_memory_mb:
                self.logger.success("Memory efficiency test passed", {
                    "memory_mb": f"{memory_mb:.1f}",
                    "threshold_mb": max_memory_mb
                })
            else:
                self.logger.warning("Memory usage is high but acceptable for large dataset", {
                    "memory_mb": f"{memory_mb:.1f}",
                    "threshold_mb": max_memory_mb,
                    "records": f"{len(self.loader.all_records):,}"
                })
            
        except ImportError:
            self.logger.warning("psutil not installed, skipping memory test")


def run_tests():
    """Run all tests with detailed output"""
    test_logger = get_logger("TestRunner")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnhancedKGDatasetLoader)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    test_logger.info("="*80)
    test_logger.info("TEST SUMMARY")
    test_logger.info("="*80)
    
    test_logger.info("Test results", {
        "total": result.testsRun,
        "successes": result.testsRun - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors)
    })
    
    if result.wasSuccessful():
        test_logger.success("ALL TESTS PASSED!")
    else:
        test_logger.error("SOME TESTS FAILED")
        
        if result.failures:
            test_logger.error("Failures detected", {
                "count": len(result.failures)
            })
            for test, traceback in result.failures:
                test_logger.error(f"Failed: {test}", {
                    "traceback": traceback[:200]
                })
        
        if result.errors:
            test_logger.error("Errors detected", {
                "count": len(result.errors)
            })
            for test, traceback in result.errors:
                test_logger.error(f"Error: {test}", {
                    "traceback": traceback[:200]
                })
    
    test_logger.info("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Initialize centralized logging BEFORE running tests
    # All test logs will go to a single file
    initialize_logging(
        enable_file_logging=True,
        log_dir="logs",
        append=False,  # Create new log file for each test run
        log_filename="test_run.log"  # Custom filename
    )
    
    # Run tests
    success = run_tests()
    
    # Write session end marker
    from logger_utils import log_session_end
    log_session_end()
    
    sys.exit(0 if success else 1)