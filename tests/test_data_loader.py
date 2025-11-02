# tests/test_data_loader.py
"""
Comprehensive tests for data loading system.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from core.data.dataset_loader import EnhancedDatasetLoader
from core.data.preprocessing import DataPreprocessor
from core.data.validators import DataValidator
from utils.cache_manager import MemoryCache, DiskCache, HybridCache
from utils.memory_monitor import MemoryMonitor

class TestMemoryCache:
    """Test memory cache functionality."""
    
    def test_basic_operations(self):
        """Test get/set operations."""
        cache = MemoryCache(max_size=10, max_memory_mb=1)
        
        # Set and get
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Get non-existent
        assert cache.get('key2') is None
        assert cache.get('key2', 'default') == 'default'
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = MemoryCache(max_size=3, max_memory_mb=10)
        
        # Fill cache
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)
        
        # Access 'a' to make it recent
        cache.get('a')
        
        # Add new item, should evict 'b' (least recently used)
        cache.set('d', 4)
        
        assert cache.get('a') == 1
        assert cache.get('b') is None
        assert cache.get('c') == 3
        assert cache.get('d') == 4
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        import time
        
        cache = MemoryCache(default_ttl=1)  # 1 second
        
        cache.set('temp', 'value')
        assert cache.get('temp') == 'value'
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get('temp') is None
    
    def test_statistics(self):
        """Test cache statistics."""
        cache = MemoryCache()
        
        cache.set('a', 1)
        cache.set('b', 2)
        
        cache.get('a')  # Hit
        cache.get('c')  # Miss
        
        stats = cache.get_stats()
        
        assert stats['size'] == 2
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 50.0


class TestDiskCache:
    """Test disk cache functionality."""
    
    def test_basic_operations(self):
        """Test disk cache get/set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(tmpdir)
            
            cache.set('key1', {'data': 'value1'})
            result = cache.get('key1')
            
            assert result == {'data': 'value1'}
    
    def test_compression(self):
        """Test compression reduces file size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Large data
            large_data = {'array': np.random.randn(1000, 1000).tolist()}
            
            # Without compression
            cache_no_comp = DiskCache(tmpdir + '/no_comp', compression=False)
            cache_no_comp.set('data', large_data)
            
            # With compression
            cache_comp = DiskCache(tmpdir + '/comp', compression=True)
            cache_comp.set('data', large_data)
            
            # Check file sizes
            size_no_comp = list(Path(tmpdir + '/no_comp').glob('*.cache'))[0].stat().st_size
            size_comp = list(Path(tmpdir + '/comp').glob('*.cache'))[0].stat().st_size
            
            assert size_comp < size_no_comp


class TestDataPreprocessor:
    """Test data preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    def test_clean_text(self, preprocessor):
        """Test text cleaning."""
        dirty_text = "  Extra   spaces   and\n\nnewlines  "
        clean = preprocessor._clean_text(dirty_text)
        
        assert "  " not in clean
        assert clean == "Extra spaces and newlines"
    
    def test_normalize_regulation_type(self, preprocessor):
        """Test regulation type normalization."""
        assert preprocessor._normalize_regulation_type('uu') == 'Undang-Undang'
        assert preprocessor._normalize_regulation_type('PP') == 'Peraturan Pemerintah'
        assert preprocessor._normalize_regulation_type('perpres') == 'Peraturan Presiden'
    
    def test_clean_year(self, preprocessor):
        """Test year extraction."""
        assert preprocessor._clean_year('2023') == '2023'
        assert preprocessor._clean_year('Tahun 2023') == '2023'
        assert preprocessor._clean_year('1945-2023') == '1945'
    
    def test_remove_duplicates(self, preprocessor):
        """Test duplicate removal."""
        records = [
            {'regulation_type': 'UU', 'regulation_number': '1', 'year': '2023', 'chunk_id': 1},
            {'regulation_type': 'UU', 'regulation_number': '1', 'year': '2023', 'chunk_id': 1},  # Duplicate
            {'regulation_type': 'PP', 'regulation_number': '2', 'year': '2023', 'chunk_id': 1}
        ]
        
        unique = preprocessor.remove_duplicates(records)
        
        assert len(unique) == 2
        assert preprocessor.cleaning_stats['duplicates'] == 1


class TestDataValidator:
    """Test data validation."""
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    def test_validate_complete_record(self, validator):
        """Test validation of complete record."""
        record = {
            'global_id': 1,
            'regulation_type': 'UU',
            'regulation_number': '13',
            'year': '2023',
            'about': 'Ketenagakerjaan',
            'content': 'Lorem ipsum dolor sit amet',
            'kg_authority_score': 0.8
        }
        
        issues = validator.validate_record(record, 0)
        
        assert len(issues['errors']) == 0
    
    def test_validate_missing_fields(self, validator):
        """Test validation catches missing fields."""
        record = {
            'global_id': 1,
            # Missing required fields
        }
        
        issues = validator.validate_record(record, 0)
        
        assert len(issues['errors']) > 0
        assert any('regulation_type' in err for err in issues['errors'])
    
    def test_validate_suspicious_year(self, validator):
        """Test validation warns on suspicious year."""
        record = {
            'global_id': 1,
            'regulation_type': 'UU',
            'regulation_number': '1',
            'year': '3000',  # Future year
            'about': 'Test',
            'content': 'Test content'
        }
        
        issues = validator.validate_record(record, 0)
        
        assert len(issues['warnings']) > 0


class TestMemoryMonitor:
    """Test memory monitoring."""
    
    def test_get_memory_info(self):
        """Test getting memory information."""
        monitor = MemoryMonitor()
        info = monitor.get_memory_info()
        
        assert 'system' in info
        assert 'process' in info
        assert 'total_gb' in info['system']
    
    def test_check_memory(self):
        """Test memory checking."""
        monitor = MemoryMonitor(warning_threshold_percent=95)
        result = monitor.check_memory()
        
        # Should be OK unless system is actually at 95%
        assert isinstance(result, bool)
    
    def test_optimize_memory(self):
        """Test memory optimization runs."""
        monitor = MemoryMonitor()
        
        # Should not raise
        monitor.optimize_memory()


# Integration test
class TestDataLoaderIntegration:
    """Integration tests for complete loader."""
    
    @pytest.mark.skipif(
        not Path('./test_dataset').exists(),
        reason="Test dataset not available"
    )
    def test_full_loading_pipeline(self):
        """Test complete loading pipeline (requires test data)."""
        loader = EnhancedDatasetLoader(
            dataset_name='test/dataset',
            embedding_dim=768,
            cache_dir=Path('./test_cache'),
            chunk_size=100,
            validate_data=True
        )
        
        success = loader.load()
        
        assert success
        assert loader.is_loaded()
        assert len(loader.all_records) > 0
        
        # Get statistics
        stats = loader.get_statistics()
        assert stats['total_records'] > 0
        
        # Cleanup
        loader.clear_cache()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--log-cli-level=INFO'])