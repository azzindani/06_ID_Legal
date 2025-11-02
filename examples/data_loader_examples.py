# examples/data_loader_examples.py
"""
Examples of using the modular data loading system.
"""
from pathlib import Path
from core.data.dataset_loader import EnhancedDatasetLoader
from core.data.index_builder import IndexBuilder
from utils.memory_monitor import MemoryMonitor
from utils.logging_config import get_logger

logger = get_logger('examples')

def example_basic_loading():
    """Example 1: Basic dataset loading."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Dataset Loading")
    logger.info("=" * 80)
    
    # Create loader
    loader = EnhancedDatasetLoader(
        dataset_name="Azzindani/ID_REG_KG_2510",
        embedding_dim=768,
        cache_dir=Path('./cache'),
        enable_cache=True,
        chunk_size=1000,
        validate_data=True
    )
    
    # Load with progress callback
    def progress_callback(msg):
        logger.info(f"  📋 {msg}")
    
    success = loader.load(progress_callback=progress_callback)
    
    if success:
        logger.info("✅ Loading successful!")
        
        # Get statistics
        stats = loader.get_statistics()
        logger.info(f"Total records: {stats['total_records']:,}")
        logger.info(f"KG enhanced: {stats['kg_enhanced']:,}")
        logger.info(f"Load time: {stats['load_time']:.1f}s")
    else:
        logger.error("❌ Loading failed")


def example_with_memory_monitoring():
    """Example 2: Loading with memory monitoring."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Loading with Memory Monitoring")
    logger.info("=" * 80)
    
    # Create memory monitor
    monitor = MemoryMonitor(warning_threshold_percent=85)
    
    # Create loader
    loader = EnhancedDatasetLoader(
        dataset_name="Azzindani/ID_REG_KG_2510",
        embedding_dim=768
    )
    
    # Check memory before
    logger.info("Memory before loading:")
    monitor.check_memory(log_level='info')
    
    # Load
    loader.load()
    
    # Check memory after
    logger.info("Memory after loading:")
    monitor.check_memory(log_level='info')
    
    # Get recommendations
    recommendations = monitor.get_recommendations()
    if recommendations:
        logger.info("Memory recommendations:")
        for rec in recommendations:
            logger.info(f"  {rec}")
    
    # Optimize if needed
    if monitor.get_memory_info()['system']['percent'] > 80:
        logger.info("Optimizing memory...")
        monitor.optimize_memory()


def example_searching_records():
    """Example 3: Searching loaded records."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Searching Records")
    logger.info("=" * 80)
    
    loader = EnhancedDatasetLoader(
        dataset_name="Azzindani/ID_REG_KG_2510",
        embedding_dim=768
    )
    
    if not loader.load():
        logger.error("Failed to load dataset")
        return
    
    # Search by regulation
    logger.info("Searching for UU No. 13 Tahun 2003...")
    results = loader.search_by_regulation(
        regulation_type='Undang-Undang',
        regulation_number='13',
        year='2003'
    )
    
    logger.info(f"Found {len(results)} matching records")
    for record in results[:3]:
        logger.info(f"  - {record['about']}")
    
    # Search by domain
    logger.info("\nSearching in labor_law domain...")
    domain_records = loader.get_records_by_domain('labor_law')
    logger.info(f"Found {len(domain_records)} records in labor_law")


def example_custom_indexes():
    """Example 4: Building custom indexes."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Building Custom Indexes")
    logger.info("=" * 80)
    
    loader = EnhancedDatasetLoader(
        dataset_name="Azzindani/ID_REG_KG_2510",
        embedding_dim=768
    )
    
    if not loader.load():
        return
    
    # Build additional indexes
    index_builder = IndexBuilder(loader.all_records)
    
    logger.info("Building all indexes...")
    indexes = index_builder.build_all_indexes()
    
    logger.info(f"Built {len(indexes)} indexes:")
    for name, index in indexes.items():
        logger.info(f"  - {name}: {len(index)} keys")
    
    # Use keyword index
    keyword_index = indexes['keyword_index']
    if 'ketenagakerjaan' in keyword_index:
        record_indices = keyword_index['ketenagakerjaan']
        logger.info(f"\nDocuments containing 'ketenagakerjaan': {len(record_indices)}")


def example_cache_usage():
    """Example 5: Cache usage and statistics."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Cache Usage")
    logger.info("=" * 80)
    
    cache_dir = Path('./example_cache')
    
    # First load (will cache)
    logger.info("First load (will cache)...")
    loader1 = EnhancedDatasetLoader(
        dataset_name="Azzindani/ID_REG_KG_2510",
        embedding_dim=768,
        cache_dir=cache_dir,
        enable_cache=True
    )
    loader1.load()
    
    # Get cache stats
    if loader1.cache:
        stats = loader1.cache.get_stats()
        logger.info(f"Cache stats after first load:")
        logger.info(f"  Memory cache: {stats['memory']['size']} items")
        logger.info(f"  Disk cache: {stats['disk']['files']} files")
    
    # Second load (from cache)
    logger.info("\nSecond load (from cache)...")
    loader2 = EnhancedDatasetLoader(
        dataset_name="Azzindani/ID_REG_KG_2510",
        embedding_dim=768,
        cache_dir=cache_dir,
        enable_cache=True
    )
    loader2.load()
    
    # Compare load times
    logger.info(f"\nLoad time comparison:")
    logger.info(f"  First load: {loader1.load_stats['load_time']:.1f}s")
    logger.info(f"  Second load: {loader2.load_stats['load_time']:.1f}s")
    
    speedup = loader1.load_stats['load_time'] / loader2.load_stats['load_time']
    logger.info(f"  Speedup: {speedup:.1f}x faster")


if __name__ == '__main__':
    import sys
    
    examples = {
        '1': example_basic_loading,
        '2': example_with_memory_monitoring,
        '3': example_searching_records,
        '4': example_custom_indexes,
        '5': example_cache_usage
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Available examples:")
        print("  python examples/data_loader_examples.py 1  # Basic loading")
        print("  python examples/data_loader_examples.py 2  # With memory monitoring")
        print("  python examples/data_loader_examples.py 3  # Searching records")
        print("  python examples/data_loader_examples.py 4  # Custom indexes")
        print("  python examples/data_loader_examples.py 5  # Cache usage")
        print("\nRunning all examples...")
        for example_func in examples.values():
            example_func()
            print("\n")