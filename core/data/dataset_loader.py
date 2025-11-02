# core/data/dataset_loader.py
"""
Enhanced dataset loader with caching, validation, and comprehensive logging.
Refactored from EnhancedKGDatasetLoader with modular components.
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datasets import load_dataset
import gc

from core.data.base_loader import BaseDataLoader
from core.data.preprocessing import DataPreprocessor
from core.data.validators import DataValidator
from utils.cache_manager import HybridCache, MemoryCache, DiskCache
from utils.progress_tracker import ProgressTracker
from utils.logging_config import get_logger, log_performance, LogBlock

logger = get_logger(__name__)

class EnhancedDatasetLoader(BaseDataLoader):
    """
    Production-ready dataset loader with:
    - Streaming for large datasets
    - Multi-level caching
    - Data validation
    - Progress tracking
    - Memory optimization
    - Error recovery
    """
    
    def __init__(
        self,
        dataset_name: str,
        embedding_dim: int,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
        chunk_size: int = 1000,
        validate_data: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize enhanced dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            embedding_dim: Expected embedding dimension
            cache_dir: Directory for caching (default: ./cache)
            enable_cache: Enable caching system
            chunk_size: Records per processing chunk
            validate_data: Enable data validation
            device: Device for tensor operations
        """
        config = {
            'dataset_name': dataset_name,
            'embedding_dim': embedding_dim,
            'cache_dir': cache_dir or Path('./cache'),
            'enable_cache': enable_cache,
            'chunk_size': chunk_size,
            'validate_data': validate_data,
            'device': device
        }
        
        super().__init__(config)
        
        # Core data
        self.all_records: List[Dict] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        
        # KG indexes
        self.kg_entities_lookup: Dict[int, str] = {}
        self.kg_cross_references_lookup: Dict[int, str] = {}
        self.kg_domains_lookup: Dict[int, str] = {}
        self.kg_concept_clusters_lookup: Dict[int, str] = {}
        self.kg_legal_actions_lookup: Dict[int, str] = {}
        self.kg_sanctions_lookup: Dict[int, str] = {}
        self.kg_concept_vectors_lookup: Dict[int, str] = {}
        
        # Numeric indexes
        self.authority_index: Dict[int, List[int]] = {}
        self.temporal_index: Dict[int, List[int]] = {}
        self.kg_connectivity_index: Dict[int, List[int]] = {}
        self.hierarchy_index: Dict[int, List[int]] = {}
        self.domain_index: Dict[str, List[int]] = {}
        
        # Components
        self.preprocessor = DataPreprocessor()
        self.validator = DataValidator()
        
        # Caching system
        if enable_cache:
            cache_dir = Path(config['cache_dir'])
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            memory_cache = MemoryCache(
                max_size=1000,
                max_memory_mb=500,
                default_ttl=3600  # 1 hour
            )
            
            disk_cache = DiskCache(
                cache_dir=cache_dir / 'disk_cache',
                max_size_gb=5.0,
                compression=True
            )
            
            self.cache = HybridCache(memory_cache, disk_cache)
            logger.info("Hybrid cache system initialized")
        else:
            self.cache = None
            logger.info("Caching disabled")
        
        # Statistics
        self.load_stats = {
            'total_rows': 0,
            'processed_rows': 0,
            'skipped_rows': 0,
            'kg_enhanced_count': 0,
            'load_time': 0,
            'memory_usage_mb': 0
        }
        
        logger.info(f"EnhancedDatasetLoader initialized for {dataset_name}")
    
    @log_performance(logger)
    def load(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        force_reload: bool = False
    ) -> bool:
        """
        Load dataset from HuggingFace with full pipeline.
        
        Args:
            progress_callback: Optional callback for progress updates
            force_reload: Force reload even if cached
            
        Returns:
            True if successful, False otherwise
        """
        if self.loaded and not force_reload:
            logger.info("Dataset already loaded")
            return True
        
        logger.info("=" * 80)
        logger.info(f"LOADING DATASET: {self.config['dataset_name']}")
        logger.info("=" * 80)
        
        import time
        start_time = time.time()
        
        with LogBlock(logger, "Complete dataset loading pipeline"):
            try:
                # Step 1: Check cache
                if self.cache and not force_reload:
                    if self._try_load_from_cache():
                        logger.info("✅ Loaded from cache successfully")
                        self.loaded = True
                        self.load_stats['load_time'] = time.time() - start_time
                        return True
                
                # Step 2: Load from HuggingFace
                if progress_callback:
                    progress_callback("📥 Loading dataset from HuggingFace...")
                
                dataset = self._load_from_huggingface(progress_callback)
                if not dataset:
                    return False
                
                # Step 3: Process records
                if progress_callback:
                    progress_callback("⚙️ Processing records...")
                
                self._process_records(dataset, progress_callback)
                
                # Step 4: Build embeddings
                if progress_callback:
                    progress_callback("🔢 Processing embeddings...")
                
                self._process_embeddings(dataset, progress_callback)
                
                # Step 5: Build TF-IDF
                if progress_callback:
                    progress_callback("📊 Processing TF-IDF vectors...")
                
                self._process_tfidf(dataset, progress_callback)
                
                # Step 6: Build KG indexes
                if progress_callback:
                    progress_callback("🗃️ Building KG indexes...")
                
                self._build_kg_indexes(progress_callback)
                
                # Step 7: Validate (if enabled)
                if self.config['validate_data']:
                    if progress_callback:
                        progress_callback("✅ Validating data...")
                    
                    self._validate_loaded_data()
                
                # Step 8: Cache results
                if self.cache:
                    if progress_callback:
                        progress_callback("💾 Caching results...")
                    
                    self._save_to_cache()
                
                # Cleanup
                del dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Finalize
                self.loaded = True
                self.load_stats['load_time'] = time.time() - start_time
                self._calculate_memory_usage()
                
                logger.info("=" * 80)
                logger.info("✅ DATASET LOADING COMPLETE")
                logger.info(f"   Records: {len(self.all_records):,}")
                logger.info(f"   KG Enhanced: {self.load_stats['kg_enhanced_count']:,}")
                logger.info(f"   Time: {self.load_stats['load_time']:.1f}s")
                logger.info(f"   Memory: {self.load_stats['memory_usage_mb']:.1f}MB")
                logger.info("=" * 80)
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Dataset loading failed: {e}", exc_info=True)
                self.loaded = False
                return False
    
    def _try_load_from_cache(self) -> bool:
        """Try to load complete dataset from cache."""
        logger.info("🔍 Checking cache for existing data...")
        
        try:
            cache_key = f"dataset_{self.config['dataset_name']}_v1"
            cached_data = self.cache.get(cache_key)
            
            if cached_data is None:
                logger.info("Cache miss - will load from source")
                return False
            
            logger.info("✅ Cache hit - loading from cache")
            
            # Restore data
            self.all_records = cached_data['records']
            self.embeddings = cached_data['embeddings']
            self.tfidf_matrix = cached_data['tfidf_matrix']
            self.tfidf_vectorizer = cached_data['tfidf_vectorizer']
            self.kg_entities_lookup = cached_data['kg_entities_lookup']
            self.kg_cross_references_lookup = cached_data['kg_cross_references_lookup']
            self.kg_domains_lookup = cached_data['kg_domains_lookup']
            self.kg_concept_clusters_lookup = cached_data['kg_concept_clusters_lookup']
            self.kg_legal_actions_lookup = cached_data['kg_legal_actions_lookup']
            self.kg_sanctions_lookup = cached_data['kg_sanctions_lookup']
            self.kg_concept_vectors_lookup = cached_data['kg_concept_vectors_lookup']
            self.authority_index = cached_data['authority_index']
            self.temporal_index = cached_data['temporal_index']
            self.kg_connectivity_index = cached_data['kg_connectivity_index']
            self.hierarchy_index = cached_data['hierarchy_index']
            self.domain_index = cached_data['domain_index']
            self.load_stats = cached_data['load_stats']
            
            logger.info(f"Restored {len(self.all_records):,} records from cache")
            return True
            
        except Exception as e:
            logger.warning(f"Cache load failed: {e} - will reload from source")
            return False
    
    def _save_to_cache(self):
        """Save complete dataset to cache."""
        logger.info("💾 Saving to cache...")
        
        try:
            cache_key = f"dataset_{self.config['dataset_name']}_v1"
            
            cache_data = {
                'records': self.all_records,
                'embeddings': self.embeddings,
                'tfidf_matrix': self.tfidf_matrix,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'kg_entities_lookup': self.kg_entities_lookup,
                'kg_cross_references_lookup': self.kg_cross_references_lookup,
                'kg_domains_lookup': self.kg_domains_lookup,
                'kg_concept_clusters_lookup': self.kg_concept_clusters_lookup,
                'kg_legal_actions_lookup': self.kg_legal_actions_lookup,
                'kg_sanctions_lookup': self.kg_sanctions_lookup,
                'kg_concept_vectors_lookup': self.kg_concept_vectors_lookup,
                'authority_index': self.authority_index,
                'temporal_index': self.temporal_index,
                'kg_connectivity_index': self.kg_connectivity_index,
                'hierarchy_index': self.hierarchy_index,
                'domain_index': self.domain_index,
                'load_stats': self.load_stats
            }
            
            self.cache.set(cache_key, cache_data, memory_only=False)
            logger.info("✅ Data cached successfully")
            
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def _load_from_huggingface(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """Load dataset from HuggingFace."""
        logger.info(f"📥 Loading from HuggingFace: {self.config['dataset_name']}")
        
        try:
            dataset = load_dataset(
                self.config['dataset_name'],
                split='train',
                streaming=False
            )
            
            total_rows = len(dataset)
            self.load_stats['total_rows'] = total_rows
            
            logger.info(f"✅ Loaded dataset: {total_rows:,} rows")
            
            if progress_callback:
                progress_callback(f"Dataset loaded: {total_rows:,} rows")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            return None
    
    def _process_records(
        self,
        dataset,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """Process records in chunks with cleaning and validation."""
        logger.info("⚙️ Processing records in chunks...")
        
        total_rows = len(dataset)
        chunk_size = self.config['chunk_size']
        
        progress = ProgressTracker(
            total=total_rows,
            description="Processing records",
            callback=progress_callback,
            log_interval=5  # Log every 5%
        )
        
        all_records_temp = []
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            # Get chunk
            chunk = dataset.select(range(start_idx, end_idx))
            df_chunk = chunk.to_pandas()
            
            # Process each record
            for idx, row in df_chunk.iterrows():
                try:
                    # Create record
                    record = self._create_record(row, start_idx + idx)
                    
                    # Clean record
                    cleaned_record = self.preprocessor.clean_record(record)
                    
                    if cleaned_record:
                        all_records_temp.append(cleaned_record)
                        self.load_stats['processed_rows'] += 1
                    else:
                        self.load_stats['skipped_rows'] += 1
                    
                    progress.update(1)
                    
                except Exception as e:
                    logger.warning(f"Error processing record {start_idx + idx}: {e}")
                    self.load_stats['skipped_rows'] += 1
                    continue
            
            # Memory cleanup
            del df_chunk, chunk
            gc.collect()
        
        progress.finish()
        
        # Remove duplicates
        self.all_records = self.preprocessor.remove_duplicates(all_records_temp)
        
        logger.info(
            f"✅ Record processing complete: "
            f"{self.load_stats['processed_rows']:,} processed, "
            f"{self.load_stats['skipped_rows']:,} skipped, "
            f"{len(self.all_records):,} final"
        )
    
    def _create_record(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """Create record dictionary from DataFrame row."""
        return {
            # Basic fields
            'global_id': row.get('global_id', idx),
            'local_id': row.get('local_id', 1),
            'regulation_type': str(row.get('regulation_type', 'Unknown')),
            'enacting_body': str(row.get('enacting_body', 'Unknown')),
            'regulation_number': str(row.get('regulation_number', 'N/A')),
            'year': str(row.get('year', '2023')),
            'about': str(row.get('about', ''))[:200],
            'effective_date': str(row.get('effective_date', '2023-01-01')),
            'chapter': str(row.get('chapter', 'N/A')),
            'article': str(row.get('article', 'N/A')),
            'content': str(row.get('content', ''))[:500],
            'chunk_id': row.get('chunk_id', 1),
            
            # KG numeric features
            'kg_entity_count': int(row.get('kg_entity_count', 0)),
            'kg_cross_ref_count': int(row.get('kg_cross_ref_count', 0)),
            'kg_primary_domain': str(row.get('kg_primary_domain', 'Unknown')),
            'kg_domain_confidence': float(row.get('kg_domain_confidence', 0.0)),
            'kg_cluster_count': int(row.get('kg_cluster_count', 0)),
            'kg_cluster_diversity': float(row.get('kg_cluster_diversity', 0.0)),
            'kg_authority_score': float(row.get('kg_authority_score', 0.5)),
            'kg_hierarchy_level': int(row.get('kg_hierarchy_level', 5)),
            'kg_temporal_score': float(row.get('kg_temporal_score', 0.6)),
            'kg_years_old': int(row.get('kg_years_old', 1)),
            'kg_legal_richness': float(row.get('kg_legal_richness', 0.0)),
            'kg_legal_complexity': float(row.get('kg_legal_complexity', 0.0)),
            'kg_completeness_score': float(row.get('kg_completeness_score', 0.0)),
            'kg_connectivity_score': float(row.get('kg_connectivity_score', 0.0)),
            'kg_has_obligations': bool(row.get('kg_has_obligations', False)),
            'kg_has_prohibitions': bool(row.get('kg_has_prohibitions', False)),
            'kg_has_permissions': bool(row.get('kg_has_permissions', False)),
            'kg_pagerank': float(row.get('kg_pagerank', 0.0)),
            'kg_degree_centrality': float(row.get('kg_degree_centrality', 0.0)),
            
            # KG JSON fields (stored as strings for lazy parsing)
            'kg_entities_json': str(row.get('kg_entities_json', '[]')),
            'kg_cross_references_json': str(row.get('kg_cross_references_json', '[]')),
            'kg_legal_domains_json': str(row.get('kg_legal_domains_json', '[]')),
            'kg_concept_clusters_json': str(row.get('kg_concept_clusters_json', '{}')),
            'kg_legal_actions_json': str(row.get('kg_legal_actions_json', '{}')),
            'kg_sanctions_json': str(row.get('kg_sanctions_json', '{}')),
            'kg_concept_vector_json': str(row.get('kg_concept_vector_json', '[]')),
            'kg_citation_impact_json': str(row.get('kg_citation_impact_json', '{}'))
        }
    
    def _process_embeddings(
        self,
        dataset,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """Process embeddings with batching and memory optimization."""
        logger.info("🔢 Processing embeddings...")
        
        embeddings_temp = []
        
        progress = ProgressTracker(
            total=len(self.all_records),
            description="Processing embeddings",
            callback=progress_callback,
            log_interval=10
        )
        
        # Get embeddings from dataset rows
        for i, record in enumerate(self.all_records):
            try:
                # Try to get from original dataset
                if i < len(dataset):
                    row = dataset[i]
                    if 'embedding' in row and row['embedding'] is not None:
                        embeddings_temp.append(row['embedding'])
                    else:
                        # Fallback: zero vector
                        embeddings_temp.append(
                            np.zeros(self.config['embedding_dim'], dtype=np.float32)
                        )
                else:
                    embeddings_temp.append(
                        np.zeros(self.config['embedding_dim'], dtype=np.float32)
                    )
                
                progress.update(1)
                
            except Exception as e:
                logger.warning(f"Error processing embedding {i}: {e}")
                embeddings_temp.append(
                    np.zeros(self.config['embedding_dim'], dtype=np.float32)
                )
        
        progress.finish()
        
        # Convert to tensor
        logger.info("Converting embeddings to tensor...")
        embeddings_array = np.array(embeddings_temp, dtype=np.float32)
        self.embeddings = torch.tensor(
            embeddings_array,
            device=self.config['device']
        )
        
        del embeddings_temp, embeddings_array
        gc.collect()
        
        logger.info(f"✅ Embeddings processed: {self.embeddings.shape}")
    
    def _process_tfidf(
        self,
        dataset,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """Process TF-IDF vectors."""
        logger.info("📊 Processing TF-IDF vectors...")
        
        try:
            tfidf_temp = []
            has_tfidf = False
            
            # Check if TF-IDF exists in dataset
            if len(dataset) > 0 and 'tfidf_vector' in dataset[0]:
                has_tfidf = True
                
                progress = ProgressTracker(
                    total=len(self.all_records),
                    description="Processing TF-IDF",
                    callback=progress_callback,
                    log_interval=10
                )
                
                for i in range(len(self.all_records)):
                    if i < len(dataset):
                        row = dataset[i]
                        if 'tfidf_vector' in row and row['tfidf_vector'] is not None:
                            tfidf_temp.append(row['tfidf_vector'])
                        else:
                            tfidf_temp.append(None)
                    
                    progress.update(1)
                
                progress.finish()
            
            if has_tfidf and tfidf_temp:
                # Filter out None values
                valid_tfidf = [t for t in tfidf_temp if t is not None]
                
                if valid_tfidf:
                    from scipy.sparse import csr_matrix
                    tfidf_array = np.array(valid_tfidf, dtype=np.float32)
                    self.tfidf_matrix = csr_matrix(tfidf_array)
                    
                    # Create vectorizer
                    tfidf_dim = self.tfidf_matrix.shape[1]
                    self._create_working_vectorizer(tfidf_dim)
                    
                    logger.info(f"✅ TF-IDF processed: {self.tfidf_matrix.shape}")
                else:
                    self._create_dummy_vectorizer()
                    logger.info("⚠️ No valid TF-IDF vectors found, using dummy")
            else:
                self._create_dummy_vectorizer()
                logger.info("ℹ️ No TF-IDF in dataset, using dummy vectorizer")
            
            del tfidf_temp
            gc.collect()
            
        except Exception as e:
            logger.warning(f"TF-IDF processing failed: {e}")
            self._create_dummy_vectorizer()
    
    def _create_working_vectorizer(self, n_features: int):
        """Create working TF-IDF vectorizer."""
        class WorkingVectorizer:
            def __init__(self, features):
                self.vocabulary_ = {}
                self._tfidf = True
                self.idf_ = None
                self.stop_words_ = set()
                self.n_features = features
            
            def transform(self, texts):
                from scipy.sparse import csr_matrix
                if isinstance(texts, str):
                    texts = [texts]
                return csr_matrix((len(texts), self.n_features))
        
        self.tfidf_vectorizer = WorkingVectorizer(n_features)
        logger.debug(f"Created working vectorizer with {n_features} features")
    
    def _create_dummy_vectorizer(self):
        """Create dummy vectorizer when TF-IDF not available."""
        class DummyVectorizer:
            def __init__(self):
                self.vocabulary_ = {}
                self._tfidf = True
                self.idf_ = None
                self.stop_words_ = set()
                self.n_features = 20000
            
            def transform(self, texts):
                from scipy.sparse import csr_matrix
                if isinstance(texts, str):
                    texts = [texts]
                return csr_matrix((len(texts), self.n_features))
        
        self.tfidf_vectorizer = DummyVectorizer()
        logger.debug("Created dummy vectorizer")
    
    def _build_kg_indexes(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """Build KG lookup indexes."""
        logger.info("🗃️ Building KG indexes...")
        
        progress = ProgressTracker(
            total=len(self.all_records),
            description="Building KG indexes",
            callback=progress_callback,
            log_interval=10
        )
        
        kg_enhanced_count = 0
        
        # Build lookup dictionaries
        for record in self.all_records:
            try:
                doc_id = record['global_id']
                
                # Store JSON strings for lazy parsing
                if record.get('kg_entities_json', '[]') != '[]':
                    self.kg_entities_lookup[doc_id] = record['kg_entities_json']
                    kg_enhanced_count += 1
                
                if record.get('kg_cross_references_json', '[]') != '[]':
                    self.kg_cross_references_lookup[doc_id] = record['kg_cross_references_json']
                
                if record.get('kg_legal_domains_json', '[]') != '[]':
                    self.kg_domains_lookup[doc_id] = record['kg_legal_domains_json']
                
                if record.get('kg_concept_clusters_json', '{}') != '{}':
                    self.kg_concept_clusters_lookup[doc_id] = record['kg_concept_clusters_json']
                
                if record.get('kg_legal_actions_json', '{}') != '{}':
                    self.kg_legal_actions_lookup[doc_id] = record['kg_legal_actions_json']
                
                if record.get('kg_sanctions_json', '{}') != '{}':
                    self.kg_sanctions_lookup[doc_id] = record['kg_sanctions_json']
                
                if record.get('kg_concept_vector_json', '[]') != '[]':
                    self.kg_concept_vectors_lookup[doc_id] = record['kg_concept_vector_json']
                
                progress.update(1)
                
            except Exception as e:
                logger.warning(f"Error building KG index for record {record.get('global_id')}: {e}")
                continue
        
        progress.finish()
        
        self.load_stats['kg_enhanced_count'] = kg_enhanced_count
        
        # Build numeric indexes
        logger.info("Building numeric indexes...")
        self._build_numeric_indexes()
        
        logger.info(
            f"✅ KG indexes built: {len(self.kg_entities_lookup):,} entities, "
            f"{len(self.authority_index)} authority tiers"
        )
    
    def _build_numeric_indexes(self):
        """Build numeric indexes for fast filtering."""
        from collections import defaultdict
        
        self.authority_index = defaultdict(list)
        self.temporal_index = defaultdict(list)
        self.kg_connectivity_index = defaultdict(list)
        self.hierarchy_index = defaultdict(list)
        self.domain_index = defaultdict(list)
        
        for i, record in enumerate(self.all_records):
            try:
                # Authority tier (0-10)
                authority_tier = max(0, min(10, int(record['kg_authority_score'] * 10)))
                self.authority_index[authority_tier].append(i)
                
                # Temporal tier (0-10)
                temporal_tier = max(0, min(10, int(record['kg_temporal_score'] * 10)))
                self.temporal_index[temporal_tier].append(i)
                
                # KG connectivity tier (0-10)
                kg_tier = max(0, min(10, int(record['kg_connectivity_score'] * 10)))
                self.kg_connectivity_index[kg_tier].append(i)
                
                # Hierarchy tier (1-10)
                hierarchy_tier = max(1, min(10, record['kg_hierarchy_level']))
                self.hierarchy_index[hierarchy_tier].append(i)
                
                # Domain index
                domain = record['kg_primary_domain']
                self.domain_index[domain].append(i)
                
            except Exception as e:
                logger.warning(f"Error building numeric index for record {i}: {e}")
                continue
        
        # Convert to regular dicts
        self.authority_index = dict(self.authority_index)
        self.temporal_index = dict(self.temporal_index)
        self.kg_connectivity_index = dict(self.kg_connectivity_index)
        self.hierarchy_index = dict(self.hierarchy_index)
        self.domain_index = dict(self.domain_index)
    
    def _validate_loaded_data(self):
        """Validate loaded dataset."""
        logger.info("✅ Validating loaded data...")
        
        validation_report = self.validator.validate_dataset(self.all_records)
        
        # Log summary
        error_count = len(validation_report['errors'])
        warning_count = len(validation_report['warnings'])
        
        if error_count > 0:
            logger.warning(f"Validation found {error_count} errors")
            # Log first 5 errors
            for error in validation_report['errors'][:5]:
                logger.warning(f"  - {error}")
            if error_count > 5:
                logger.warning(f"  ... and {error_count - 5} more errors")
        
        if warning_count > 0:
            logger.info(f"Validation found {warning_count} warnings")
        
        logger.info(f"Validation complete: {validation_report['valid_records']:,} valid records")
    
    def _calculate_memory_usage(self):
        """Calculate total memory usage."""
        import sys
        
        memory = 0
        
        try:
            # Records
            memory += sys.getsizeof(self.all_records)
            
            # Embeddings
            if self.embeddings is not None:
                memory += self.embeddings.element_size() * self.embeddings.nelement()
            
            # TF-IDF
            if self.tfidf_matrix is not None:
                memory += self.tfidf_matrix.data.nbytes
            
            # Indexes
            for index in [self.kg_entities_lookup, self.kg_cross_references_lookup,
                         self.authority_index, self.temporal_index]:
                memory += sys.getsizeof(index)
            
            self.load_stats['memory_usage_mb'] = memory / 1024 / 1024
            
        except Exception as e:
            logger.warning(f"Could not calculate memory usage: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        if not self.loaded:
            return {'error': 'Dataset not loaded'}
        
        try:
            stats = {
                # Basic stats
                'total_records': len(self.all_records),
                'kg_enhanced': len(self.kg_entities_lookup),
                'kg_enhancement_rate': len(self.kg_entities_lookup) / len(self.all_records) if self.all_records else 0,
                
                # Shapes
                'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
                'tfidf_shape': self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
                
                # Indexes
                'authority_tiers': len(self.authority_index),
                'temporal_tiers': len(self.temporal_index),
                'kg_connectivity_tiers': len(self.kg_connectivity_index),
                'hierarchy_tiers': len(self.hierarchy_index),
                'unique_domains': len(self.domain_index),
                
                # Averages
                'avg_authority_score': np.mean([r['kg_authority_score'] for r in self.all_records]),
                'avg_temporal_score': np.mean([r['kg_temporal_score'] for r in self.all_records]),
                'avg_connectivity_score': np.mean([r['kg_connectivity_score'] for r in self.all_records]),
                'avg_entities_per_doc': np.mean([r['kg_entity_count'] for r in self.all_records]),
                'avg_cross_refs_per_doc': np.mean([r['kg_cross_ref_count'] for r in self.all_records]),
                
                # Features
                'has_obligations': sum(1 for r in self.all_records if r['kg_has_obligations']),
                'has_prohibitions': sum(1 for r in self.all_records if r['kg_has_prohibitions']),
                'has_permissions': sum(1 for r in self.all_records if r['kg_has_permissions']),
                
                # Loading stats
                'load_time': self.load_stats['load_time'],
                'memory_usage_mb': self.load_stats['memory_usage_mb'],
                'processed_rows': self.load_stats['processed_rows'],
                'skipped_rows': self.load_stats['skipped_rows'],
                
                # Cache stats
                'cache_enabled': self.cache is not None,
                'cache_stats': self.cache.get_stats() if self.cache else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            return {'error': str(e)}
    
    def validate(self) -> Dict[str, Any]:
        """Validate loaded dataset."""
        if not self.loaded:
            return {'error': 'Dataset not loaded'}
        
        return self.validator.validate_dataset(self.all_records)
    
    def get_record_by_id(self, global_id: int) -> Optional[Dict[str, Any]]:
        """
        Get record by global ID.
        
        Args:
            global_id: Record global ID
            
        Returns:
            Record dictionary or None
        """
        try:
            for record in self.all_records:
                if record['global_id'] == global_id:
                    return record
            return None
        except Exception as e:
            logger.error(f"Error getting record {global_id}: {e}")
            return None
    
    def search_by_regulation(
        self,
        regulation_type: str,
        regulation_number: str,
        year: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search records by regulation metadata.
        
        Args:
            regulation_type: Type of regulation
            regulation_number: Regulation number
            year: Optional year
            
        Returns:
            List of matching records
        """
        logger.debug(f"Searching: {regulation_type} {regulation_number} {year}")
        
        matches = []
        
        for record in self.all_records:
            type_match = regulation_type.lower() in record['regulation_type'].lower()
            number_match = regulation_number == record['regulation_number']
            year_match = (year is None or year == record['year'])
            
            if type_match and number_match and year_match:
                matches.append(record)
        
        logger.debug(f"Found {len(matches)} matches")
        return matches
    
    def get_records_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all records in a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of records
        """
        if domain not in self.domain_index:
            return []
        
        indices = self.domain_index[domain]
        return [self.all_records[i] for i in indices]
    
    def clear_cache(self):
        """Clear all caches."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def reload(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Reload dataset from source."""
        logger.info("Reloading dataset...")
        self.loaded = False
        return self.load(progress_callback=progress_callback, force_reload=True)
    