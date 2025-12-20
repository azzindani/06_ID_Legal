"""
FAISS Index Manager - High-Performance Semantic Search for Million-Row Datasets

Provides approximate nearest neighbor (ANN) search using FAISS, enabling:
- 10-100x speedup over linear search
- Efficient handling of 1M+ document datasets
- Index persistence (save/load)
- Incremental indexing (add documents without full rebuild)
- GPU acceleration support

Performance characteristics:
- Linear search: O(n) - ~1000ms for 1M docs
- FAISS IndexIVFFlat: O(log n) - ~10-50ms for 1M docs
- FAISS IndexHNSW: O(log n) - ~5-20ms for 1M docs
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle
import time
from utils.logger_utils import get_logger

logger = get_logger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library found - high-performance indexing enabled")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - falling back to linear search. Install with: pip install faiss-cpu (or faiss-gpu)")


class FAISSIndexManager:
    """
    Manages FAISS index for high-performance semantic search

    Supports multiple index types:
    - IndexFlatIP: Exact search (baseline, no approximation)
    - IndexIVFFlat: Fast approximate search (recommended for 100k-10M docs)
    - IndexHNSW: Very fast graph-based search (recommended for <1M docs)

    Usage:
        # Initialize
        index_manager = FAISSIndexManager(embedding_dim=768, index_type='IVF')

        # Build index from embeddings
        index_manager.build_index(embeddings)

        # Search
        scores, indices = index_manager.search(query_embedding, top_k=50)

        # Save/load
        index_manager.save_index('path/to/index.faiss')
        index_manager.load_index('path/to/index.faiss')

        # Incremental updates
        index_manager.add_embeddings(new_embeddings, start_id=10000)
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = 'IVF',
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        """
        Initialize FAISS index manager

        Args:
            embedding_dim: Dimension of embeddings (e.g., 768 for BERT, 1536 for text-embedding-ada)
            index_type: Type of index ('Flat', 'IVF', 'HNSW')
                - 'Flat': Exact search, slow for large datasets
                - 'IVF': Inverted file index, good balance (recommended)
                - 'HNSW': Hierarchical graph, very fast but more memory
            nlist: Number of clusters for IVF (more = faster but less accurate)
            nprobe: Number of clusters to search in IVF (more = slower but more accurate)
            use_gpu: Use GPU for indexing (requires faiss-gpu)
            gpu_id: GPU device ID
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        self.index = None
        self.is_trained = False
        self.num_vectors = 0

        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - index will not be created")
            return

        # Create index
        self._create_index()

        logger.info(f"FAISSIndexManager initialized", {
            "embedding_dim": embedding_dim,
            "index_type": index_type,
            "nlist": nlist,
            "nprobe": nprobe,
            "use_gpu": use_gpu,
            "faiss_available": FAISS_AVAILABLE
        })

    def _create_index(self):
        """Create FAISS index based on configuration"""
        if not FAISS_AVAILABLE:
            return

        if self.index_type == 'Flat':
            # Exact search using inner product (for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.is_trained = True  # Flat index doesn't need training

        elif self.index_type == 'IVF':
            # Inverted file index with flat storage
            # Good for 100k-10M documents
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = self.nprobe  # Number of clusters to search

        elif self.index_type == 'HNSW':
            # Hierarchical Navigable Small World graph
            # Very fast but uses more memory
            # Good for <1M documents
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
            self.is_trained = True  # HNSW doesn't need training

        else:
            raise ValueError(f"Unknown index type: {self.index_type}. Use 'Flat', 'IVF', or 'HNSW'")

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
                logger.info(f"Index moved to GPU {self.gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}. Using CPU.")

    def build_index(self, embeddings: np.ndarray, normalize: bool = True) -> float:
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Numpy array of shape (n_docs, embedding_dim)
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)

        Returns:
            Time taken to build index (seconds)
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - cannot build index")
            return 0.0

        start_time = time.time()

        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Normalize if requested (for cosine similarity using inner product)
        if normalize:
            faiss.normalize_L2(embeddings)

        # Train index if needed (only for IVF)
        if not self.is_trained:
            logger.info(f"Training FAISS index on {len(embeddings)} vectors...")
            # For IVF, use subset for training if dataset is large
            train_size = min(len(embeddings), max(self.nlist * 50, 10000))
            train_indices = np.random.choice(len(embeddings), train_size, replace=False)
            train_embeddings = embeddings[train_indices]

            self.index.train(train_embeddings)
            self.is_trained = True
            logger.info("Index training completed")

        # Add all vectors to index
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        self.index.add(embeddings)
        self.num_vectors = self.index.ntotal

        build_time = time.time() - start_time

        logger.info(f"FAISS index built successfully", {
            "num_vectors": self.num_vectors,
            "index_type": self.index_type,
            "build_time_sec": f"{build_time:.2f}",
            "vectors_per_sec": f"{len(embeddings)/build_time:.0f}"
        })

        return build_time

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors

        Args:
            query_embedding: Query vector of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
            normalize: Whether to normalize query (should match index normalization)

        Returns:
            Tuple of (scores, indices)
            - scores: Similarity scores (higher = more similar)
            - indices: Document indices
        """
        if not FAISS_AVAILABLE or self.index is None:
            logger.warning("FAISS index not available - returning empty results")
            return np.zeros(0), np.array([], dtype=np.int64)

        if self.num_vectors == 0:
            logger.warning("Index is empty - no vectors added")
            return np.zeros(0), np.array([], dtype=np.int64)

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Normalize if requested
        if normalize:
            faiss.normalize_L2(query_embedding)

        # Search
        top_k = min(top_k, self.num_vectors)  # Can't return more than we have
        scores, indices = self.index.search(query_embedding, top_k)

        # Return first row (we only searched with 1 query)
        return scores[0], indices[0]

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        normalize: bool = True,
        start_id: Optional[int] = None
    ):
        """
        Add new embeddings to existing index (incremental indexing)

        Args:
            embeddings: New embeddings to add
            normalize: Whether to normalize
            start_id: Starting ID for new vectors (for tracking)

        Note: For IVF index, adding vectors after training may reduce quality.
        Consider rebuilding index periodically.
        """
        if not FAISS_AVAILABLE or self.index is None:
            logger.warning("FAISS index not available")
            return

        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Normalize if requested
        if normalize:
            faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)
        self.num_vectors = self.index.ntotal

        logger.info(f"Added {len(embeddings)} vectors to index", {
            "total_vectors": self.num_vectors,
            "start_id": start_id
        })

    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        if not FAISS_AVAILABLE or self.index is None:
            logger.warning("No index to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Move to CPU if on GPU
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        # Save index
        faiss.write_index(index_to_save, str(filepath))

        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'num_vectors': self.num_vectors,
            'is_trained': self.is_trained
        }
        metadata_path = filepath.with_suffix('.meta')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"FAISS index saved", {
            "filepath": str(filepath),
            "num_vectors": self.num_vectors,
            "size_mb": f"{filepath.stat().st_size / 1024 / 1024:.2f}"
        })

    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - cannot load index")
            return

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")

        # Load metadata
        metadata_path = filepath.with_suffix('.meta')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
            self.nlist = metadata['nlist']
            self.nprobe = metadata['nprobe']
            self.num_vectors = metadata['num_vectors']
            self.is_trained = metadata['is_trained']

        # Load index
        self.index = faiss.read_index(str(filepath))

        # Set nprobe for IVF index
        if self.index_type == 'IVF' and hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
                logger.info(f"Loaded index moved to GPU {self.gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to move loaded index to GPU: {e}")

        logger.info(f"FAISS index loaded", {
            "filepath": str(filepath),
            "num_vectors": self.num_vectors,
            "index_type": self.index_type
        })

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not FAISS_AVAILABLE or self.index is None:
            return {"faiss_available": False}

        return {
            "faiss_available": True,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "num_vectors": self.num_vectors,
            "is_trained": self.is_trained,
            "nlist": self.nlist if self.index_type == 'IVF' else None,
            "nprobe": self.nprobe if self.index_type == 'IVF' else None,
            "use_gpu": self.use_gpu,
        }

    def clear_index(self):
        """Clear all vectors from index"""
        if FAISS_AVAILABLE and self.index is not None:
            self.index.reset()
            self.num_vectors = 0
            logger.info("Index cleared")


def choose_optimal_index_config(num_docs: int, embedding_dim: int) -> Dict[str, Any]:
    """
    Choose optimal FAISS index configuration based on dataset size

    Args:
        num_docs: Number of documents in dataset
        embedding_dim: Embedding dimension

    Returns:
        Dictionary with recommended configuration
    """
    if num_docs < 10000:
        # Small dataset - use exact search
        return {
            'index_type': 'Flat',
            'nlist': None,
            'nprobe': None,
            'description': 'Exact search (small dataset)'
        }

    elif num_docs < 100000:
        # Medium dataset - use HNSW for speed
        return {
            'index_type': 'HNSW',
            'nlist': None,
            'nprobe': None,
            'description': 'HNSW graph index (medium dataset, very fast)'
        }

    elif num_docs < 1000000:
        # Large dataset - use IVF with moderate parameters
        nlist = int(np.sqrt(num_docs))  # Rule of thumb
        return {
            'index_type': 'IVF',
            'nlist': nlist,
            'nprobe': min(20, nlist // 5),  # Search 20% of clusters
            'description': f'IVF index (large dataset, {nlist} clusters)'
        }

    else:
        # Very large dataset - use IVF with more clusters
        nlist = int(np.sqrt(num_docs))
        return {
            'index_type': 'IVF',
            'nlist': nlist,
            'nprobe': min(50, nlist // 10),  # Search 10% of clusters
            'description': f'IVF index (very large dataset, {nlist} clusters)'
        }
