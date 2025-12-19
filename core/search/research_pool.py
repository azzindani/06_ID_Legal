"""
Research Pool - Central Document Memory for Iterative Retrieval

This module implements a document pool that tracks:
- All retrieved documents (deduplicated)
- How each document was found (provenance)
- Relationships between documents (graph)
- Expansion statistics

Usage:
    pool = ResearchPool()
    pool.add(doc, source='initial_retrieval', score=0.85)
    pool.add(doc2, source='metadata_expansion', seed=doc['global_id'])

    all_docs = pool.get_all_documents()
    stats = pool.get_stats()
"""

from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from utils.logger_utils import get_logger
import time


class ResearchPool:
    """
    Central memory for iterative document retrieval

    Features:
    - Deduplication by global_id
    - Provenance tracking (how/why doc was added)
    - Source statistics
    - Scoring and ranking
    """

    def __init__(self, logger=None):
        """Initialize empty research pool"""
        self.documents: Dict[str, Dict[str, Any]] = {}  # global_id → document
        self.provenance: Dict[str, Dict[str, Any]] = {}  # global_id → provenance info
        self.stats = defaultdict(int)
        self.logger = logger or get_logger("ResearchPool")

        self.stats['created_at'] = time.time()

    def add(
        self,
        doc: Dict[str, Any],
        source: str = 'unknown',
        seed_id: Optional[str] = None,
        round_num: int = 0,
        score: Optional[float] = None
    ) -> bool:
        """
        Add document to pool with provenance tracking

        Args:
            doc: Document dictionary (must have 'global_id')
            source: How this doc was found ('initial_retrieval', 'metadata_expansion', etc.)
            seed_id: global_id of seed document that led to this doc
            round_num: Expansion round number (0 = initial)
            score: Relevance score (if available)

        Returns:
            True if added (new), False if duplicate
        """
        doc_id = doc.get('global_id')

        if not doc_id:
            self.logger.warning(f"Document missing global_id, cannot add to pool")
            return False

        # Check for duplicate
        if doc_id in self.documents:
            # Update provenance if this is a better source
            existing_prov = self.provenance[doc_id]

            # Priority: initial > metadata > kg > other
            source_priority = {
                'initial_retrieval': 3,
                'metadata_expansion': 2,
                'kg_expansion': 1,
                'citation_expansion': 1,
                'semantic_expansion': 0
            }

            current_priority = source_priority.get(existing_prov['source'], 0)
            new_priority = source_priority.get(source, 0)

            if new_priority > current_priority:
                # Update to better source
                self.provenance[doc_id].update({
                    'source': source,
                    'seed_id': seed_id,
                    'round': round_num,
                    'updated_at': time.time()
                })
                self.stats['updates'] += 1

            self.stats['duplicates'] += 1
            return False  # Not added (duplicate)

        # Add new document
        self.documents[doc_id] = doc
        self.provenance[doc_id] = {
            'source': source,
            'seed_id': seed_id,
            'round': round_num,
            'score': score,
            'added_at': time.time()
        }

        # Update statistics
        self.stats[f'added_via_{source}'] += 1
        self.stats['total_documents'] += 1

        return True  # Added successfully

    def contains(self, doc_id: str) -> bool:
        """Check if document is already in pool"""
        return doc_id in self.documents

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get specific document by ID"""
        return self.documents.get(doc_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in pool (unordered)"""
        return list(self.documents.values())

    def get_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents added via specific source"""
        doc_ids = [
            doc_id for doc_id, prov in self.provenance.items()
            if prov['source'] == source
        ]
        return [self.documents[doc_id] for doc_id in doc_ids]

    def get_by_round(self, round_num: int) -> List[Dict[str, Any]]:
        """Get all documents added in specific round"""
        doc_ids = [
            doc_id for doc_id, prov in self.provenance.items()
            if prov['round'] == round_num
        ]
        return [self.documents[doc_id] for doc_id in doc_ids]

    def get_seeds_for_expansion(self, top_k: int = 10, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get best documents to use as expansion seeds

        Args:
            top_k: Number of seeds to return
            min_score: Minimum score threshold

        Returns:
            List of top-scoring documents
        """
        # Get documents with scores
        docs_with_scores = [
            (doc_id, self.provenance[doc_id].get('score', 0.0))
            for doc_id in self.documents
        ]

        # Filter by score threshold
        docs_with_scores = [
            (doc_id, score) for doc_id, score in docs_with_scores
            if score >= min_score
        ]

        # Sort by score descending
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-K
        top_doc_ids = [doc_id for doc_id, _ in docs_with_scores[:top_k]]
        return [self.documents[doc_id] for doc_id in top_doc_ids]

    def get_ranked_documents(self, diversity_filter: bool = False, max_per_source: int = 5) -> List[Dict[str, Any]]:
        """
        Get all documents ranked by score

        Args:
            diversity_filter: If True, limit docs per regulation
            max_per_source: Max documents from same regulation

        Returns:
            List of documents sorted by score
        """
        # Get all docs with scores
        docs_with_scores = [
            (doc_id, self.provenance[doc_id].get('score', 0.0))
            for doc_id in self.documents
        ]

        # Sort by score
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        if not diversity_filter:
            return [self.documents[doc_id] for doc_id, _ in docs_with_scores]

        # Apply diversity filtering
        result = []
        regulation_counts = defaultdict(int)

        for doc_id, score in docs_with_scores:
            doc = self.documents[doc_id]

            # Create regulation key
            reg_key = f"{doc.get('regulation_type', 'UNK')}_{doc.get('regulation_number', 'UNK')}"

            # Check if we've already added too many from this regulation
            if regulation_counts[reg_key] < max_per_source:
                result.append(doc)
                regulation_counts[reg_key] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics

        Returns:
            Dictionary with statistics
        """
        # Source breakdown
        sources = defaultdict(int)
        for prov in self.provenance.values():
            sources[prov['source']] += 1

        # Round breakdown
        rounds = defaultdict(int)
        for prov in self.provenance.values():
            rounds[prov['round']] += 1

        # Score statistics
        scores = [prov.get('score', 0.0) for prov in self.provenance.values() if prov.get('score')]

        stats_dict = {
            'total_documents': len(self.documents),
            'duplicates_encountered': self.stats.get('duplicates', 0),
            'updates': self.stats.get('updates', 0),
            'sources': dict(sources),
            'rounds': dict(rounds),
            'avg_score': sum(scores) / len(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'pool_age_seconds': time.time() - self.stats['created_at']
        }

        return stats_dict

    def clear(self):
        """Clear all documents and reset pool"""
        self.documents.clear()
        self.provenance.clear()
        self.stats.clear()
        self.stats['created_at'] = time.time()
        self.logger.info("Research pool cleared")

    def __len__(self) -> int:
        """Return number of documents in pool"""
        return len(self.documents)

    def __contains__(self, doc_id: str) -> bool:
        """Support 'in' operator for checking doc existence"""
        return doc_id in self.documents

    def __repr__(self) -> str:
        """String representation"""
        return f"ResearchPool(documents={len(self.documents)}, sources={len(set(p['source'] for p in self.provenance.values()))})"
