"""
Iterative Expansion Engine - Detective-Style Document Retrieval

Implements multi-round expansion strategies to find related documents
beyond initial scoring-based retrieval.

Expansion Strategies (Phase 1):
1. Metadata Expansion: Find documents from same regulation

Future Strategies:
2. KG Expansion: Follow entity and citation networks
3. Citation Traversal: Multi-hop citation following
4. Semantic Clustering: Embedding space neighbors

Usage:
    engine = IterativeExpansionEngine(data_loader, config)
    pool = engine.expand(initial_documents, query)
    expanded_docs = pool.get_ranked_documents()
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict
from utils.logger_utils import get_logger
from .research_pool import ResearchPool


class IterativeExpansionEngine:
    """
    Controls multi-round document expansion

    Phase 1: Metadata expansion (same regulation context)
    """

    def __init__(self, data_loader, config: Dict[str, Any]):
        """
        Initialize expansion engine

        Args:
            data_loader: Data loader with all_records
            config: Expansion configuration dictionary
        """
        self.data_loader = data_loader
        self.config = config
        self.logger = get_logger("ExpansionEngine")

        # Extract configuration
        self.enabled = config.get('enable_expansion', False)
        self.max_rounds = config.get('max_expansion_rounds', 2)
        self.max_pool_size = config.get('max_pool_size', 1000)
        self.min_docs_per_round = config.get('min_docs_per_round', 5)
        self.seeds_per_round = config.get('seeds_per_round', 10)
        self.seed_score_threshold = config.get('seed_score_threshold', 0.50)

        # Strategy-specific config
        self.metadata_config = config.get('metadata_expansion', {
            'enabled': True,
            'max_docs_per_regulation': 50,
            'include_preamble': True,
            'include_attachments': True
        })

        self.logger.info(f"ExpansionEngine initialized (enabled={self.enabled}, max_rounds={self.max_rounds})")

    def expand(
        self,
        initial_documents: List[Dict[str, Any]],
        query: str
    ) -> ResearchPool:
        """
        Main expansion loop

        Args:
            initial_documents: Seed documents from initial retrieval
            query: Original query (for context, future use)

        Returns:
            ResearchPool with all expanded documents
        """
        if not self.enabled:
            self.logger.info("Expansion disabled, returning initial documents only")
            pool = ResearchPool(logger=self.logger)
            for doc in initial_documents:
                pool.add(doc, source='initial_retrieval', round_num=0,
                        score=doc.get('scores', {}).get('final', 0.0))
            return pool

        self.logger.info(f"Starting expansion with {len(initial_documents)} seed documents")

        # Initialize pool with initial documents
        pool = ResearchPool(logger=self.logger)

        for doc in initial_documents:
            # Extract score from nested structure if available
            if 'scores' in doc and isinstance(doc['scores'], dict):
                score = doc['scores'].get('final', 0.0)
            else:
                score = doc.get('score', 0.0)

            pool.add(doc, source='initial_retrieval', round_num=0, score=score)

        # Expansion rounds
        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"Expansion round {round_num}/{self.max_rounds}")

            # Select seeds for this round
            if round_num == 1:
                # Use top initial documents as seeds
                seeds = pool.get_seeds_for_expansion(
                    top_k=self.seeds_per_round,
                    min_score=self.seed_score_threshold
                )
            else:
                # Use documents added in previous round
                seeds = pool.get_by_round(round_num - 1)

                # If previous round didn't add docs, use best from pool
                if not seeds:
                    seeds = pool.get_seeds_for_expansion(
                        top_k=self.seeds_per_round,
                        min_score=self.seed_score_threshold
                    )

            if not seeds:
                self.logger.info(f"No seeds available for round {round_num}, stopping expansion")
                break

            self.logger.info(f"Round {round_num}: Expanding from {len(seeds)} seeds")

            # Apply expansion strategies
            added_count = 0

            # Strategy 1: Metadata Expansion
            if self.metadata_config.get('enabled', True):
                for seed in seeds:
                    count = self._metadata_expansion(seed, pool, round_num)
                    added_count += count

            self.logger.info(f"Round {round_num}: Added {added_count} new documents (pool size: {len(pool)})")

            # Check stop conditions
            if added_count < self.min_docs_per_round:
                self.logger.info(f"Convergence reached (added only {added_count} docs)")
                break

            if len(pool) >= self.max_pool_size:
                self.logger.warning(f"Max pool size reached ({len(pool)} >= {self.max_pool_size})")
                break

        # Log final statistics
        stats = pool.get_stats()
        self.logger.info(f"Expansion complete: {stats['total_documents']} total docs, "
                        f"{stats['duplicates_encountered']} duplicates filtered, "
                        f"sources: {stats['sources']}")

        return pool

    def _metadata_expansion(
        self,
        seed_doc: Dict[str, Any],
        pool: ResearchPool,
        round_num: int
    ) -> int:
        """
        Strategy 1: Metadata-based expansion

        When we find a relevant article, fetch the entire regulation:
        - All other articles/sections
        - Preamble (contains definitions, rationale)
        - Attachments (procedures, forms)

        Example:
            Seed: UU 6/1983 Pasal 25 (tax objections)
            Expand: Fetch ALL UU 6/1983 articles (Pasal 1-99)

        Args:
            seed_doc: Document to expand from
            pool: Research pool to add results
            round_num: Current expansion round

        Returns:
            Number of documents added
        """
        added_count = 0

        # Extract regulation identifiers
        regulation_type = seed_doc.get('regulation_type')
        regulation_number = seed_doc.get('regulation_number')
        regulation_year = seed_doc.get('year')

        if not regulation_type or not regulation_number:
            self.logger.debug(f"Seed doc {seed_doc.get('global_id')} missing regulation metadata, skipping")
            return 0

        self.logger.debug(f"Metadata expansion for {regulation_type} {regulation_number}/{regulation_year}")

        # Find all documents from same regulation
        max_docs = self.metadata_config.get('max_docs_per_regulation', 50)
        found_count = 0

        for doc in self.data_loader.all_records:
            # Check if already in pool
            if pool.contains(doc.get('global_id')):
                continue

            # Match regulation
            if (doc.get('regulation_type') == regulation_type and
                doc.get('regulation_number') == regulation_number and
                doc.get('year') == regulation_year):

                # Add to pool
                added = pool.add(
                    doc,
                    source='metadata_expansion',
                    seed_id=seed_doc.get('global_id'),
                    round_num=round_num,
                    score=None  # No score for expanded docs (will be scored later if needed)
                )

                if added:
                    added_count += 1
                    found_count += 1

                # Limit docs per regulation
                if found_count >= max_docs:
                    self.logger.debug(f"Reached max docs limit ({max_docs}) for regulation")
                    break

        self.logger.debug(f"Metadata expansion: Added {added_count} docs from {regulation_type} {regulation_number}")

        return added_count

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            'enabled': self.enabled,
            'max_rounds': self.max_rounds,
            'max_pool_size': self.max_pool_size,
            'min_docs_per_round': self.min_docs_per_round,
            'seeds_per_round': self.seeds_per_round,
            'seed_score_threshold': self.seed_score_threshold,
            'metadata_expansion': self.metadata_config
        }
