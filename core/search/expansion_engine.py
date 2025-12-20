"""
Iterative Expansion Engine - Detective-Style Document Retrieval

Implements multi-round expansion strategies to find related documents
beyond initial scoring-based retrieval.

Expansion Strategies:
1. Metadata Expansion (Phase 1): Find documents from same regulation
2. KG Expansion (Phase 2): Follow entity and citation networks
3. Citation Traversal (Phase 2): Multi-hop citation following
4. Semantic Clustering (Phase 3): Embedding space neighbors
5. Hybrid Adaptive (Phase 4): Query-type-specific strategy selection

Usage:
    engine = IterativeExpansionEngine(data_loader, config)
    pool = engine.expand(initial_documents, query)
    expanded_docs = pool.get_ranked_documents()
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from utils.logger_utils import get_logger
from utils.memory_utils import cleanup_memory, aggressive_cleanup
from .research_pool import ResearchPool
import numpy as np
import gc
import time

# Optional GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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

        # Phase 2: KG expansion config
        self.kg_config = config.get('kg_expansion', {
            'enabled': False,
            'max_entity_docs': 20,
            'entity_score_threshold': 0.3,
            'follow_citations': True,
            'citation_max_hops': 2
        })

        # Phase 2: Citation expansion config
        self.citation_config = config.get('citation_expansion', {
            'enabled': False,
            'max_hops': 2,
            'bidirectional': True
        })

        # Phase 3: Semantic clustering config
        self.semantic_config = config.get('semantic_expansion', {
            'enabled': False,
            'cluster_radius': 0.15,
            'min_cluster_size': 3,
            'max_neighbors': 30,
            'similarity_threshold': 0.70
        })

        # Phase 4: Hybrid adaptive config
        self.hybrid_config = config.get('hybrid_expansion', {
            'enabled': False,
            'adaptive_strategy': True,
            'query_type_detection': True,
            'strategy_weights': {
                'metadata': 0.4,
                'kg': 0.3,
                'citation': 0.2,
                'semantic': 0.1
            }
        })

        # Phase 5: Temporal expansion config (legal amendments/versions)
        self.temporal_config = config.get('temporal_expansion', {
            'enabled': True,  # Important for Indonesian law
            'max_years_range': 30,  # Look back 30 years for amendments
            'prioritize_recent': True,  # Newest versions rank higher
            'include_superseded': True  # Include old versions for context
        })

        # Phase 6: Hierarchical expansion config (UU → PP → Perpres)
        self.hierarchical_config = config.get('hierarchical_expansion', {
            'enabled': True,  # Critical for legal hierarchy
            'expand_up': True,  # Find parent regulations (PP → UU)
            'expand_down': True,  # Find implementing regulations (UU → PP)
            'max_hierarchy_distance': 2  # Max levels up/down
        })

        # Phase 7: Topical expansion config (same legal domain)
        self.topical_config = config.get('topical_expansion', {
            'enabled': True,  # Important for legal topic clustering
            'max_docs_per_topic': 20,  # Limit docs from same topic
            'domain_threshold': 0.7  # Minimum domain confidence
        })

        # Smart filtering config
        self.filtering_config = config.get('smart_filtering', {
            'enabled': True,
            'semantic_threshold': 0.60,  # Similarity to top docs
            'max_pool_size': 500,  # Maximum docs after filtering
            'diversity_weight': 0.3  # Balance relevance vs diversity
        })

        # Conversational mode config
        self.conversational_config = config.get('conversational_mode', {
            'enabled': True,
            'conservative_expansion': True,
            'max_expansion_rounds': 1,
            'max_pool_multiplier': 0.5
        })

        self.logger.info(f"ExpansionEngine initialized (enabled={self.enabled}, max_rounds={self.max_rounds}, "
                        f"metadata={self.metadata_config.get('enabled')}, "
                        f"kg={self.kg_config.get('enabled')}, "
                        f"citation={self.citation_config.get('enabled')}, "
                        f"semantic={self.semantic_config.get('enabled')}, "
                        f"hybrid={self.hybrid_config.get('enabled')}, "
                        f"temporal={self.temporal_config.get('enabled')}, "
                        f"hierarchical={self.hierarchical_config.get('enabled')}, "
                        f"topical={self.topical_config.get('enabled')}, "
                        f"conversational={self.conversational_config.get('enabled')})")

    def _detect_conversational_mode(self, query: str, initial_documents: List[Dict[str, Any]]) -> bool:
        """
        Detect if this is a conversational query (follow-up question)

        Indicators:
        - Short query length (< 50 chars suggests follow-up)
        - Pronouns/references (tersebut, ini, itu, dia)
        - Follow-up indicators (lalu, kemudian, terus, dan, bagaimana)
        - Few initial documents (< 20 suggests narrow/specific query)

        Args:
            query: User query
            initial_documents: Initial retrieval results

        Returns:
            True if conversational mode detected
        """
        if not self.conversational_config.get('enabled', True):
            return False

        query_lower = query.lower()

        # Check for conversational indicators
        conversational_indicators = [
            'tersebut', 'ini', 'itu', 'dia', 'mereka',  # Pronouns
            'lalu', 'kemudian', 'terus', 'selanjutnya',  # Follow-up
            'bagaimana dengan', 'apa yang', 'apakah',  # Questions
            'dan', 'serta', 'juga'  # Conjunctions
        ]

        has_indicators = sum(1 for ind in conversational_indicators if ind in query_lower)

        # Conversational if:
        # 1. Short query (< 50 chars) AND has indicators
        # 2. Multiple indicators (>= 2)
        # 3. Very few initial results (< 15)
        is_conversational = (
            (len(query) < 50 and has_indicators >= 1) or
            has_indicators >= 2 or
            len(initial_documents) < 15
        )

        if is_conversational:
            self.logger.info(f"Conversational mode detected (query_len={len(query)}, "
                           f"indicators={has_indicators}, init_docs={len(initial_documents)})")

        return is_conversational

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

        # Detect conversational mode and adjust settings
        is_conversational = self._detect_conversational_mode(query, initial_documents)
        self._is_conversational_mode = is_conversational  # Store for use in expansion strategies

        if is_conversational and self.conversational_config.get('conservative_expansion', True):
            # Use conservative settings for conversational mode
            original_max_rounds = self.max_rounds
            original_max_pool_size = self.max_pool_size
            original_filtering_max = self.filtering_config.get('max_pool_size', 500)

            # Reduce rounds and pool sizes
            self.max_rounds = min(self.max_rounds, self.conversational_config.get('max_expansion_rounds', 1))
            pool_multiplier = self.conversational_config.get('max_pool_multiplier', 0.5)
            self.max_pool_size = int(self.max_pool_size * pool_multiplier)
            self.filtering_config['max_pool_size'] = int(original_filtering_max * pool_multiplier)

            self.logger.info(f"Conversational mode: Conservative expansion enabled "
                           f"(rounds: {original_max_rounds}→{self.max_rounds}, "
                           f"pool: {original_max_pool_size}→{self.max_pool_size}, "
                           f"filter: {original_filtering_max}→{self.filtering_config['max_pool_size']})")

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

        # Safety: Track expansion start time and set timeout
        expansion_start_time = time.time()
        max_expansion_time = 300  # 5 minutes max

        # Safety: Track processed seed IDs to prevent re-processing loops
        processed_seed_ids = set()

        # Safety: Track stalled rounds (no progress)
        stalled_rounds = 0
        max_stalled_rounds = 2

        # Expansion rounds
        for round_num in range(1, self.max_rounds + 1):
            # Safety check: Timeout
            elapsed_time = time.time() - expansion_start_time
            if elapsed_time > max_expansion_time:
                self.logger.warning(f"Expansion timeout after {elapsed_time:.1f}s, stopping")
                break

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

            # Safety check: Filter out already-processed seeds to prevent loops
            current_seed_ids = {self._get_doc_id(s) for s in seeds if self._get_doc_id(s)}
            new_seed_ids = current_seed_ids - processed_seed_ids

            if not new_seed_ids:
                self.logger.warning(f"Round {round_num}: All seeds already processed, stopping to prevent loop")
                break

            # Update processed seeds
            processed_seed_ids.update(new_seed_ids)

            # Filter seeds to only new ones
            seeds = [s for s in seeds if self._get_doc_id(s) in new_seed_ids]

            self.logger.info(f"Round {round_num}: Expanding from {len(seeds)} seeds ({len(new_seed_ids)} new)")

            # Apply expansion strategies
            added_count = 0

            # Strategy 1: Metadata Expansion
            if self.metadata_config.get('enabled', True):
                for seed in seeds:
                    count = self._metadata_expansion(seed, pool, round_num)
                    added_count += count

            # Strategy 2: KG Expansion (Phase 2)
            if self.kg_config.get('enabled', False):
                for seed in seeds:
                    count = self._kg_expansion(seed, pool, round_num)
                    added_count += count

            # Strategy 3: Citation Expansion (Phase 2)
            if self.citation_config.get('enabled', False):
                for seed in seeds:
                    count = self._citation_expansion(seed, pool, round_num)
                    added_count += count

            # Strategy 4: Semantic Clustering (Phase 3)
            if self.semantic_config.get('enabled', False):
                for seed in seeds:
                    count = self._semantic_expansion(seed, pool, round_num)
                    added_count += count

            # Strategy 5: Hybrid Adaptive (Phase 4) - Runs after all strategies
            if self.hybrid_config.get('enabled', False) and self.hybrid_config.get('adaptive_strategy', True):
                # Apply query-adaptive strategy selection
                count = self._hybrid_adaptive_expansion(seeds, pool, round_num, query)
                added_count += count

            # Strategy 6: Temporal Expansion (Phase 5) - Legal amendments/versions
            if self.temporal_config.get('enabled', True):
                for seed in seeds:
                    count = self._temporal_expansion(seed, pool, round_num)
                    added_count += count

            # Strategy 7: Hierarchical Expansion (Phase 6) - Legal hierarchy
            if self.hierarchical_config.get('enabled', True):
                for seed in seeds:
                    count = self._hierarchical_expansion(seed, pool, round_num)
                    added_count += count

            # Strategy 8: Topical Expansion (Phase 7) - Same legal domain
            if self.topical_config.get('enabled', True):
                for seed in seeds:
                    count = self._topical_expansion(seed, pool, round_num)
                    added_count += count

            self.logger.info(f"Round {round_num}: Added {added_count} new documents (pool size: {len(pool)})")

            # Safety check: Detect stalled rounds (no progress)
            if added_count == 0:
                stalled_rounds += 1
                self.logger.warning(f"Stalled round {stalled_rounds}/{max_stalled_rounds} (no docs added)")
                if stalled_rounds >= max_stalled_rounds:
                    self.logger.warning(f"Stopping: {stalled_rounds} consecutive rounds with no progress")
                    break
            else:
                stalled_rounds = 0  # Reset on progress

            # Periodic memory cleanup after each round to prevent memory buildup
            if round_num % 1 == 0:  # Every round
                cleanup_memory(aggressive=False, reason=f"expansion round {round_num}")

            # Check stop conditions
            if added_count < self.min_docs_per_round:
                self.logger.info(f"Convergence reached (added only {added_count} docs)")
                break

            if len(pool) >= self.max_pool_size:
                self.logger.warning(f"Max pool size reached ({len(pool)} >= {self.max_pool_size})")
                break

        # Apply smart filtering before returning pool
        if self.filtering_config.get('enabled', True) and len(pool) > 0:
            old_pool_size = len(pool)
            filtered_pool = self._apply_smart_filtering(pool, initial_documents)

            # Explicit memory cleanup: Delete old unfiltered pool
            del pool
            pool = filtered_pool

            # Aggressive cleanup after filtering large pool
            aggressive_cleanup(f"after filtering ({old_pool_size} -> {len(pool)} docs)")

            self.logger.info(f"Memory cleanup: Released {old_pool_size - len(pool)} documents from memory")

        # Final aggressive cleanup after expansion completes
        aggressive_cleanup("expansion complete")

        # Log final statistics
        stats = pool.get_stats()
        self.logger.info(f"Expansion complete: {stats['total_documents']} total docs, "
                        f"{stats['duplicates_encountered']} duplicates filtered, "
                        f"sources: {stats['sources']}")

        return pool

    def _get_doc_id(self, doc: Dict[str, Any]) -> Optional[str]:
        """
        Extract global_id from document (handles both top-level and nested metadata)

        Args:
            doc: Document dictionary

        Returns:
            global_id string or None
        """
        # Try top-level first
        doc_id = doc.get('global_id')

        # Try metadata
        if not doc_id and 'metadata' in doc:
            doc_id = doc['metadata'].get('global_id')

        return doc_id

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

        # Extract regulation identifiers (check both top-level and metadata)
        regulation_type = seed_doc.get('regulation_type')
        regulation_number = seed_doc.get('regulation_number')
        regulation_year = seed_doc.get('year')

        # Fallback to metadata dict if not in top-level
        if not regulation_type or not regulation_number:
            metadata = seed_doc.get('metadata', {})
            regulation_type = regulation_type or metadata.get('regulation_type')
            regulation_number = regulation_number or metadata.get('regulation_number')
            regulation_year = regulation_year or metadata.get('year')

        if not regulation_type or not regulation_number:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} missing regulation metadata, skipping")
            return 0

        self.logger.debug(f"Metadata expansion for {regulation_type} {regulation_number}/{regulation_year}")

        # Find all documents from same regulation
        max_docs = self.metadata_config.get('max_docs_per_regulation', 50)
        found_count = 0

        for doc in self.data_loader.all_records:
            # Check if already in pool
            doc_id = self._get_doc_id(doc)

            if not doc_id or pool.contains(doc_id):
                continue

            # Match regulation
            if (doc.get('regulation_type') == regulation_type and
                doc.get('regulation_number') == regulation_number and
                doc.get('year') == regulation_year):

                # Add to pool
                added = pool.add(
                    doc,
                    source='metadata_expansion',
                    seed_id=self._get_doc_id(seed_doc),
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

    def _kg_expansion(
        self,
        seed_doc: Dict[str, Any],
        pool: ResearchPool,
        round_num: int
    ) -> int:
        """
        Strategy 2: Knowledge Graph Expansion (Phase 2)

        Follow entity relationships and citations:
        1. Find docs mentioning same entities (co-occurrence)
        2. Follow citation relationships (cited regulations)
        3. Extract cross-references

        Example:
            Seed: Doc mentions "Pengadilan Pajak" entity
            Expand: Find ALL docs mentioning "Pengadilan Pajak"
            Also: Follow citations to related regulations

        Args:
            seed_doc: Document to expand from
            pool: Research pool to add results
            round_num: Current expansion round

        Returns:
            Number of documents added
        """
        added_count = 0

        # Extract KG entities from seed
        kg_entities = seed_doc.get('kg_entities', [])
        kg_citations = seed_doc.get('kg_citations', [])
        kg_cross_references = seed_doc.get('kg_cross_references', [])

        if not kg_entities and not kg_citations and not kg_cross_references:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} has no KG data, skipping")
            return 0

        self.logger.debug(f"KG expansion for {len(kg_entities)} entities, {len(kg_citations)} citations")

        max_entity_docs = self.kg_config.get('max_entity_docs', 20)

        # 1. Entity co-occurrence expansion
        for entity in kg_entities[:5]:  # Limit to top 5 entities
            entity_lower = entity.lower() if isinstance(entity, str) else str(entity).lower()
            found_count = 0

            for doc in self.data_loader.all_records:
                doc_id = self._get_doc_id(doc)

                if not doc_id or pool.contains(doc_id):
                    continue

                # Check if doc mentions this entity
                doc_entities = doc.get('kg_entities', [])
                doc_entities_lower = [e.lower() if isinstance(e, str) else str(e).lower() for e in doc_entities]

                if entity_lower in doc_entities_lower:
                    added = pool.add(
                        doc,
                        source='kg_expansion',
                        seed_id=self._get_doc_id(seed_doc),
                        round_num=round_num,
                        score=None
                    )

                    if added:
                        added_count += 1
                        found_count += 1

                    if found_count >= max_entity_docs:
                        break

        # 2. Citation following (if enabled)
        if self.kg_config.get('follow_citations', True):
            for citation in kg_citations[:10]:  # Limit to 10 citations
                # Parse citation (format may vary)
                citation_parts = self._parse_citation(citation)

                if not citation_parts:
                    continue

                for doc in self.data_loader.all_records:
                    doc_id = self._get_doc_id(doc)

                    if not doc_id or pool.contains(doc_id):
                        continue

                    # Match citation to document
                    if self._citation_matches_doc(citation_parts, doc):
                        added = pool.add(
                            doc,
                            source='kg_expansion',
                            seed_id=self._get_doc_id(seed_doc),
                            round_num=round_num,
                            score=None
                        )

                        if added:
                            added_count += 1

        self.logger.debug(f"KG expansion: Added {added_count} docs via entity/citation links")

        return added_count

    def _citation_expansion(
        self,
        seed_doc: Dict[str, Any],
        pool: ResearchPool,
        round_num: int
    ) -> int:
        """
        Strategy 3: Citation Network Traversal (Phase 2)

        Multi-hop citation following:
        - Find regulations cited BY seed document
        - Find regulations that CITE seed document (bidirectional)
        - Can traverse multiple hops

        Example:
            Seed: UU KUP cites UU Pengadilan Pajak
            Hop 1: Fetch UU Pengadilan Pajak
            Hop 2: Find what UU Pengadilan Pajak cites

        Args:
            seed_doc: Document to expand from
            pool: Research pool to add results
            round_num: Current expansion round

        Returns:
            Number of documents added
        """
        added_count = 0

        # Extract citations from seed
        kg_citations = seed_doc.get('kg_citations', [])

        if not kg_citations:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} has no citations, skipping")
            return 0

        max_hops = self.citation_config.get('max_hops', 2)
        bidirectional = self.citation_config.get('bidirectional', True)

        self.logger.debug(f"Citation expansion for {len(kg_citations)} citations (max_hops={max_hops})")

        # Multi-hop traversal using BFS
        visited = set()
        queue = [(seed_doc, 0)]  # (doc, hop_level)

        while queue:
            current_doc, hop = queue.pop(0)
            doc_id = self._get_doc_id(current_doc)

            if doc_id in visited or hop >= max_hops:
                continue

            visited.add(doc_id)

            # Get citations from current doc
            current_citations = current_doc.get('kg_citations', [])

            for citation in current_citations:
                citation_parts = self._parse_citation(citation)

                if not citation_parts:
                    continue

                # Find matching documents
                for doc in self.data_loader.all_records:
                    doc_id = self._get_doc_id(doc)

                    if not doc_id or pool.contains(doc_id):
                        continue

                    if self._citation_matches_doc(citation_parts, doc):
                        added = pool.add(
                            doc,
                            source='citation_expansion',
                            seed_id=doc_id,
                            round_num=round_num,
                            score=None
                        )

                        if added:
                            added_count += 1

                            # Add to queue for next hop
                            if hop + 1 < max_hops:
                                queue.append((doc, hop + 1))

        # Bidirectional: Find docs that cite this one
        if bidirectional:
            seed_reg_type = seed_doc.get('regulation_type')
            seed_reg_number = seed_doc.get('regulation_number')
            seed_year = seed_doc.get('year')

            # Look for docs that cite this regulation
            for doc in self.data_loader.all_records:
                doc_id = self._get_doc_id(doc)

                if not doc_id or pool.contains(doc_id):
                    continue

                doc_citations = doc.get('kg_citations', [])

                for citation in doc_citations:
                    # Check if citation matches seed document
                    if self._citation_text_matches(citation, seed_reg_type, seed_reg_number, seed_year):
                        added = pool.add(
                            doc,
                            source='citation_expansion',
                            seed_id=self._get_doc_id(seed_doc),
                            round_num=round_num,
                            score=None
                        )

                        if added:
                            added_count += 1

        self.logger.debug(f"Citation expansion: Added {added_count} docs via citation network")

        return added_count

    def _parse_citation(self, citation: str) -> Optional[Dict[str, str]]:
        """
        Parse citation string into components

        Examples:
            "UU No. 6 Tahun 1983" → {type: UU, number: 6, year: 1983}
            "PP 80/2007" → {type: PP, number: 80, year: 2007}
        """
        if not citation or not isinstance(citation, str):
            return None

        import re

        # Pattern 1: "UU No. X Tahun YYYY" or "UU Nomor X Tahun YYYY"
        pattern1 = r'(UU|PP|PERPRES|PERATURAN PEMERINTAH|UNDANG-UNDANG).*?(?:No\.|Nomor)\s*(\d+).*?(?:Tahun|tahun)\s*(\d{4})'
        match = re.search(pattern1, citation, re.IGNORECASE)

        if match:
            return {
                'type': match.group(1),
                'number': match.group(2),
                'year': match.group(3)
            }

        # Pattern 2: "UU X/YYYY" or "PP X/YYYY"
        pattern2 = r'(UU|PP|PERPRES)\s*(\d+)/(\d{4})'
        match = re.search(pattern2, citation, re.IGNORECASE)

        if match:
            return {
                'type': match.group(1),
                'number': match.group(2),
                'year': match.group(3)
            }

        return None

    def _citation_matches_doc(self, citation_parts: Dict[str, str], doc: Dict[str, Any]) -> bool:
        """Check if citation parts match document metadata"""
        if not citation_parts:
            return False

        doc_type = doc.get('regulation_type', '')
        doc_number = str(doc.get('regulation_number', ''))
        doc_year = str(doc.get('year', ''))

        # Normalize regulation type
        citation_type = citation_parts.get('type', '').upper()
        doc_type_upper = doc_type.upper()

        # Match type
        type_match = (
            citation_type in doc_type_upper or
            doc_type_upper.startswith(citation_type) or
            (citation_type == 'UU' and 'UNDANG-UNDANG' in doc_type_upper) or
            (citation_type == 'PP' and 'PERATURAN PEMERINTAH' in doc_type_upper)
        )

        if not type_match:
            return False

        # Match number and year
        return (
            citation_parts.get('number') == doc_number and
            citation_parts.get('year') == doc_year
        )

    def _citation_text_matches(self, citation_text: str, reg_type: str, reg_number: str, year: str) -> bool:
        """Check if citation text mentions the given regulation"""
        if not citation_text:
            return False

        citation_lower = citation_text.lower()
        reg_type_upper = reg_type.upper() if reg_type else ''

        # Check for number and year in citation
        if reg_number and year:
            return (str(reg_number) in citation_text and str(year) in citation_text)

        return False

    def _semantic_expansion(
        self,
        seed_doc: Dict[str, Any],
        pool: ResearchPool,
        round_num: int
    ) -> int:
        """
        Strategy 4: Semantic Clustering Expansion (Phase 3)

        Find documents in embedding space that are semantically similar:
        1. Calculate cosine similarity between seed and all docs
        2. Find neighbors within cluster radius
        3. Group by semantic similarity
        4. Expand to topic-related documents

        Example:
            Seed: Tax objection procedure doc
            Expand: Find docs about tax disputes, tax appeals, tax courts
            (semantically related but may not share exact keywords)

        Args:
            seed_doc: Document to expand from
            pool: Research pool to add results
            round_num: Current expansion round

        Returns:
            Number of documents added
        """
        added_count = 0

        # Get seed embedding
        seed_embedding = seed_doc.get('embedding')
        if seed_embedding is None:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} has no embedding, skipping")
            return 0

        # Convert to numpy array
        if isinstance(seed_embedding, list):
            seed_embedding = np.array(seed_embedding)

        # Normalize seed embedding
        seed_norm = np.linalg.norm(seed_embedding)
        if seed_norm == 0:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} has zero embedding, skipping")
            return 0

        seed_embedding_normalized = seed_embedding / seed_norm

        # Config parameters
        cluster_radius = self.semantic_config.get('cluster_radius', 0.15)
        max_neighbors = self.semantic_config.get('max_neighbors', 30)
        similarity_threshold = self.semantic_config.get('similarity_threshold', 0.70)

        # Find semantically similar documents
        similarities = []

        for doc in self.data_loader.all_records:
            doc_id = self._get_doc_id(doc)

            if not doc_id or pool.contains(doc_id):
                continue

            doc_embedding = doc.get('embedding')
            if doc_embedding is None:
                continue

            # Convert to numpy array
            if isinstance(doc_embedding, list):
                doc_embedding = np.array(doc_embedding)

            # Normalize doc embedding
            doc_norm = np.linalg.norm(doc_embedding)
            if doc_norm == 0:
                continue

            doc_embedding_normalized = doc_embedding / doc_norm

            # Calculate cosine similarity
            similarity = np.dot(seed_embedding_normalized, doc_embedding_normalized)

            # Filter by threshold
            if similarity >= similarity_threshold:
                similarities.append((doc, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Add top-K neighbors
        for doc, similarity in similarities[:max_neighbors]:
            added = pool.add(
                doc,
                source='semantic_expansion',
                seed_id=self._get_doc_id(seed_doc),
                round_num=round_num,
                score=float(similarity)
            )

            if added:
                added_count += 1

        self.logger.debug(f"Semantic expansion: Added {added_count} docs with similarity >= {similarity_threshold}")

        return added_count

    def _hybrid_adaptive_expansion(
        self,
        seeds: List[Dict[str, Any]],
        pool: ResearchPool,
        round_num: int,
        query: str
    ) -> int:
        """
        Strategy 5: Hybrid Adaptive Expansion (Phase 4)

        Intelligently select expansion strategies based on query characteristics:
        1. Detect query type (tax, labor, criminal, procedural, etc.)
        2. Select appropriate strategies for that query type
        3. Apply weighted combination of strategies
        4. Adaptive thresholds based on query complexity

        Example Query Types & Strategies:
            - Tax procedural query → Metadata + Citation (follow tax regulations)
            - Entity-focused query → KG + Semantic (find entity relationships)
            - Research query → All strategies (comprehensive expansion)
            - Simple lookup → Metadata only (fast, focused)

        Args:
            seeds: List of seed documents
            pool: Research pool to add results
            round_num: Current expansion round
            query: Original user query

        Returns:
            Number of documents added
        """
        added_count = 0

        # Detect query type
        query_type = self._detect_query_type(query)
        self.logger.debug(f"Detected query type: {query_type}")

        # Get strategy weights from config
        base_weights = self.hybrid_config.get('strategy_weights', {
            'metadata': 0.4,
            'kg': 0.3,
            'citation': 0.2,
            'semantic': 0.1
        })

        # Adaptive strategy selection based on query type
        if query_type == 'procedural':
            # Procedural queries: Focus on same regulation + citations
            strategies = ['metadata', 'citation']
            weights = {'metadata': 0.6, 'citation': 0.4}
        elif query_type == 'entity_focused':
            # Entity queries: Focus on KG + semantic
            strategies = ['kg', 'semantic']
            weights = {'kg': 0.6, 'semantic': 0.4}
        elif query_type == 'research':
            # Research queries: Use all strategies
            strategies = ['metadata', 'kg', 'citation', 'semantic']
            weights = base_weights
        elif query_type == 'citation_heavy':
            # Citation-heavy queries: Focus on citation networks
            strategies = ['citation', 'kg']
            weights = {'citation': 0.7, 'kg': 0.3}
        else:
            # Default: Balanced approach
            strategies = ['metadata', 'kg']
            weights = {'metadata': 0.6, 'kg': 0.4}

        self.logger.info(f"Hybrid adaptive: Selected strategies {strategies} for query type '{query_type}'")

        # Apply selected strategies to seeds
        for seed in seeds[:5]:  # Limit to top 5 seeds for adaptive expansion
            # Apply each strategy with weighting
            for strategy in strategies:
                if strategy == 'metadata' and self.metadata_config.get('enabled', True):
                    count = self._metadata_expansion(seed, pool, round_num)
                    added_count += int(count * weights.get('metadata', 1.0))

                elif strategy == 'kg' and self.kg_config.get('enabled', False):
                    count = self._kg_expansion(seed, pool, round_num)
                    added_count += int(count * weights.get('kg', 1.0))

                elif strategy == 'citation' and self.citation_config.get('enabled', False):
                    count = self._citation_expansion(seed, pool, round_num)
                    added_count += int(count * weights.get('citation', 1.0))

                elif strategy == 'semantic' and self.semantic_config.get('enabled', False):
                    count = self._semantic_expansion(seed, pool, round_num)
                    added_count += int(count * weights.get('semantic', 1.0))

        self.logger.debug(f"Hybrid adaptive expansion: Added {added_count} docs")

        return added_count

    def _detect_query_type(self, query: str) -> str:
        """
        Detect query type from query text

        Types:
        - procedural: About legal procedures (keberatan, banding, permohonan)
        - entity_focused: About specific entities (Pengadilan Pajak, Dirjen Pajak)
        - citation_heavy: Contains many regulation references
        - research: Complex, multi-faceted research question
        - simple: Simple fact lookup

        Args:
            query: User query string

        Returns:
            Query type string
        """
        query_lower = query.lower()

        # Procedural indicators
        procedural_keywords = [
            'prosedur', 'tata cara', 'keberatan', 'banding', 'gugatan',
            'permohonan', 'pengajuan', 'proses', 'mekanisme', 'cara'
        ]
        procedural_count = sum(1 for kw in procedural_keywords if kw in query_lower)

        # Entity indicators
        entity_keywords = [
            'pengadilan', 'dirjen', 'direktur jenderal', 'menteri',
            'gubernur', 'bupati', 'walikota', 'pejabat', 'instansi'
        ]
        entity_count = sum(1 for kw in entity_keywords if kw in query_lower)

        # Citation indicators (regulation mentions)
        citation_patterns = [
            'uu ', 'undang-undang', 'peraturan pemerintah', 'pp ', 'perpres',
            'tahun ', 'nomor ', 'no.', 'pasal '
        ]
        citation_count = sum(1 for pattern in citation_patterns if pattern in query_lower)

        # Research indicators
        research_keywords = [
            'analisis', 'bagaimana', 'mengapa', 'jelaskan', 'uraikan',
            'bandingkan', 'hubungan', 'perbedaan', 'persamaan'
        ]
        research_count = sum(1 for kw in research_keywords if kw in query_lower)

        # Determine query type based on counts
        if procedural_count >= 2:
            return 'procedural'
        elif entity_count >= 2:
            return 'entity_focused'
        elif citation_count >= 3:
            return 'citation_heavy'
        elif research_count >= 2 or len(query.split()) > 15:
            return 'research'
        else:
            return 'simple'

    def _temporal_expansion(
        self,
        seed_doc: Dict[str, Any],
        pool: ResearchPool,
        round_num: int
    ) -> int:
        """
        Strategy 6: Temporal Expansion (Phase 5) - Legal Amendments/Versions

        Find temporal relationships in Indonesian legal regulations:
        1. Amendments (perubahan): UU 6/1983 → UU 16/2000 → UU 28/2007
        2. Revisions (revisi): Later versions of same regulation
        3. Superseded regulations: Old versions replaced by new ones
        4. Related years: Same regulation type + number, different years

        Critical for Indonesian law where amendments create complex chains.

        Args:
            seed_doc: Document to expand from
            pool: Research pool to add results
            round_num: Current expansion round

        Returns:
            Number of documents added
        """
        added_count = 0

        # Extract regulation identifiers
        reg_type = seed_doc.get('regulation_type')
        reg_number = seed_doc.get('regulation_number')
        seed_year = seed_doc.get('year')

        if not reg_type or not reg_number or not seed_year:
            metadata = seed_doc.get('metadata', {})
            reg_type = reg_type or metadata.get('regulation_type')
            reg_number = reg_number or metadata.get('regulation_number')
            seed_year = seed_year or metadata.get('year')

        if not reg_type or not reg_number or not seed_year:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} missing temporal metadata, skipping")
            return 0

        try:
            seed_year_int = int(seed_year)
        except (ValueError, TypeError):
            self.logger.debug(f"Invalid year format: {seed_year}")
            return 0

        max_years_range = self.temporal_config.get('max_years_range', 30)
        prioritize_recent = self.temporal_config.get('prioritize_recent', True)
        include_superseded = self.temporal_config.get('include_superseded', True)

        self.logger.debug(f"Temporal expansion for {reg_type} {reg_number}/{seed_year}")

        # Find all versions of this regulation (same type + number, different years)
        temporal_docs = []

        for doc in self.data_loader.all_records:
            doc_id = self._get_doc_id(doc)

            if not doc_id or pool.contains(doc_id):
                continue

            doc_reg_type = doc.get('regulation_type')
            doc_reg_number = doc.get('regulation_number')
            doc_year = doc.get('year')

            # Match regulation type and number
            if doc_reg_type == reg_type and doc_reg_number == reg_number:
                try:
                    doc_year_int = int(doc_year)
                except (ValueError, TypeError):
                    continue

                # Check year range
                year_diff = abs(doc_year_int - seed_year_int)

                if year_diff <= max_years_range and doc_year != seed_year:
                    # Calculate temporal score (newer = higher if prioritize_recent)
                    if prioritize_recent:
                        # Newer versions get higher scores
                        temporal_score = 0.5 + (0.3 * (doc_year_int >= seed_year_int))
                    else:
                        # Equal weight to all versions
                        temporal_score = 0.5

                    # Penalize very old versions if not including superseded
                    if not include_superseded and doc_year_int < seed_year_int - 10:
                        continue

                    temporal_docs.append((doc, temporal_score, doc_year_int))

        # Sort by year (newest first if prioritize_recent)
        if prioritize_recent:
            temporal_docs.sort(key=lambda x: x[2], reverse=True)
        else:
            temporal_docs.sort(key=lambda x: x[2])

        # Add temporal documents to pool
        for doc, score, _ in temporal_docs[:10]:  # Limit to 10 versions
            added = pool.add(
                doc,
                source='temporal_expansion',
                seed_id=self._get_doc_id(seed_doc),
                round_num=round_num,
                score=score
            )

            if added:
                added_count += 1

        self.logger.debug(f"Temporal expansion: Added {added_count} temporal versions")

        # Cleanup temporary list
        del temporal_docs

        return added_count

    def _hierarchical_expansion(
        self,
        seed_doc: Dict[str, Any],
        pool: ResearchPool,
        round_num: int
    ) -> int:
        """
        Strategy 7: Hierarchical Expansion (Phase 6) - Legal Hierarchy

        Follow Indonesian legal hierarchy:
        1. UUD 1945 (Constitution) - Level 1
        2. UU (Undang-Undang / Law) - Level 2
        3. PP (Peraturan Pemerintah / Government Regulation) - Level 3
        4. PERPRES (Peraturan Presiden / Presidential Regulation) - Level 4
        5. Peraturan Menteri (Ministerial Regulation) - Level 5
        6. Perda (Peraturan Daerah / Local Regulation) - Level 6

        Expand both UP (implementing regs → parent law) and DOWN (law → implementing regs)

        Args:
            seed_doc: Document to expand from
            pool: Research pool to add results
            round_num: Current expansion round

        Returns:
            Number of documents added
        """
        added_count = 0

        # Extract hierarchy level
        seed_hierarchy_level = seed_doc.get('kg_hierarchy_level')
        seed_reg_type = seed_doc.get('regulation_type')
        seed_reg_number = seed_doc.get('regulation_number')
        seed_year = seed_doc.get('year')

        if not seed_hierarchy_level:
            # Try to infer from regulation type
            seed_hierarchy_level = self._infer_hierarchy_level(seed_reg_type)

        if not seed_hierarchy_level or not seed_reg_type:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} missing hierarchy data, skipping")
            return 0

        expand_up = self.hierarchical_config.get('expand_up', True)
        expand_down = self.hierarchical_config.get('expand_down', True)
        max_distance = self.hierarchical_config.get('max_hierarchy_distance', 1)
        max_docs_per_level = self.hierarchical_config.get('max_docs_per_level', 15)
        year_range = self.hierarchical_config.get('year_range', 5)

        # Apply even stricter limits if conversational mode is active
        if self.hierarchical_config.get('conservative_in_conversation', True):
            if hasattr(self, '_is_conversational_mode') and self._is_conversational_mode:
                max_docs_per_level = max(5, max_docs_per_level // 2)
                year_range = max(3, year_range - 2)
                self.logger.debug(f"Conversational mode: Using stricter hierarchical limits "
                               f"(docs_per_level={max_docs_per_level}, year_range=±{year_range})")

        self.logger.debug(f"Hierarchical expansion for level {seed_hierarchy_level} ({seed_reg_type}), "
                         f"max_distance={max_distance}, max_per_level={max_docs_per_level}, year_range=±{year_range}")

        # Track documents added per hierarchy level
        level_counts = defaultdict(int)

        # Collect candidates with scores
        candidates = []

        # Find hierarchically related documents
        for doc in self.data_loader.all_records:
            doc_id = self._get_doc_id(doc)

            if not doc_id or pool.contains(doc_id):
                continue

            doc_hierarchy_level = doc.get('kg_hierarchy_level')
            doc_reg_type = doc.get('regulation_type')
            doc_year = doc.get('year')

            if not doc_hierarchy_level:
                doc_hierarchy_level = self._infer_hierarchy_level(doc_reg_type)

            if not doc_hierarchy_level:
                continue

            # Calculate hierarchy distance
            level_diff = doc_hierarchy_level - seed_hierarchy_level

            # Check if within allowed distance
            if abs(level_diff) > max_distance:
                continue

            # Year proximity check (apply to both UP and DOWN)
            if seed_year and doc_year:
                try:
                    year_diff = abs(int(doc_year) - int(seed_year))
                    if year_diff > year_range:
                        continue
                except (ValueError, TypeError):
                    # Skip if year parsing fails
                    continue

            # Expand UP: Find parent regulations (lower level number = higher in hierarchy)
            if expand_up and level_diff < 0:
                # This is a parent regulation
                hierarchy_score = 0.6 + (0.1 * (max_distance - abs(level_diff)))
                candidates.append((doc, doc_hierarchy_level, hierarchy_score, 'up'))

            # Expand DOWN: Find implementing regulations (higher level number = lower in hierarchy)
            elif expand_down and level_diff > 0:
                # This is an implementing regulation
                hierarchy_score = 0.5 + (0.1 * (max_distance - abs(level_diff)))
                candidates.append((doc, doc_hierarchy_level, hierarchy_score, 'down'))

        # Sort candidates by score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Add candidates respecting per-level limits
        for doc, doc_level, hierarchy_score, direction in candidates:
            # Check per-level limit
            if level_counts[doc_level] >= max_docs_per_level:
                continue

            added = pool.add(
                doc,
                source='hierarchical_expansion',
                seed_id=self._get_doc_id(seed_doc),
                round_num=round_num,
                score=hierarchy_score
            )

            if added:
                added_count += 1
                level_counts[doc_level] += 1

        self.logger.debug(f"Hierarchical expansion: Added {added_count} hierarchically related docs "
                         f"(levels: {dict(level_counts)})")

        return added_count

    def _infer_hierarchy_level(self, reg_type: str) -> Optional[int]:
        """
        Infer hierarchy level from regulation type string

        Args:
            reg_type: Regulation type (UU, PP, PERPRES, etc.)

        Returns:
            Hierarchy level (1-6) or None
        """
        if not reg_type:
            return None

        reg_type_upper = reg_type.upper()

        # Map regulation types to hierarchy levels
        if 'UUD' in reg_type_upper or 'KONSTITUSI' in reg_type_upper:
            return 1
        elif 'UNDANG-UNDANG' in reg_type_upper or reg_type_upper == 'UU':
            return 2
        elif 'PERATURAN PEMERINTAH' in reg_type_upper or reg_type_upper == 'PP':
            return 3
        elif 'PERPRES' in reg_type_upper or 'PERATURAN PRESIDEN' in reg_type_upper:
            return 4
        elif 'PERMEN' in reg_type_upper or 'PERATURAN MENTERI' in reg_type_upper:
            return 5
        elif 'PERDA' in reg_type_upper or 'PERATURAN DAERAH' in reg_type_upper:
            return 6
        else:
            return None

    def _apply_smart_filtering(
        self,
        pool: ResearchPool,
        initial_documents: List[Dict[str, Any]]
    ) -> ResearchPool:
        """
        Apply smart filtering to reduce noise in expanded pool

        Strategy:
        1. Keep all initial retrieval documents (high quality)
        2. Filter expanded documents by semantic similarity to top-10 initial docs
        3. Apply diversity filtering to prevent over-representation
        4. Limit pool size to max_pool_size

        Args:
            pool: Research pool with all documents
            initial_documents: Original seed documents from initial retrieval

        Returns:
            Filtered research pool
        """
        if len(pool) <= self.filtering_config.get('max_pool_size', 500):
            self.logger.info("Pool size within limit, skipping smart filtering")
            return pool

        self.logger.info(f"Applying smart filtering to pool of {len(pool)} documents")

        # Safety: Track filtering start time
        filtering_start_time = time.time()
        max_filtering_time = 60  # 1 minute max for filtering

        semantic_threshold = self.filtering_config.get('semantic_threshold', 0.60)
        max_pool_size = self.filtering_config.get('max_pool_size', 500)
        diversity_weight = self.filtering_config.get('diversity_weight', 0.3)

        # Get top-10 initial documents (reference set for filtering)
        top_initial = sorted(
            initial_documents,
            key=lambda x: x.get('scores', {}).get('final', 0.0) if isinstance(x.get('scores'), dict) else x.get('score', 0.0),
            reverse=True
        )[:10]

        # Calculate average embedding of top initial docs
        top_embeddings = []
        for doc in top_initial:
            emb = doc.get('embedding')
            if emb is not None:
                if isinstance(emb, list):
                    emb = np.array(emb)
                top_embeddings.append(emb)

        if not top_embeddings:
            self.logger.warning("No embeddings found in top initial docs, skipping semantic filtering")
            return pool

        # Average embedding (centroid of top results)
        avg_embedding = np.mean(top_embeddings, axis=0)
        avg_norm = np.linalg.norm(avg_embedding)
        if avg_norm == 0:
            self.logger.warning("Zero average embedding, skipping semantic filtering")
            return pool
        avg_embedding_normalized = avg_embedding / avg_norm

        # Filter documents
        filtered_pool = ResearchPool(logger=self.logger)
        regulation_counts = defaultdict(int)

        # Always keep initial retrieval documents
        for doc_id, doc in pool.documents.items():
            prov = pool.provenance[doc_id]
            if prov['source'] == 'initial_retrieval':
                filtered_pool.add(
                    doc,
                    source=prov['source'],
                    seed_id=prov.get('seed_id'),
                    round_num=prov['round'],
                    score=prov.get('score')
                )
                # Track regulation diversity
                reg_key = f"{doc.get('regulation_type', 'UNK')}_{doc.get('regulation_number', 'UNK')}"
                regulation_counts[reg_key] += 1

        # Score and filter expanded documents
        expanded_docs_scored = []

        for doc_id, doc in pool.documents.items():
            prov = pool.provenance[doc_id]

            # Skip initial retrieval (already added)
            if prov['source'] == 'initial_retrieval':
                continue

            # Calculate semantic similarity to top initial docs
            doc_embedding = doc.get('embedding')
            if doc_embedding is None:
                # No embedding, use expansion score only
                similarity = 0.0
            else:
                if isinstance(doc_embedding, list):
                    doc_embedding = np.array(doc_embedding)

                doc_norm = np.linalg.norm(doc_embedding)
                if doc_norm == 0:
                    similarity = 0.0
                else:
                    doc_embedding_normalized = doc_embedding / doc_norm
                    similarity = float(np.dot(avg_embedding_normalized, doc_embedding_normalized))

            # Get expansion score (quality score from expansion strategy)
            expansion_score = prov.get('score', 0.0)
            if expansion_score is None:
                expansion_score = 0.0

            # Combined score: weighted average of similarity and expansion score
            combined_score = (1 - diversity_weight) * similarity + diversity_weight * expansion_score

            # Calculate diversity penalty
            reg_key = f"{doc.get('regulation_type', 'UNK')}_{doc.get('regulation_number', 'UNK')}"
            diversity_penalty = min(regulation_counts.get(reg_key, 0) * 0.05, 0.3)  # Max 30% penalty

            final_score = combined_score - diversity_penalty

            expanded_docs_scored.append((doc, prov, similarity, final_score, reg_key))

        # Sort by final score
        expanded_docs_scored.sort(key=lambda x: x[3], reverse=True)

        # Add top expanded docs up to max_pool_size
        added_expanded = 0
        max_expanded = max_pool_size - len(filtered_pool)

        for doc, prov, similarity, final_score, reg_key in expanded_docs_scored:
            # Safety check: Timeout during filtering
            if time.time() - filtering_start_time > max_filtering_time:
                self.logger.warning(f"Filtering timeout after {time.time() - filtering_start_time:.1f}s, stopping early")
                break

            # HARD LIMIT: Stop at max pool size
            if added_expanded >= max_expanded:
                break

            # Relaxed semantic threshold for certain expansion types
            # Temporal/hierarchical/topical get lower threshold (0.50) vs standard (0.60)
            effective_threshold = semantic_threshold
            if prov['source'] in ['temporal_expansion', 'hierarchical_expansion', 'topical_expansion']:
                effective_threshold = max(0.50, semantic_threshold - 0.10)

            # Check semantic threshold
            if similarity < effective_threshold:
                continue

            # Add to filtered pool
            filtered_pool.add(
                doc,
                source=prov['source'],
                seed_id=prov.get('seed_id'),
                round_num=prov['round'],
                score=final_score  # Use filtered score
            )

            regulation_counts[reg_key] += 1
            added_expanded += 1

        self.logger.info(f"Smart filtering: Reduced pool from {len(pool)} to {len(filtered_pool)} docs "
                        f"({len(filtered_pool) - len([p for p in pool.provenance.values() if p['source'] == 'initial_retrieval'])} expanded kept)")

        # Cleanup temporary objects to free CPU memory
        del top_embeddings, avg_embedding, avg_embedding_normalized
        del expanded_docs_scored, regulation_counts

        # Aggressive cleanup after intensive similarity calculations
        aggressive_cleanup("smart filtering complete")

        return filtered_pool

    def _topical_expansion(
        self,
        seed_doc: Dict[str, Any],
        pool: ResearchPool,
        round_num: int
    ) -> int:
        """
        Strategy 8: Topical Expansion (Phase 7) - Legal Domain/Topic

        Find regulations on the same legal topic/domain:
        - Tax law (pajak)
        - Labor law (ketenagakerjaan)
        - Criminal law (pidana)
        - Civil law (perdata)
        - Administrative law (administrasi)
        etc.

        Uses kg_primary_domain field to find topic clusters

        Args:
            seed_doc: Document to expand from
            pool: Research pool to add results
            round_num: Current expansion round

        Returns:
            Number of documents added
        """
        added_count = 0

        # Extract legal domain
        seed_domain = seed_doc.get('kg_primary_domain')
        seed_domain_confidence = seed_doc.get('kg_domain_confidence', 0.0)

        if not seed_domain:
            self.logger.debug(f"Seed doc {self._get_doc_id(seed_doc)} missing domain data, skipping")
            return 0

        domain_threshold = self.topical_config.get('domain_threshold', 0.7)
        max_docs_per_topic = self.topical_config.get('max_docs_per_topic', 20)

        # Skip if seed domain confidence is low
        if seed_domain_confidence < domain_threshold:
            self.logger.debug(f"Seed domain confidence too low ({seed_domain_confidence:.2f} < {domain_threshold})")
            return 0

        self.logger.debug(f"Topical expansion for domain '{seed_domain}' (confidence={seed_domain_confidence:.2f})")

        # Find documents in same legal domain
        topical_docs = []

        for doc in self.data_loader.all_records:
            doc_id = self._get_doc_id(doc)

            if not doc_id or pool.contains(doc_id):
                continue

            doc_domain = doc.get('kg_primary_domain')
            doc_domain_confidence = doc.get('kg_domain_confidence', 0.0)

            # Match domain
            if doc_domain == seed_domain and doc_domain_confidence >= domain_threshold:
                # Score based on domain confidence
                topical_score = 0.3 + (0.3 * doc_domain_confidence)

                topical_docs.append((doc, topical_score))

        # Sort by score descending
        topical_docs.sort(key=lambda x: x[1], reverse=True)

        # Add topical documents to pool (limit to max_docs_per_topic)
        for doc, score in topical_docs[:max_docs_per_topic]:
            added = pool.add(
                doc,
                source='topical_expansion',
                seed_id=self._get_doc_id(seed_doc),
                round_num=round_num,
                score=score
            )

            if added:
                added_count += 1

        self.logger.debug(f"Topical expansion: Added {added_count} docs from domain '{seed_domain}'")

        # Cleanup temporary list
        del topical_docs

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
