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

        self.logger.info(f"ExpansionEngine initialized (enabled={self.enabled}, max_rounds={self.max_rounds}, "
                        f"metadata={self.metadata_config.get('enabled')}, "
                        f"kg={self.kg_config.get('enabled')}, "
                        f"citation={self.citation_config.get('enabled')})")

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
            self.logger.debug(f"Seed doc {seed_doc.get('global_id')} has no KG data, skipping")
            return 0

        self.logger.debug(f"KG expansion for {len(kg_entities)} entities, {len(kg_citations)} citations")

        max_entity_docs = self.kg_config.get('max_entity_docs', 20)

        # 1. Entity co-occurrence expansion
        for entity in kg_entities[:5]:  # Limit to top 5 entities
            entity_lower = entity.lower() if isinstance(entity, str) else str(entity).lower()
            found_count = 0

            for doc in self.data_loader.all_records:
                if pool.contains(doc.get('global_id')):
                    continue

                # Check if doc mentions this entity
                doc_entities = doc.get('kg_entities', [])
                doc_entities_lower = [e.lower() if isinstance(e, str) else str(e).lower() for e in doc_entities]

                if entity_lower in doc_entities_lower:
                    added = pool.add(
                        doc,
                        source='kg_expansion',
                        seed_id=seed_doc.get('global_id'),
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
                    if pool.contains(doc.get('global_id')):
                        continue

                    # Match citation to document
                    if self._citation_matches_doc(citation_parts, doc):
                        added = pool.add(
                            doc,
                            source='kg_expansion',
                            seed_id=seed_doc.get('global_id'),
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
            self.logger.debug(f"Seed doc {seed_doc.get('global_id')} has no citations, skipping")
            return 0

        max_hops = self.citation_config.get('max_hops', 2)
        bidirectional = self.citation_config.get('bidirectional', True)

        self.logger.debug(f"Citation expansion for {len(kg_citations)} citations (max_hops={max_hops})")

        # Multi-hop traversal using BFS
        visited = set()
        queue = [(seed_doc, 0)]  # (doc, hop_level)

        while queue:
            current_doc, hop = queue.pop(0)
            doc_id = current_doc.get('global_id')

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
                    if pool.contains(doc.get('global_id')):
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
                if pool.contains(doc.get('global_id')):
                    continue

                doc_citations = doc.get('kg_citations', [])

                for citation in doc_citations:
                    # Check if citation matches seed document
                    if self._citation_text_matches(citation, seed_reg_type, seed_reg_number, seed_year):
                        added = pool.add(
                            doc,
                            source='citation_expansion',
                            seed_id=seed_doc.get('global_id'),
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
