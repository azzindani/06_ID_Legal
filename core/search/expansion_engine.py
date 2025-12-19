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
from .research_pool import ResearchPool
import numpy as np


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

        self.logger.info(f"ExpansionEngine initialized (enabled={self.enabled}, max_rounds={self.max_rounds}, "
                        f"metadata={self.metadata_config.get('enabled')}, "
                        f"kg={self.kg_config.get('enabled')}, "
                        f"citation={self.citation_config.get('enabled')}, "
                        f"semantic={self.semantic_config.get('enabled')}, "
                        f"hybrid={self.hybrid_config.get('enabled')})")

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

        # Extract regulation identifiers
        regulation_type = seed_doc.get('regulation_type')
        regulation_number = seed_doc.get('regulation_number')
        regulation_year = seed_doc.get('year')

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
