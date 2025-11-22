"""
Knowledge Graph Core - Entity Extraction and Scoring

Provides entity extraction and KG-based scoring for legal documents.
"""

from typing import Dict, Any, List, Optional, Set
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger

logger = get_logger(__name__)


class KnowledgeGraphCore:
    """
    Core knowledge graph functionality

    Handles entity extraction, relationship identification,
    and graph-based document scoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Entity patterns for Indonesian legal documents
        self.patterns = {
            'regulation': r'(?:UU|PP|Perpres|Perda|Permen|Peraturan Pemerintah|Undang-Undang)\s*(?:No\.?\s*)?\d+\s*(?:Tahun\s*)?\d{4}',
            'article': r'Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?(?:\s+huruf\s+[a-z])?',
            'institution': r'(?:Menteri|Presiden|DPR|Pemerintah|Pengadilan|Gubernur|Bupati|Walikota)\s+\w+',
            'legal_term': r'(?:sanksi|pidana|denda|hukuman|pelanggaran|ketentuan|larangan|kewajiban|hak)',
            'legal_action': r'(?:mengatur|melarang|mewajibkan|memberikan|menetapkan|mencabut)',
        }

        # Entity weights for scoring
        self.entity_weights = {
            'regulation': 1.5,
            'article': 1.2,
            'institution': 1.0,
            'legal_term': 0.8,
            'legal_action': 0.7,
        }

        # Regulation type mappings for confidence scoring
        self.regulation_type_mappings = {
            'UU': 'Undang-Undang',
            'PP': 'Peraturan Pemerintah',
            'Perpres': 'Peraturan Presiden',
            'Permen': 'Peraturan Menteri',
            'Perda': 'Peraturan Daerah',
            'Keppres': 'Keputusan Presiden',
            'Kepmen': 'Keputusan Menteri',
        }

        # Advanced KG weights from original notebook
        self.kg_weights = {
            'direct_match': 1.0,
            'one_hop': 0.8,
            'two_hop': 0.6,
            'concept_cluster': 0.7,
            'hierarchy_boost': 0.5,
            'temporal_relevance': 0.4,
            'cross_reference': 0.6,
            'domain_match': 0.5,
            'legal_action_match': 0.7,
            'sanction_relevance': 0.8,
            'citation_impact': 0.4,
            'connectivity_boost': 0.3
        }

        # Domain mappings for legal areas
        self.domain_mappings = {
            'ketenagakerjaan': ['pekerja', 'buruh', 'upah', 'phk', 'kontrak kerja', 'cuti'],
            'perdata': ['perjanjian', 'kontrak', 'ganti rugi', 'wanprestasi'],
            'pidana': ['sanksi', 'pidana', 'denda', 'penjara', 'pelanggaran'],
            'administrasi': ['perizinan', 'izin', 'pendaftaran', 'prosedur'],
            'keuangan': ['pajak', 'cukai', 'bea', 'anggaran', 'kas negara'],
            'pendidikan': ['guru', 'dosen', 'sekolah', 'universitas', 'tunjangan profesi'],
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities from text

        Args:
            text: Input text

        Returns:
            Dictionary of entity types to entity lists
        """
        entities = {}

        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))

        return entities

    def extract_regulation_references_with_confidence(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract regulation references from text with confidence scores.

        Confidence levels:
        - 1.0: Complete reference (type + number + year) e.g., "UU No. 13 Tahun 2003"
        - 0.7: Partial reference (type + number OR type + year) e.g., "UU No. 13"
        - 0.3: Vague reference (type only or contextual) e.g., "Undang-Undang Ketenagakerjaan"

        Args:
            text: Input text to analyze

        Returns:
            List of dicts with 'reference', 'type', 'number', 'year', 'confidence'
        """
        references = []
        text_upper = text.upper()

        # Pattern 1: Complete reference (type + number + year) - confidence 1.0
        complete_pattern = r'(?:UU|PP|Perpres|Permen|Perda|Keppres|Kepmen|Undang-Undang|Peraturan\s+Pemerintah|Peraturan\s+Presiden|Peraturan\s+Menteri|Peraturan\s+Daerah)\s*(?:No\.?\s*)?(\d+)\s*(?:Tahun\s*)(\d{4})'

        for match in re.finditer(complete_pattern, text, re.IGNORECASE):
            full_match = match.group(0)
            number = match.group(1)
            year = match.group(2)

            # Determine regulation type
            reg_type = self._extract_regulation_type(full_match)

            references.append({
                'reference': full_match.strip(),
                'type': reg_type,
                'number': number,
                'year': year,
                'confidence': 1.0
            })

        # Pattern 2: Partial reference (type + number without year) - confidence 0.7
        partial_pattern = r'(?:UU|PP|Perpres|Permen|Perda|Keppres|Kepmen|Undang-Undang|Peraturan\s+Pemerintah)\s*(?:No\.?\s*)?(\d+)(?!\s*(?:Tahun\s*)\d{4})'

        for match in re.finditer(partial_pattern, text, re.IGNORECASE):
            full_match = match.group(0)
            number = match.group(1)

            # Skip if already captured as complete reference
            if any(number in ref['reference'] for ref in references if ref['confidence'] == 1.0):
                continue

            reg_type = self._extract_regulation_type(full_match)

            references.append({
                'reference': full_match.strip(),
                'type': reg_type,
                'number': number,
                'year': None,
                'confidence': 0.7
            })

        # Pattern 3: Vague reference (type + topic keyword) - confidence 0.3
        vague_patterns = [
            (r'(?:UU|Undang-Undang)\s+(?:tentang\s+)?(?:Ketenagakerjaan|Perseroan|Perkawinan|Perlindungan\s+Konsumen|Cipta\s+Kerja)', 'UU'),
            (r'(?:PP|Peraturan\s+Pemerintah)\s+(?:tentang\s+)?(?:Pengupahan|Pelaksanaan|Pendirian)', 'PP'),
            (r'(?:Perpres|Peraturan\s+Presiden)\s+(?:tentang\s+)?(?:Pengadaan|Investasi)', 'Perpres'),
        ]

        for pattern, default_type in vague_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                full_match = match.group(0)

                # Skip if already captured
                if any(full_match.lower() in ref['reference'].lower() for ref in references):
                    continue

                references.append({
                    'reference': full_match.strip(),
                    'type': default_type,
                    'number': None,
                    'year': None,
                    'confidence': 0.3
                })

        # Sort by confidence (highest first)
        references.sort(key=lambda x: x['confidence'], reverse=True)

        logger.debug(f"Extracted {len(references)} regulation references from query")
        return references

    def _extract_regulation_type(self, text: str) -> str:
        """Extract regulation type from text"""
        text_upper = text.upper()

        if 'UNDANG-UNDANG' in text_upper or text_upper.startswith('UU'):
            return 'UU'
        elif 'PERATURAN PEMERINTAH' in text_upper or text_upper.startswith('PP'):
            return 'PP'
        elif 'PERATURAN PRESIDEN' in text_upper or text_upper.startswith('PERPRES'):
            return 'Perpres'
        elif 'PERATURAN MENTERI' in text_upper or text_upper.startswith('PERMEN'):
            return 'Permen'
        elif 'PERATURAN DAERAH' in text_upper or text_upper.startswith('PERDA'):
            return 'Perda'
        elif 'KEPUTUSAN PRESIDEN' in text_upper or text_upper.startswith('KEPPRES'):
            return 'Keppres'
        elif 'KEPUTUSAN MENTERI' in text_upper or text_upper.startswith('KEPMEN'):
            return 'Kepmen'

        return 'Unknown'

    def calculate_entity_score(
        self,
        doc_entities: Dict[str, List[str]],
        query_entities: Dict[str, List[str]]
    ) -> float:
        """
        Calculate entity overlap score between document and query

        Args:
            doc_entities: Entities from document
            query_entities: Entities from query

        Returns:
            Entity overlap score
        """
        if not query_entities:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for entity_type, query_ents in query_entities.items():
            if not query_ents:
                continue

            doc_ents = doc_entities.get(entity_type, [])
            weight = self.entity_weights.get(entity_type, 1.0)

            # Calculate overlap
            query_set = set(e.lower() for e in query_ents)
            doc_set = set(e.lower() for e in doc_ents)

            if query_set:
                overlap = len(query_set & doc_set) / len(query_set)
                total_score += overlap * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def enhance_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        kg_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Enhance search results with KG-based scoring

        Args:
            results: Search results
            query: Original query
            kg_weight: Weight for KG score

        Returns:
            Enhanced results with KG scores
        """
        query_entities = self.extract_entities(query)

        enhanced = []
        for result in results:
            doc_text = result.get('content', '') or result.get('document', {}).get('content', '')
            doc_entities = self.extract_entities(doc_text)

            kg_score = self.calculate_entity_score(doc_entities, query_entities)

            # Combine scores
            original_score = result.get('score', 0.0)
            combined_score = (1 - kg_weight) * original_score + kg_weight * kg_score

            enhanced_result = result.copy()
            enhanced_result['kg_score'] = kg_score
            enhanced_result['original_score'] = original_score
            enhanced_result['score'] = combined_score
            enhanced_result['entities'] = doc_entities

            enhanced.append(enhanced_result)

        # Re-sort by combined score
        enhanced.sort(key=lambda x: x['score'], reverse=True)

        return enhanced

    def get_related_regulations(
        self,
        regulation: str,
        all_documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Find regulations related to a given regulation

        Args:
            regulation: Source regulation
            all_documents: All available documents

        Returns:
            List of related regulation identifiers
        """
        related = set()

        for doc in all_documents:
            content = doc.get('content', '')
            if regulation.lower() in content.lower():
                # Extract other regulations mentioned
                entities = self.extract_entities(content)
                for reg in entities.get('regulation', []):
                    if reg.lower() != regulation.lower():
                        related.add(reg)

        return list(related)

    def detect_domain(self, text: str) -> List[str]:
        """
        Detect legal domains present in text

        Args:
            text: Input text

        Returns:
            List of detected domains
        """
        text_lower = text.lower()
        detected = []

        for domain, keywords in self.domain_mappings.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(domain)

        return detected

    def calculate_advanced_score(
        self,
        doc_entities: Dict[str, List[str]],
        query_entities: Dict[str, List[str]],
        doc_text: str,
        query_text: str,
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate advanced KG score with multiple factors

        Args:
            doc_entities: Entities from document
            query_entities: Entities from query
            doc_text: Full document text
            query_text: Query text
            doc_metadata: Optional document metadata (kg_primary_domain, etc.)

        Returns:
            Dictionary of score components and total
        """
        scores = {}

        # 1. Direct entity match
        scores['direct_match'] = self.calculate_entity_score(doc_entities, query_entities)

        # 2. Domain match
        query_domains = self.detect_domain(query_text)
        doc_domains = self.detect_domain(doc_text)

        if doc_metadata and doc_metadata.get('kg_primary_domain'):
            doc_domains.append(doc_metadata['kg_primary_domain'])

        domain_overlap = len(set(query_domains) & set(doc_domains))
        scores['domain_match'] = min(1.0, domain_overlap * 0.5) if query_domains else 0.0

        # 3. Sanction relevance (if query mentions sanctions)
        sanction_keywords = ['sanksi', 'pidana', 'denda', 'hukuman']
        if any(kw in query_text.lower() for kw in sanction_keywords):
            if any(kw in doc_text.lower() for kw in sanction_keywords):
                scores['sanction_relevance'] = 0.8
            else:
                scores['sanction_relevance'] = 0.0
        else:
            scores['sanction_relevance'] = 0.0

        # 4. Legal action match
        query_actions = self.extract_entities(query_text).get('legal_action', [])
        doc_actions = doc_entities.get('legal_action', [])

        if query_actions and doc_actions:
            action_overlap = len(set(a.lower() for a in query_actions) & set(a.lower() for a in doc_actions))
            scores['legal_action_match'] = min(1.0, action_overlap * 0.3)
        else:
            scores['legal_action_match'] = 0.0

        # 5. Hierarchy boost from metadata
        if doc_metadata:
            hierarchy_level = doc_metadata.get('kg_hierarchy_level', 0)
            authority_score = doc_metadata.get('kg_authority_score', 0)
            scores['hierarchy_boost'] = min(1.0, (hierarchy_level * 0.1 + authority_score * 0.5))
        else:
            scores['hierarchy_boost'] = 0.0

        # 6. Cross-reference score
        cross_refs = doc_metadata.get('kg_cross_references', []) if doc_metadata else []
        scores['cross_reference'] = min(1.0, len(cross_refs) * 0.1)

        # Calculate weighted total
        total = 0.0
        for component, score in scores.items():
            weight = self.kg_weights.get(component, 0.5)
            total += score * weight

        # Normalize
        max_possible = sum(self.kg_weights.get(k, 0.5) for k in scores.keys())
        scores['total'] = total / max_possible if max_possible > 0 else 0.0

        return scores

    def enhance_results_advanced(
        self,
        results: List[Dict[str, Any]],
        query: str,
        kg_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Enhanced version using advanced KG scoring

        Args:
            results: Search results
            query: Original query
            kg_weight: Weight for KG score

        Returns:
            Enhanced results with detailed KG scores
        """
        query_entities = self.extract_entities(query)

        enhanced = []
        for result in results:
            record = result.get('record', {})
            doc_text = record.get('content', '') or result.get('content', '')
            doc_entities = self.extract_entities(doc_text)

            # Get metadata from record
            doc_metadata = {
                'kg_primary_domain': record.get('kg_primary_domain', ''),
                'kg_hierarchy_level': record.get('kg_hierarchy_level', 0),
                'kg_authority_score': record.get('kg_authority_score', 0),
                'kg_cross_references': record.get('kg_cross_references', []),
            }

            # Calculate advanced scores
            kg_scores = self.calculate_advanced_score(
                doc_entities=doc_entities,
                query_entities=query_entities,
                doc_text=doc_text,
                query_text=query,
                doc_metadata=doc_metadata
            )

            # Combine scores
            original_score = result.get('score', result.get('final_score', 0.0))
            combined_score = (1 - kg_weight) * original_score + kg_weight * kg_scores['total']

            enhanced_result = result.copy()
            enhanced_result['kg_score'] = kg_scores['total']
            enhanced_result['kg_score_breakdown'] = kg_scores
            enhanced_result['original_score'] = original_score
            enhanced_result['final_score'] = combined_score
            enhanced_result['entities'] = doc_entities
            enhanced_result['domains'] = self.detect_domain(doc_text)

            enhanced.append(enhanced_result)

        # Re-sort by combined score
        enhanced.sort(key=lambda x: x['final_score'], reverse=True)

        return enhanced

    def follow_citation_chain(
        self,
        start_doc_id: str,
        all_documents: List[Dict[str, Any]],
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Traverse citation network to find related documents.

        Follows cross-references up to max_depth hops to find
        documents in the citation chain.

        Args:
            start_doc_id: Starting document ID
            all_documents: All available documents
            max_depth: Maximum traversal depth (default 2)

        Returns:
            List of related documents with depth information
        """
        # Build document lookup
        doc_lookup = {}
        for doc in all_documents:
            doc_id = doc.get('global_id', doc.get('id'))
            if doc_id:
                doc_lookup[str(doc_id)] = doc

        visited = set()
        results = []

        def traverse(doc_id: str, depth: int):
            if depth > max_depth or doc_id in visited:
                return

            visited.add(doc_id)
            doc = doc_lookup.get(str(doc_id))

            if not doc:
                return

            if depth > 0:  # Don't include start document
                results.append({
                    'document': doc,
                    'depth': depth,
                    'global_id': doc_id
                })

            # Get cross-references
            cross_refs = doc.get('kg_cross_references', [])

            for ref in cross_refs:
                ref_id = ref if isinstance(ref, (int, str)) else ref.get('id', ref.get('global_id'))
                if ref_id:
                    traverse(str(ref_id), depth + 1)

        # Start traversal
        traverse(str(start_doc_id), 0)

        logger.debug(f"Citation chain from {start_doc_id}: found {len(results)} related docs")
        return results

    def boost_cited_documents(
        self,
        results: List[Dict[str, Any]],
        all_documents: List[Dict[str, Any]],
        boost_factor: float = 0.1,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Boost scores of documents that appear in citation chains.

        Documents that are cited by high-scoring results get a score boost.

        Args:
            results: Search results to process
            all_documents: All available documents for chain lookup
            boost_factor: Score boost per citation (default 0.1)
            max_depth: Max citation chain depth (default 2)

        Returns:
            Results with boosted scores for cited documents
        """
        if not results:
            return results

        # Track citation counts for each document
        citation_counts: Dict[str, int] = {}
        citation_depths: Dict[str, int] = {}

        # Follow citation chains from top results
        top_n = min(10, len(results))  # Process top 10 results

        for result in results[:top_n]:
            doc_id = result.get('metadata', {}).get('global_id')
            if not doc_id:
                record = result.get('record', {})
                doc_id = record.get('global_id', record.get('id'))

            if doc_id:
                chain = self.follow_citation_chain(
                    start_doc_id=str(doc_id),
                    all_documents=all_documents,
                    max_depth=max_depth
                )

                for cited in chain:
                    cited_id = str(cited['global_id'])
                    citation_counts[cited_id] = citation_counts.get(cited_id, 0) + 1

                    # Track minimum depth (closer = better)
                    if cited_id not in citation_depths:
                        citation_depths[cited_id] = cited['depth']
                    else:
                        citation_depths[cited_id] = min(citation_depths[cited_id], cited['depth'])

        # Apply boosts to results
        boosted_results = []

        for result in results:
            doc_id = result.get('metadata', {}).get('global_id')
            if not doc_id:
                record = result.get('record', {})
                doc_id = record.get('global_id', record.get('id'))

            boosted_result = result.copy()
            original_score = result.get('scores', {}).get('final', result.get('final_score', 0.0))

            if doc_id and str(doc_id) in citation_counts:
                count = citation_counts[str(doc_id)]
                depth = citation_depths[str(doc_id)]

                # Boost formula: more citations and closer = higher boost
                # Depth 1 = full boost, depth 2 = half boost
                depth_multiplier = 1.0 / depth
                boost = boost_factor * count * depth_multiplier

                new_score = min(1.0, original_score + boost)

                if 'scores' in boosted_result:
                    boosted_result['scores'] = boosted_result['scores'].copy()
                    boosted_result['scores']['final'] = new_score
                    boosted_result['scores']['citation_boost'] = boost
                else:
                    boosted_result['final_score'] = new_score
                    boosted_result['citation_boost'] = boost

                boosted_result['citation_count'] = count
                boosted_result['citation_depth'] = depth

            boosted_results.append(boosted_result)

        # Re-sort by score
        boosted_results.sort(
            key=lambda x: x.get('scores', {}).get('final', x.get('final_score', 0)),
            reverse=True
        )

        boosted_count = sum(1 for r in boosted_results if r.get('citation_count', 0) > 0)
        logger.debug(f"Citation boosting: {boosted_count} documents boosted")

        return boosted_results
