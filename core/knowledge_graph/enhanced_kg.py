"""
Enhanced Knowledge Graph - Entity Extraction, Scoring, and Citation Chain Analysis

Provides the EnhancedKnowledgeGraph class with caching support for Indonesian legal documents.
"""

import json
import re
from typing import Dict, List, Any, Optional
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import KG_WEIGHTS, REGULATION_TYPE_PATTERNS, YEAR_SEPARATORS
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class EnhancedKnowledgeGraph:
    """Enhanced KG class with caching"""

    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader
        self.entities_lookup = dataset_loader.kg_entities_lookup
        self.cross_refs_lookup = dataset_loader.kg_cross_references_lookup
        self.domains_lookup = dataset_loader.kg_domains_lookup
        self.clusters_lookup = dataset_loader.kg_concept_clusters_lookup
        self.legal_actions_lookup = dataset_loader.kg_legal_actions_lookup
        self.sanctions_lookup = dataset_loader.kg_sanctions_lookup
        self.concept_vectors_lookup = dataset_loader.kg_concept_vectors_lookup

        # Caching
        self._parse_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def extract_entities_from_text(self, text):
        """Enhanced entity extraction for ANY Indonesian regulation"""
        if not text or pd.isna(text):
            return []

        try:
            text_lower = str(text).lower()
            entities = []

            # Pattern 1: Standard format - "Type No. X Tahun YYYY"
            for reg_type, patterns in REGULATION_TYPE_PATTERNS.items():
                for pattern in patterns:
                    # Build flexible regex pattern
                    # Matches: "PP No. 41 Tahun 2009", "pp no 41 tahun 2009", "PP 41/2009", etc.
                    regex_pattern = (
                        rf'{re.escape(pattern)}\s*'  # Regulation type
                        r'(?:nomor|no\.?|num\.?)?\s*'  # Optional "nomor"/"no"
                        r'(\d+)\s*'  # Number
                        r'(?:' + '|'.join([re.escape(sep) for sep in YEAR_SEPARATORS]) + r')?\s*'  # Separator
                        r'(\d{4})?'  # Optional year
                    )

                    matches = re.finditer(regex_pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        number = match.group(1)
                        year = match.group(2) if match.group(2) else ''

                        # Create normalized entity
                        entity_text = f"{pattern} {number}"
                        if year:
                            entity_text += f" tahun {year}"

                        entities.append((entity_text, 'regulation_reference'))

            # Pattern 2: Pasal (Article) references
            pasal_pattern = r'pasal\s*(\d+)(?:\s*ayat\s*\((\d+)\))?(?:\s*huruf\s*([a-z]))?'
            matches = re.finditer(pasal_pattern, text_lower)
            for match in matches:
                entities.append((match.group(0), 'article_reference'))

            # Pattern 3: Bab (Chapter) references
            bab_pattern = r'bab\s+([IVX]+|\d+)'
            matches = re.finditer(bab_pattern, text_lower)
            for match in matches:
                entities.append((match.group(0), 'chapter_reference'))

            return entities
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return []

    def get_parsed_kg_data(self, doc_id, data_type='entities'):
        """Parse KG JSON data on demand WITH CACHING"""
        # Create cache key
        cache_key = f"{doc_id}_{data_type}"

        # Check cache first
        if cache_key in self._parse_cache:
            self._cache_hits += 1
            return self._parse_cache[cache_key]

        self._cache_misses += 1

        try:
            result = None

            if data_type == 'entities' and doc_id in self.entities_lookup:
                result = json.loads(self.entities_lookup[doc_id])
            elif data_type == 'cross_refs' and doc_id in self.cross_refs_lookup:
                result = json.loads(self.cross_refs_lookup[doc_id])
            elif data_type == 'domains' and doc_id in self.domains_lookup:
                result = json.loads(self.domains_lookup[doc_id])
            elif data_type == 'clusters' and doc_id in self.clusters_lookup:
                result = json.loads(self.clusters_lookup[doc_id])
            elif data_type == 'legal_actions' and doc_id in self.legal_actions_lookup:
                result = json.loads(self.legal_actions_lookup[doc_id])
            elif data_type == 'sanctions' and doc_id in self.sanctions_lookup:
                result = json.loads(self.sanctions_lookup[doc_id])
            elif data_type == 'concept_vector' and doc_id in self.concept_vectors_lookup:
                result = json.loads(self.concept_vectors_lookup[doc_id])

            # Store in cache (limit cache size)
            if result is not None:
                if len(self._parse_cache) > 1000:  # Max 1000 cached entries
                    # Remove oldest 100 entries
                    keys_to_remove = list(self._parse_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self._parse_cache[key]

                self._parse_cache[cache_key] = result

            return result

        except Exception as e:
            print(f"Error parsing KG data for {doc_id}/{data_type}: {e}")
            return None

    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_size': len(self._parse_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def clear_cache(self):
        """Clear the parse cache"""
        self._parse_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def calculate_enhanced_kg_score(self, query_entities, record, query_type='general'):
        """Enhanced KG scoring with new dataset features"""
        try:
            total_score = 0.0
            doc_id = record['global_id']

            # 1. Entity matching (enhanced)
            doc_entities = self.get_parsed_kg_data(doc_id, 'entities')
            if doc_entities and query_entities:
                entity_score = self._calculate_entity_match(query_entities, doc_entities)
                total_score += entity_score * KG_WEIGHTS['direct_match']

            # 2. Cross-reference boost
            cross_refs = self.get_parsed_kg_data(doc_id, 'cross_refs')
            if cross_refs:
                cross_ref_score = self._calculate_cross_ref_relevance(query_entities, cross_refs)
                total_score += cross_ref_score * KG_WEIGHTS['cross_reference']

            # 3. Domain matching
            domains = self.get_parsed_kg_data(doc_id, 'domains')
            if domains:
                domain_score = self._calculate_domain_relevance(query_type, domains, record)
                total_score += domain_score * KG_WEIGHTS['domain_match']

            # 4. Legal actions matching (for procedural/sanctions queries)
            if query_type in ['procedural', 'sanctions']:
                legal_actions = self.get_parsed_kg_data(doc_id, 'legal_actions')
                if legal_actions:
                    action_score = self._calculate_legal_action_relevance(query_type, legal_actions, record)
                    total_score += action_score * KG_WEIGHTS['legal_action_match']

            # 5. Sanctions matching (for sanctions queries)
            if query_type == 'sanctions':
                sanctions = self.get_parsed_kg_data(doc_id, 'sanctions')
                if sanctions:
                    sanction_score = self._calculate_sanction_relevance(sanctions)
                    total_score += sanction_score * KG_WEIGHTS['sanction_relevance']

            # 6. Concept clusters matching
            clusters = self.get_parsed_kg_data(doc_id, 'clusters')
            if clusters and query_entities:
                cluster_score = self._calculate_cluster_relevance(query_entities, clusters)
                total_score += cluster_score * KG_WEIGHTS['concept_cluster']

            # 7. Hierarchy boost
            hierarchy_score = self._calculate_hierarchy_boost(record)
            total_score += hierarchy_score * KG_WEIGHTS['hierarchy_boost']

            # 8. Connectivity boost
            connectivity_score = record.get('kg_connectivity_score', 0.0)
            total_score += connectivity_score * KG_WEIGHTS['connectivity_boost']

            # 9. Citation impact (if available)
            if record.get('kg_pagerank', 0.0) > 0:
                citation_score = record['kg_pagerank']
                total_score += citation_score * KG_WEIGHTS['citation_impact']

            return min(1.0, total_score)
        except Exception:
            return 0.0

    def _calculate_entity_match(self, query_entities, doc_entities):
        """Calculate entity matching score"""
        try:
            if not query_entities or not doc_entities:
                return 0.0

            query_entity_set = {str(entity).lower() for entity in query_entities}

            # Handle both list of strings and list of dicts
            doc_entity_set = set()
            for entity in doc_entities:
                if isinstance(entity, dict):
                    doc_entity_set.add(str(entity.get('text', '')).lower())
                else:
                    doc_entity_set.add(str(entity).lower())

            overlap = query_entity_set & doc_entity_set
            if overlap:
                return min(1.0, len(overlap) / len(query_entity_set))

            # Partial matching
            partial_score = 0.0
            for q_entity in query_entities:
                for d_entity in doc_entity_set:
                    if str(q_entity).lower() in str(d_entity).lower():
                        partial_score += 0.5
                        break

            return min(1.0, partial_score / len(query_entities)) if query_entities else 0.0
        except Exception:
            return 0.0

    def _calculate_cross_ref_relevance(self, query_entities, cross_refs):
        """Calculate cross-reference relevance"""
        try:
            if not cross_refs:
                return 0.0

            # More cross-references = better connected document
            ref_count = len(cross_refs) if isinstance(cross_refs, list) else 0
            return min(1.0, ref_count / 10.0)  # Normalize to 0-1
        except Exception:
            return 0.0

    def _calculate_domain_relevance(self, query_type, domains, record):
        """Calculate domain relevance"""
        try:
            # Map query types to relevant domains
            query_domain_map = {
                'procedural': ['administrative', 'procedural', 'governance'],
                'sanctions': ['criminal', 'administrative', 'sanctions'],
                'definitional': ['general', 'definitions', 'terminology'],
                'specific_article': ['all'],
                'general': ['all']
            }

            relevant_domains = query_domain_map.get(query_type, ['all'])

            if 'all' in relevant_domains:
                return record.get('kg_domain_confidence', 0.5)

            # Check if document domains match relevant domains
            if isinstance(domains, list):
                for domain_info in domains:
                    if isinstance(domain_info, dict):
                        domain_name = domain_info.get('domain', '').lower()
                        if any(rel_dom in domain_name for rel_dom in relevant_domains):
                            return domain_info.get('confidence', 0.5)

            return 0.0
        except Exception:
            return 0.0

    def _calculate_legal_action_relevance(self, query_type, legal_actions, record):
        """Calculate legal action relevance"""
        try:
            if query_type == 'procedural':
                # Check for procedural actions
                if record.get('kg_has_obligations', False):
                    return 0.8
                if record.get('kg_has_permissions', False):
                    return 0.6
            elif query_type == 'sanctions':
                # Check for prohibitions/sanctions
                if record.get('kg_has_prohibitions', False):
                    return 0.9

            return 0.0
        except Exception:
            return 0.0

    def _calculate_sanction_relevance(self, sanctions):
        """Calculate sanction relevance"""
        try:
            if not sanctions:
                return 0.0

            # Check if sanctions data is available
            if isinstance(sanctions, dict):
                if sanctions.get('has_sanctions', False):
                    return 0.9
                sanction_count = len(sanctions.get('sanctions', []))
                return min(1.0, sanction_count / 3.0)

            return 0.0
        except Exception:
            return 0.0

    def _calculate_cluster_relevance(self, query_entities, clusters):
        """Calculate concept cluster relevance"""
        try:
            if not clusters or not query_entities:
                return 0.0

            # Extract concepts from clusters
            cluster_concepts = set()
            if isinstance(clusters, dict):
                for cluster_name, concepts in clusters.items():
                    if isinstance(concepts, list):
                        cluster_concepts.update([str(c).lower() for c in concepts])

            # Check overlap with query entities
            query_set = {str(e).lower() for e in query_entities}
            overlap = query_set & cluster_concepts

            if overlap:
                return min(1.0, len(overlap) / len(query_set))

            return 0.0
        except Exception:
            return 0.0

    def _calculate_hierarchy_boost(self, record):
        """Calculate hierarchy-based boost"""
        try:
            # Lower hierarchy level = higher authority
            hierarchy_level = record.get('kg_hierarchy_level', 5)
            # Normalize: level 1 = 1.0, level 10 = 0.1
            return max(0.1, (11 - hierarchy_level) / 10.0)
        except Exception:
            return 0.5

    def follow_citation_chain(self, seed_document_ids, max_depth=2):
        """Follow citation chains from seed documents"""
        try:
            citation_network = {}
            visited = set()

            def traverse_citations(doc_id, depth):
                if depth > max_depth or doc_id in visited:
                    return []

                visited.add(doc_id)
                related_docs = []

                # Get cross-references
                cross_refs = self.get_parsed_kg_data(doc_id, 'cross_refs')

                if cross_refs:
                    for ref in cross_refs[:5]:  # Limit to top 5 per document
                        try:
                            if isinstance(ref, dict):
                                ref_id = ref.get('target_id')
                            else:
                                ref_id = str(ref)

                            if ref_id and ref_id not in visited:
                                related_docs.append({
                                    'doc_id': ref_id,
                                    'citation_depth': depth,
                                    'cited_by': doc_id
                                })

                                # Recursive traversal
                                if depth < max_depth:
                                    deeper_docs = traverse_citations(ref_id, depth + 1)
                                    related_docs.extend(deeper_docs)
                        except Exception:
                            continue

                return related_docs

            # Start traversal from each seed
            for seed_id in seed_document_ids:
                if isinstance(seed_id, dict):
                    seed_id = seed_id.get('global_id', seed_id)

                citation_network[seed_id] = traverse_citations(seed_id, 1)

            return citation_network

        except Exception as e:
            print(f"Error following citations: {e}")
            return {}

    def boost_cited_documents(self, candidates, citation_network):
        """Boost documents that appear in citation chains"""
        try:
            # Collect all documents in citation network
            cited_docs = set()
            citation_depths = {}

            for seed_id, citations in citation_network.items():
                for citation in citations:
                    doc_id = citation['doc_id']
                    depth = citation['citation_depth']
                    cited_docs.add(doc_id)

                    # Track minimum depth (closer = better)
                    if doc_id not in citation_depths or depth < citation_depths[doc_id]:
                        citation_depths[doc_id] = depth

            # Apply citation bonuses
            for candidate in candidates:
                doc_id = candidate['record'].get('global_id')

                if doc_id in cited_docs:
                    depth = citation_depths[doc_id]
                    # Closer citations get higher boost
                    citation_bonus = 0.15 / depth if depth > 0 else 0.15

                    # Update scores
                    if 'final_consensus_score' in candidate:
                        candidate['final_consensus_score'] = min(1.0,
                            candidate['final_consensus_score'] + citation_bonus)
                    if 'composite_score' in candidate:
                        candidate['composite_score'] = min(1.0,
                            candidate['composite_score'] + citation_bonus)

                    candidate['in_citation_chain'] = True
                    candidate['citation_depth'] = depth

            return candidates

        except Exception as e:
            print(f"Error boosting cited documents: {e}")
            return candidates

    def extract_regulation_references_with_confidence(self, text):
        """
        Extract regulation references with confidence scores.
        Returns: List of (regulation_dict, confidence_score)
        """
        if not text or pd.isna(text):
            return []

        try:
            text_lower = str(text).lower()
            references = []

            # Pattern 1: Complete reference with year (HIGHEST CONFIDENCE)
            # e.g., "UU No. 13 Tahun 2003", "PP 41/2009"
            for reg_type, patterns in REGULATION_TYPE_PATTERNS.items():
                for pattern in patterns:
                    # Complete format: Type + Number + Year
                    complete_pattern = (
                        rf'{re.escape(pattern)}\s*'
                        r'(?:nomor|no\.?|num\.?|number)?\s*'
                        r'(\d+)\s*'
                        r'(?:tahun|th\.?|\/)\s*'
                        r'(\d{4})'
                    )

                    matches = re.finditer(complete_pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        number = match.group(1)
                        year = match.group(2)

                        references.append({
                            'regulation': {
                                'type': pattern,
                                'number': number,
                                'year': year,
                                'full_text': match.group(0)
                            },
                            'confidence': 1.0,  # HIGHEST - has type, number, and year
                            'specificity': 'complete'
                        })

            # Pattern 2: Reference with number but no year (MEDIUM CONFIDENCE)
            # e.g., "UU No. 13", "PP 41"
            for reg_type, patterns in REGULATION_TYPE_PATTERNS.items():
                for pattern in patterns:
                    partial_pattern = (
                        rf'{re.escape(pattern)}\s*'
                        r'(?:nomor|no\.?|num\.?|number)?\s*'
                        r'(\d+)(?!\s*(?:tahun|th\.?|\/)\s*\d{4})'  # Negative lookahead for year
                    )

                    matches = re.finditer(partial_pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        number = match.group(1)

                        # Check if not already captured with year
                        already_exists = any(
                            ref['regulation']['type'] == pattern and
                            ref['regulation']['number'] == number and
                            ref['confidence'] == 1.0
                            for ref in references
                        )

                        if not already_exists:
                            references.append({
                                'regulation': {
                                    'type': pattern,
                                    'number': number,
                                    'year': '',
                                    'full_text': match.group(0)
                                },
                                'confidence': 0.7,  # MEDIUM - has type and number
                                'specificity': 'partial'
                            })

            # Pattern 3: Just regulation type mentioned (LOW CONFIDENCE)
            # e.g., "undang-undang tersebut", "peraturan pemerintah ini"
            if not references:  # Only if no specific references found
                for reg_type, patterns in REGULATION_TYPE_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in text_lower:
                            references.append({
                                'regulation': {
                                    'type': pattern,
                                    'number': '',
                                    'year': '',
                                    'full_text': pattern
                                },
                                'confidence': 0.3,  # LOW - only type mentioned
                                'specificity': 'vague'
                            })
                            break  # Only add once per type

            # Sort by confidence (highest first)
            references.sort(key=lambda x: x['confidence'], reverse=True)

            return references

        except Exception as e:
            print(f"Error extracting regulation references: {e}")
            return []
