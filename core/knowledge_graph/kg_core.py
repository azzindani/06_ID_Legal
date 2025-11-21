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
            'regulation': r'(?:UU|PP|Perpres|Perda|Permen)\s*(?:No\.?\s*)?\d+\s*(?:Tahun\s*)?\d{4}',
            'article': r'Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?',
            'institution': r'(?:Menteri|Presiden|DPR|Pemerintah|Pengadilan)\s+\w+',
            'legal_term': r'(?:sanksi|pidana|denda|hukuman|pelanggaran|ketentuan)',
        }

        # Entity weights for scoring
        self.entity_weights = {
            'regulation': 1.5,
            'article': 1.2,
            'institution': 1.0,
            'legal_term': 0.8,
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
