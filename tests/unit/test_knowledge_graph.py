"""
Unit tests for Knowledge Graph module

Run with: pytest tests/unit/test_knowledge_graph.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.knowledge_graph import KnowledgeGraphCore, RelationshipGraph, CommunityDetector


@pytest.mark.unit
class TestKnowledgeGraphCore:
    """Test KG core functionality"""

    @pytest.fixture
    def kg_core(self):
        return KnowledgeGraphCore()

    def test_extract_regulation(self, kg_core):
        """Test regulation extraction"""
        text = "Berdasarkan UU No. 13 Tahun 2003 tentang Ketenagakerjaan"
        entities = kg_core.extract_entities(text)

        assert 'regulation' in entities
        assert len(entities['regulation']) > 0

    def test_extract_article(self, kg_core):
        """Test article extraction"""
        text = "Pasal 1 ayat (1) menyatakan bahwa"
        entities = kg_core.extract_entities(text)

        assert 'article' in entities

    def test_extract_multiple_entities(self, kg_core):
        """Test multiple entity extraction"""
        text = "UU No. 13 Tahun 2003 Pasal 1 tentang sanksi pidana"
        entities = kg_core.extract_entities(text)

        assert len(entities) >= 2

    def test_entity_score_calculation(self, kg_core):
        """Test entity overlap scoring"""
        doc_entities = {
            'regulation': ['UU No. 13 Tahun 2003'],
            'legal_term': ['sanksi', 'pidana']
        }
        query_entities = {
            'regulation': ['UU No. 13 Tahun 2003'],
            'legal_term': ['sanksi']
        }

        score = kg_core.calculate_entity_score(doc_entities, query_entities)
        assert score > 0

    def test_enhance_results(self, kg_core):
        """Test result enhancement"""
        results = [
            {'content': 'UU No. 13 Tahun 2003 tentang sanksi', 'score': 0.8},
            {'content': 'Peraturan umum', 'score': 0.9}
        ]
        query = "UU No. 13 Tahun 2003 sanksi"

        enhanced = kg_core.enhance_results(results, query)

        assert len(enhanced) == 2
        assert 'kg_score' in enhanced[0]


@pytest.mark.unit
class TestRelationshipGraph:
    """Test relationship graph functionality"""

    @pytest.fixture
    def graph(self):
        return RelationshipGraph()

    def test_add_document(self, graph):
        """Test adding documents"""
        graph.add_document('doc-1', {'type': 'UU'})
        stats = graph.get_graph_stats()

        assert stats['nodes'] >= 1

    def test_add_relationship(self, graph):
        """Test adding relationships"""
        graph.add_document('doc-1', {})
        graph.add_document('doc-2', {})
        graph.add_relationship('doc-1', 'doc-2', 'cites')

        stats = graph.get_graph_stats()
        assert stats['edges'] >= 1


@pytest.mark.unit
class TestCommunityDetector:
    """Test community detection"""

    def test_init(self):
        """Test initialization"""
        detector = CommunityDetector()
        assert detector.communities == []

    def test_get_community_stats_empty(self):
        """Test stats with no communities"""
        detector = CommunityDetector()
        stats = detector.get_community_stats()

        assert stats['num_communities'] == 0
