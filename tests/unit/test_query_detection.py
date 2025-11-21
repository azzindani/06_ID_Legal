"""
Unit tests for query detection module

Run with: pytest tests/unit/test_query_detection.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.search.query_detection import QueryDetector


@pytest.mark.unit
class TestQueryDetector:
    """Test query type detection"""

    @pytest.fixture
    def detector(self):
        return QueryDetector()

    def test_detect_sanctions_query(self, detector):
        """Test detection of sanctions queries"""
        query = "Apa sanksi pelanggaran UU Ketenagakerjaan?"
        result = detector.detect(query)

        assert result['query_type'] == 'sanctions'
        assert result['confidence'] > 0.5

    def test_detect_definition_query(self, detector):
        """Test detection of definition queries"""
        query = "Apa definisi tenaga kerja?"
        result = detector.detect(query)

        assert result['query_type'] == 'definition'

    def test_detect_procedure_query(self, detector):
        """Test detection of procedure queries"""
        query = "Bagaimana prosedur PHK?"
        result = detector.detect(query)

        assert result['query_type'] == 'procedure'

    def test_detect_requirement_query(self, detector):
        """Test detection of requirement queries"""
        query = "Apa syarat mendirikan PT?"
        result = detector.detect(query)

        assert result['query_type'] == 'requirement'

    def test_detect_general_query(self, detector):
        """Test detection of general queries"""
        query = "Ceritakan tentang hukum Indonesia"
        result = detector.detect(query)

        assert result['query_type'] == 'general'

    def test_empty_query(self, detector):
        """Test handling of empty query"""
        result = detector.detect("")

        assert result['query_type'] == 'general'

    def test_extract_entities(self, detector):
        """Test entity extraction from query"""
        query = "Apa sanksi UU No. 13 Tahun 2003?"
        result = detector.detect(query)

        # Should extract regulation reference
        assert 'entities' in result or 'regulation_refs' in result


@pytest.mark.unit
class TestQueryKeywords:
    """Test keyword extraction"""

    @pytest.fixture
    def detector(self):
        return QueryDetector()

    def test_extract_keywords(self, detector):
        """Test keyword extraction"""
        query = "sanksi pidana pelanggaran ketenagakerjaan"
        result = detector.detect(query)

        assert 'keywords' in result
        assert len(result['keywords']) > 0

    def test_stopword_removal(self, detector):
        """Test that stopwords are removed"""
        query = "apa yang dimaksud dengan tenaga kerja"
        result = detector.detect(query)

        keywords = result.get('keywords', [])
        # Common stopwords should be filtered
        assert 'yang' not in keywords
        assert 'dengan' not in keywords
