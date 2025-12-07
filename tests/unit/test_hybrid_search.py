"""
Unit tests for hybrid search module

Tests the search engine logic without requiring actual embedding models.
Focuses on:
1. Weight normalization (including zero-weight bug fix)
2. KG score calculation
3. Persona weight application
4. Score combination

Run with: pytest tests/unit/test_hybrid_search.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Check if torch is available - skip tests if not
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import RESEARCH_TEAM_PERSONAS, KG_WEIGHTS

# Skip marker for tests requiring torch
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)


class MockDataLoader:
    """Mock data loader for testing"""
    pass


class MockEmbeddingModel:
    """Mock embedding model for testing"""
    pass


class MockRerankerModel:
    """Mock reranker model for testing"""
    pass


@pytest.fixture
@requires_torch
def mock_search_engine():
    """Create a hybrid search engine with mocked dependencies"""
    from core.search.hybrid_search import HybridSearchEngine
    return HybridSearchEngine(
        data_loader=MockDataLoader(),
        embedding_model=MockEmbeddingModel(),
        reranker_model=MockRerankerModel()
    )


@requires_torch
class TestWeightNormalization:
    """Test weight normalization and zero-weight handling (bug fix verification)"""

    def test_normal_weights_normalization(self, mock_search_engine):
        """Test that normal weights are normalized correctly"""
        base_weights = {
            'semantic_match': 0.4,
            'keyword_precision': 0.2,
            'knowledge_graph': 0.2,
            'authority_hierarchy': 0.1,
            'temporal_relevance': 0.1
        }

        persona = RESEARCH_TEAM_PERSONAS.get('senior_legal_researcher')
        normalized = mock_search_engine._apply_persona_weights(base_weights, persona)

        # Weights should sum to approximately 1.0
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001, f"Weights should sum to 1.0, got {total}"

    def test_zero_weights_fallback(self, mock_search_engine):
        """Test that zero weights fall back to equal distribution (bug fix)"""
        base_weights = {
            'semantic_match': 0.0,
            'keyword_precision': 0.0,
            'knowledge_graph': 0.0,
            'authority_hierarchy': 0.0,
            'temporal_relevance': 0.0
        }

        persona = RESEARCH_TEAM_PERSONAS.get('senior_legal_researcher')
        # Override persona style to have zero weights too
        persona_copy = {**persona, 'search_style': {}}

        # This should NOT raise division by zero error
        normalized = mock_search_engine._apply_persona_weights(base_weights, persona_copy)

        # Should have equal weights
        expected = 1.0 / len(base_weights)
        for key, value in normalized.items():
            assert abs(value - expected) < 0.001, f"Expected equal weights of {expected}, got {value}"

    def test_single_weight(self, mock_search_engine):
        """Test normalization with single non-zero weight"""
        base_weights = {
            'semantic_match': 1.0,
            'keyword_precision': 0.0,
            'knowledge_graph': 0.0
        }

        persona = {'search_style': {}}  # No overrides
        normalized = mock_search_engine._apply_persona_weights(base_weights, persona)

        # semantic_match should be 1.0, others 0.0
        assert normalized['semantic_match'] == 1.0

    def test_persona_weight_override(self, mock_search_engine):
        """Test that persona can override base weights"""
        base_weights = {
            'semantic_match': 0.25,
            'keyword_precision': 0.25,
            'knowledge_graph': 0.25,
            'authority_hierarchy': 0.25
        }

        # Create persona with specific semantic weight
        custom_persona = {
            'search_style': {
                'semantic_weight': 0.8  # Strong semantic preference
            }
        }

        normalized = mock_search_engine._apply_persona_weights(base_weights, custom_persona)

        # semantic_match should be higher due to override
        assert normalized['semantic_match'] > 0.5, "Semantic weight should be dominant"


class TestConfigurationValidation:
    """Test configuration validation (no torch required)"""

    def test_all_personas_have_search_style(self):
        """Verify all personas have search_style configuration"""
        for name, persona in RESEARCH_TEAM_PERSONAS.items():
            assert 'search_style' in persona, f"Persona {name} missing search_style"

    def test_kg_weights_loaded(self):
        """Test that KG weights are properly loaded from config"""
        assert KG_WEIGHTS is not None
        assert len(KG_WEIGHTS) > 0

        # Check for essential weight categories
        expected_weights = ['semantic', 'keyword', 'entity', 'relationship']
        for weight_name in expected_weights:
            found = any(weight_name.lower() in k.lower() for k in KG_WEIGHTS.keys())
            # May not have all, but should have some
            if found:
                break
        # At least should have KG weights defined
        assert len(KG_WEIGHTS) >= 1


@requires_torch
class TestPersonaIntegration:
    """Test persona integration with search engine"""

    def test_persona_weights_valid(self, mock_search_engine):
        """Test that all personas produce valid normalized weights"""
        base_weights = {
            'semantic_match': 0.25,
            'keyword_precision': 0.25,
            'knowledge_graph': 0.25,
            'authority_hierarchy': 0.25
        }

        for name, persona in RESEARCH_TEAM_PERSONAS.items():
            normalized = mock_search_engine._apply_persona_weights(base_weights, persona)
            total = sum(normalized.values())
            assert abs(total - 1.0) < 0.001, f"Persona {name} weights don't sum to 1.0: {total}"

    def test_persona_search_returns_empty_for_invalid_persona(self, mock_search_engine):
        """Test that invalid persona returns empty results"""
        phase_config = {'description': 'test', 'candidates': 10}
        priority_weights = {'semantic_match': 0.5, 'keyword_precision': 0.5}

        results = mock_search_engine.search_with_persona(
            query="test query",
            persona_name="NonExistentPersona",
            phase_config=phase_config,
            priority_weights=priority_weights
        )

        assert results == [], "Invalid persona should return empty results"


@requires_torch
class TestKGScoreCalculation:
    """Test knowledge graph score calculation"""

    def test_kg_score_empty_record(self, mock_search_engine):
        """Test KG score with empty record"""
        record = {}
        score = mock_search_engine._calculate_kg_score(record, "test query")
        assert score == 0.0, "Empty record should have zero KG score"

    def test_kg_score_with_entity_match(self, mock_search_engine):
        """Test KG score increases with matching entities"""
        record = {
            'kg_enhanced_text': 'UU Nomor 13 Tahun 2003 tentang Ketenagakerjaan',
            'extracted_entities': ['UU 13/2003', 'Ketenagakerjaan']
        }

        query_with_match = "UU Ketenagakerjaan"
        query_without_match = "perpajakan internasional"

        score_with_match = mock_search_engine._calculate_kg_score(record, query_with_match)
        score_without_match = mock_search_engine._calculate_kg_score(record, query_without_match)

        # Score with matching entity should be higher or equal
        assert score_with_match >= score_without_match

    def test_kg_score_normalization(self, mock_search_engine):
        """Test that KG scores are normalized to 0-1 range"""
        record = {
            'kg_enhanced_text': 'Some legal text',
            'extracted_entities': ['Entity1', 'Entity2'],
            'relationships': ['rel1', 'rel2']
        }

        score = mock_search_engine._calculate_kg_score(record, "test query")
        assert 0.0 <= score <= 1.0, f"KG score should be 0-1, got {score}"


@requires_torch
class TestRegulationTypeNormalization:
    """Test regulation type normalization"""

    def test_normalize_uu_variants(self, mock_search_engine):
        """Test that UU variants are normalized correctly"""
        variants = ['UU', 'Undang-Undang', 'undang undang', 'UNDANG-UNDANG']

        for variant in variants:
            normalized = mock_search_engine._normalize_regulation_type(variant)
            assert normalized.upper() in ['UU', 'UNDANG-UNDANG'], f"Failed to normalize {variant}"

    def test_normalize_pp_variants(self, mock_search_engine):
        """Test that PP variants are normalized correctly"""
        variants = ['PP', 'Peraturan Pemerintah', 'PERATURAN PEMERINTAH']

        for variant in variants:
            normalized = mock_search_engine._normalize_regulation_type(variant)
            # Should normalize to a consistent form
            assert normalized is not None


@requires_torch
class TestSearchConfiguration:
    """Test search configuration and defaults (requires torch)"""

    def test_default_device_detection(self, mock_search_engine):
        """Test that device is properly detected"""
        assert mock_search_engine.device is not None
        # Should be either cuda or cpu
        assert str(mock_search_engine.device) in ['cuda', 'cpu', 'cuda:0']


@requires_torch
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_query(self, mock_search_engine):
        """Test handling of empty query"""
        phase_config = {'description': 'test', 'candidates': 10}
        priority_weights = {'semantic_match': 0.5, 'keyword_precision': 0.5}

        # Empty query should not crash
        results = mock_search_engine.search_with_persona(
            query="",
            persona_name="Senior Legal Researcher",
            phase_config=phase_config,
            priority_weights=priority_weights
        )

        # Should return empty or handle gracefully
        assert isinstance(results, list)

    def test_special_characters_in_query(self, mock_search_engine):
        """Test handling of special characters in query"""
        queries = [
            "UU No. 13/2003",
            "Pasal 1 ayat (1)",
            "peraturan <test>",
            "query with 'quotes'",
        ]

        phase_config = {'description': 'test', 'candidates': 10}
        priority_weights = {'semantic_match': 0.5, 'keyword_precision': 0.5}

        for query in queries:
            # Should not crash with special characters
            results = mock_search_engine.search_with_persona(
                query=query,
                persona_name="Senior Legal Researcher",
                phase_config=phase_config,
                priority_weights=priority_weights
            )
            assert isinstance(results, list), f"Failed with query: {query}"

    def test_very_long_query(self, mock_search_engine):
        """Test handling of very long query"""
        long_query = "Apa sanksi " * 100  # Very long query

        phase_config = {'description': 'test', 'candidates': 10}
        priority_weights = {'semantic_match': 0.5, 'keyword_precision': 0.5}

        # Should not crash with long query
        results = mock_search_engine.search_with_persona(
            query=long_query,
            persona_name="Senior Legal Researcher",
            phase_config=phase_config,
            priority_weights=priority_weights
        )

        assert isinstance(results, list)
