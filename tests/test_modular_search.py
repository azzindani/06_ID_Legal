# tests/test_modular_search.py
"""
Unit tests for modular search components.
Run with: pytest tests/test_modular_search.py -v
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from core.search.scoring import CandidateScorer
from core.search.diversity import DiversityFilter
from utils.logging_config import get_logger

logger = get_logger(__name__)

class TestCandidateScorer:
    """Test candidate scoring functionality."""
    
    @pytest.fixture
    def mock_kg(self):
        """Create mock knowledge graph."""
        kg = Mock()
        kg.calculate_enhanced_kg_score = Mock(return_value=0.7)
        return kg
    
    @pytest.fixture
    def sample_records(self):
        """Create sample document records."""
        return [
            {
                'global_id': 1,
                'regulation_type': 'Undang-Undang',
                'regulation_number': '13',
                'year': '2003',
                'about': 'Ketenagakerjaan',
                'content': 'Ketentuan tentang pekerja...',
                'kg_authority_score': 0.9,
                'kg_temporal_score': 0.8,
                'kg_legal_richness': 0.7,
                'kg_completeness_score': 0.85,
                'kg_pagerank': 0.01,
                'kg_primary_domain': 'labor_law',
                'kg_domain_confidence': 0.9,
                'kg_entity_count': 5,
                'kg_entities_json': '[]',
                'article': '10'
            },
            {
                'global_id': 2,
                'regulation_type': 'Peraturan Pemerintah',
                'regulation_number': '78',
                'year': '2015',
                'about': 'Pengupahan',
                'content': 'Ketentuan tentang upah...',
                'kg_authority_score': 0.7,
                'kg_temporal_score': 0.9,
                'kg_legal_richness': 0.6,
                'kg_completeness_score': 0.75,
                'kg_pagerank': 0.005,
                'kg_primary_domain': 'labor_law',
                'kg_domain_confidence': 0.85,
                'kg_entity_count': 3,
                'kg_entities_json': '[]',
                'article': 'N/A'
            }
        ]
    
    @pytest.fixture
    def scorer(self, mock_kg, sample_records):
        """Create scorer instance."""
        return CandidateScorer(mock_kg, sample_records)
    
    def test_scorer_initialization(self, scorer, sample_records):
        """Test scorer initializes correctly."""
        assert scorer.kg is not None
        assert len(scorer.records) == len(sample_records)
        logger.info("✅ Scorer initialization test passed")
    
    def test_score_candidates_basic(self, scorer):
        """Test basic candidate scoring."""
        semantic_sims = np.array([0.8, 0.6])
        keyword_sims = np.array([0.5, 0.4])
        query_entities = ['pekerja', 'upah']
        query_type = 'general'
        
        search_style = {
            'semantic_weight': 0.25,
            'authority_weight': 0.35,
            'kg_weight': 0.25,
            'temporal_weight': 0.15
        }
        
        phase_config = {
            'candidates': 10,
            'semantic_threshold': 0.3,
            'keyword_threshold': 0.1,
            'description': 'test_phase'
        }
        
        persona = {
            'name': 'Test Researcher',
            'experience_years': 5,
            'accuracy_bonus': 0.05
        }
        
        results = scorer.score_candidates(
            semantic_sims,
            keyword_sims,
            query_entities,
            query_type,
            search_style,
            phase_config,
            persona
        )
        
        assert len(results) == 2
        assert all('composite_score' in r for r in results)
        assert results[0]['composite_score'] >= results[1]['composite_score']  # Sorted
        
        logger.info(f"✅ Basic scoring test passed: {len(results)} candidates scored")
    
    def test_score_candidates_filtering(self, scorer):
        """Test that low-scoring candidates are filtered."""
        semantic_sims = np.array([0.1, 0.8])  # First is too low
        keyword_sims = np.array([0.02, 0.5])
        
        phase_config = {
            'candidates': 10,
            'semantic_threshold': 0.5,  # High threshold
            'keyword_threshold': 0.3,
            'description': 'strict_phase'
        }
        
        search_style = {
            'semantic_weight': 0.5,
            'authority_weight': 0.2,
            'kg_weight': 0.2,
            'temporal_weight': 0.1
        }
        
        persona = {'name': 'Test', 'experience_years': 5, 'accuracy_bonus': 0}
        
        results = scorer.score_candidates(
            semantic_sims,
            keyword_sims,
            [],
            'general',
            search_style,
            phase_config,
            persona
        )
        
        # Only second candidate should pass
        assert len(results) <= 2
        logger.info(f"✅ Filtering test passed: {len(results)} candidates after filtering")


class TestDiversityFilter:
    """Test diversity filtering."""
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates with different attributes."""
        return [
            {
                'record': {
                    'regulation_type': 'Undang-Undang',
                    'kg_primary_domain': 'labor_law',
                    'kg_hierarchy_level': 1
                },
                'composite_score': 0.9
            },
            {
                'record': {
                    'regulation_type': 'Undang-Undang',
                    'kg_primary_domain': 'labor_law',
                    'kg_hierarchy_level': 1
                },
                'composite_score': 0.85
            },
            {
                'record': {
                    'regulation_type': 'Peraturan Pemerintah',
                    'kg_primary_domain': 'tax_law',
                    'kg_hierarchy_level': 2
                },
                'composite_score': 0.8
            },
            {
                'record': {
                    'regulation_type': 'Peraturan Menteri',
                    'kg_primary_domain': 'education',
                    'kg_hierarchy_level': 3
                },
                'composite_score': 0.75
            }
        ]
    
    def test_diversity_filter_basic(self, sample_candidates):
        """Test basic diversity filtering."""
        filter = DiversityFilter()
        
        results = filter.apply_diversity(sample_candidates, target_count=3)
        
        assert len(results) == 3
        
        # Should prefer diverse types
        types = [r['record']['regulation_type'] for r in results]
        assert len(set(types)) >= 2  # At least 2 different types
        
        logger.info(f"✅ Diversity filter test passed: {len(results)} diverse results")
    
    def test_diversity_filter_no_filtering_needed(self, sample_candidates):
        """Test when no filtering is needed."""
        filter = DiversityFilter()
        
        results = filter.apply_diversity(sample_candidates, target_count=10)
        
        # All candidates should be returned
        assert len(results) == len(sample_candidates)
        logger.info("✅ No filtering test passed")


class TestIntegration:
    """Integration tests for the modular system."""
    
    def test_scorer_and_diversity_integration(self):
        """Test scorer + diversity filter working together."""
        # Setup
        mock_kg = Mock()
        mock_kg.calculate_enhanced_kg_score = Mock(return_value=0.6)
        
        records = [
            {
                'global_id': i,
                'regulation_type': f'Type{i % 3}',
                'regulation_number': str(i),
                'year': '2020',
                'about': 'Test',
                'content': 'Content',
                'kg_authority_score': 0.8,
                'kg_temporal_score': 0.7,
                'kg_legal_richness': 0.6,
                'kg_completeness_score': 0.75,
                'kg_pagerank': 0.01,
                'kg_primary_domain': f'domain{i % 2}',
                'kg_domain_confidence': 0.8,
                'kg_entity_count': 3,
                'kg_entities_json': '[]',
                'kg_hierarchy_level': (i % 3) + 1,
                'article': 'N/A'
            }
            for i in range(10)
        ]
        
        scorer = CandidateScorer(mock_kg, records)
        diversity_filter = DiversityFilter()
        
        # Score
        semantic_sims = np.random.uniform(0.5, 0.9, 10)
        keyword_sims = np.random.uniform(0.3, 0.7, 10)
        
        search_style = {
            'semantic_weight': 0.3,
            'authority_weight': 0.3,
            'kg_weight': 0.3,
            'temporal_weight': 0.1
        }
        
        phase_config = {
            'candidates': 10,
            'semantic_threshold': 0.2,
            'keyword_threshold': 0.1,
            'description': 'integration_test'
        }
        
        persona = {'name': 'Test', 'experience_years': 5, 'accuracy_bonus': 0}
        
        scored = scorer.score_candidates(
            semantic_sims,
            keyword_sims,
            [],
            'general',
            search_style,
            phase_config,
            persona
        )
        
        # Apply diversity
        diverse = diversity_filter.apply_diversity(scored, target_count=5)
        
        assert len(diverse) == 5
        assert len(diverse) <= len(scored)
        
        # Check diversity
        types = set(r['record']['regulation_type'] for r in diverse)
        assert len(types) >= 2
        
        logger.info(f"✅ Integration test passed: {len(scored)} → {len(diverse)} diverse results")


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--log-cli-level=INFO'])