"""
Unit tests for consensus building module

Run with: pytest tests/unit/test_consensus.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.search.consensus import ConsensusBuilder


@pytest.mark.integration  # Requires complex data structure from real pipeline
class TestConsensusBuilder:
    """Test consensus building logic"""

    @pytest.fixture
    def builder(self):
        return ConsensusBuilder({})

    def test_build_consensus_single_researcher(self, builder):
        """Test consensus with single researcher results"""
        research_data = {
            'all_results': [
                {'record': {'global_id': 'doc-1'}, 'scores': {'final': 0.9, 'semantic': 0.85, 'keyword': 0.8, 'kg': 0.7, 'authority': 0.8, 'temporal': 0.9, 'completeness': 0.85}, 'metadata': {'persona': 'analyst'}},
                {'record': {'global_id': 'doc-2'}, 'scores': {'final': 0.8, 'semantic': 0.75, 'keyword': 0.7, 'kg': 0.6, 'authority': 0.7, 'temporal': 0.8, 'completeness': 0.75}, 'metadata': {'persona': 'analyst'}},
            ]
        }
        team = ['analyst']

        consensus = builder.build_consensus(research_data, team)

        assert 'validated_results' in consensus

    def test_build_consensus_multiple_researchers(self, builder):
        """Test consensus with multiple researchers agreeing"""
        research_data = {
            'all_results': [
                {'record': {'global_id': 'doc-1'}, 'scores': {'final': 0.9, 'semantic': 0.85, 'keyword': 0.8, 'kg': 0.7, 'authority': 0.8, 'temporal': 0.9, 'completeness': 0.85}, 'metadata': {'persona': 'analyst'}},
                {'record': {'global_id': 'doc-1'}, 'scores': {'final': 0.85, 'semantic': 0.8, 'keyword': 0.75, 'kg': 0.65, 'authority': 0.75, 'temporal': 0.85, 'completeness': 0.8}, 'metadata': {'persona': 'specialist'}},
                {'record': {'global_id': 'doc-2'}, 'scores': {'final': 0.7, 'semantic': 0.65, 'keyword': 0.6, 'kg': 0.5, 'authority': 0.6, 'temporal': 0.7, 'completeness': 0.65}, 'metadata': {'persona': 'analyst'}},
            ]
        }
        team = ['analyst', 'specialist']

        consensus = builder.build_consensus(research_data, team)

        assert 'validated_results' in consensus
        assert 'agreement_level' in consensus

    def test_consensus_scoring(self, builder):
        """Test that consensus scores are calculated"""
        research_data = {
            'all_results': [
                {'record': {'global_id': 'doc-1'}, 'scores': {'final': 0.9, 'semantic': 0.85, 'keyword': 0.8, 'kg': 0.7, 'authority': 0.8, 'temporal': 0.9, 'completeness': 0.85}, 'metadata': {'persona': 'analyst'}},
                {'record': {'global_id': 'doc-1'}, 'scores': {'final': 0.8, 'semantic': 0.75, 'keyword': 0.7, 'kg': 0.6, 'authority': 0.7, 'temporal': 0.8, 'completeness': 0.75}, 'metadata': {'persona': 'specialist'}},
            ]
        }
        team = ['analyst', 'specialist']

        consensus = builder.build_consensus(research_data, team)

        # Should have consensus-related data
        assert 'consensus_scores' in consensus

    def test_empty_results(self, builder):
        """Test handling of empty results"""
        research_data = {'all_results': []}
        team = ['analyst']

        consensus = builder.build_consensus(research_data, team)

        assert consensus['validated_results'] == []

    def test_disagreement_handling(self, builder):
        """Test handling when researchers disagree (each prefers a different doc)"""
        research_data = {
            'all_results': [
                # Each researcher prefers a different document - no overlap
                {'record': {'global_id': 'doc-1'}, 'scores': {'final': 0.9, 'semantic': 0.85, 'keyword': 0.8, 'kg': 0.7, 'authority': 0.8, 'temporal': 0.9, 'completeness': 0.85}, 'metadata': {'persona': 'analyst'}},
                {'record': {'global_id': 'doc-2'}, 'scores': {'final': 0.9, 'semantic': 0.85, 'keyword': 0.8, 'kg': 0.7, 'authority': 0.8, 'temporal': 0.9, 'completeness': 0.85}, 'metadata': {'persona': 'specialist'}},
                {'record': {'global_id': 'doc-3'}, 'scores': {'final': 0.9, 'semantic': 0.85, 'keyword': 0.8, 'kg': 0.7, 'authority': 0.8, 'temporal': 0.9, 'completeness': 0.85}, 'metadata': {'persona': 'generalist'}},
            ]
        }
        team = ['analyst', 'specialist', 'generalist']

        consensus = builder.build_consensus(research_data, team)

        # Should still produce results even with disagreement
        assert 'validated_results' in consensus


@pytest.mark.unit
class TestConsensusConfiguration:
    """Test consensus builder configuration"""

    def test_custom_threshold(self):
        """Test custom consensus threshold"""
        builder = ConsensusBuilder({'consensus_threshold': 0.8})

        assert builder.config.get('consensus_threshold') == 0.8

    def test_custom_min_agreement(self):
        """Test custom minimum agreement"""
        builder = ConsensusBuilder({'min_agreement': 2})

        assert builder.config.get('min_agreement') == 2
