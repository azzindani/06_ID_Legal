"""
Unit tests for consensus building module

Run with: pytest tests/unit/test_consensus.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.search.consensus import ConsensusBuilder


@pytest.mark.unit
class TestConsensusBuilder:
    """Test consensus building logic"""

    @pytest.fixture
    def builder(self):
        return ConsensusBuilder()

    def test_build_consensus_single_researcher(self, builder):
        """Test consensus with single researcher results"""
        results = {
            'researcher_1': [
                {'id': 'doc-1', 'score': 0.9},
                {'id': 'doc-2', 'score': 0.8},
            ]
        }

        consensus = builder.build(results)

        assert len(consensus) > 0
        assert consensus[0]['id'] == 'doc-1'

    def test_build_consensus_multiple_researchers(self, builder):
        """Test consensus with multiple researchers agreeing"""
        results = {
            'researcher_1': [
                {'id': 'doc-1', 'score': 0.9},
                {'id': 'doc-2', 'score': 0.7},
            ],
            'researcher_2': [
                {'id': 'doc-1', 'score': 0.85},
                {'id': 'doc-3', 'score': 0.75},
            ],
            'researcher_3': [
                {'id': 'doc-1', 'score': 0.88},
                {'id': 'doc-2', 'score': 0.72},
            ]
        }

        consensus = builder.build(results)

        # doc-1 should be ranked highest (3 researchers agree)
        assert consensus[0]['id'] == 'doc-1'

    def test_consensus_scoring(self, builder):
        """Test that consensus scores are calculated"""
        results = {
            'researcher_1': [{'id': 'doc-1', 'score': 0.9}],
            'researcher_2': [{'id': 'doc-1', 'score': 0.8}],
        }

        consensus = builder.build(results)

        # Should have consensus-related scores
        assert 'consensus_score' in consensus[0] or 'final_score' in consensus[0]

    def test_empty_results(self, builder):
        """Test handling of empty results"""
        results = {}

        consensus = builder.build(results)

        assert consensus == []

    def test_disagreement_handling(self, builder):
        """Test handling when researchers disagree"""
        results = {
            'researcher_1': [{'id': 'doc-1', 'score': 0.9}],
            'researcher_2': [{'id': 'doc-2', 'score': 0.9}],
            'researcher_3': [{'id': 'doc-3', 'score': 0.9}],
        }

        consensus = builder.build(results)

        # Should still produce results
        assert len(consensus) > 0


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
