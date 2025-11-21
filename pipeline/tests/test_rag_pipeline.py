"""
RAG Pipeline Tests

Unit and integration tests for the RAG Pipeline module.
Run with: pytest pipeline/tests/test_rag_pipeline.py -v

Test categories:
- Unit tests: Test individual methods without full initialization
- Integration tests: Test complete pipeline with actual models (requires GPU)

File: pipeline/tests/test_rag_pipeline.py
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.rag_pipeline import RAGPipeline, create_pipeline


# =============================================================================
# UNIT TESTS - No GPU required
# =============================================================================

class TestPipelineCreation:
    """Test pipeline instantiation"""

    def test_create_pipeline_default_config(self):
        """Test creating pipeline with default config"""
        pipeline = RAGPipeline()

        assert pipeline is not None
        assert pipeline._initialized is False
        assert pipeline.config is not None
        assert 'final_top_k' in pipeline.config
        assert 'consensus_threshold' in pipeline.config

    def test_create_pipeline_custom_config(self):
        """Test creating pipeline with custom config"""
        config = {
            'final_top_k': 5,
            'temperature': 0.5,
            'max_new_tokens': 4096
        }

        pipeline = RAGPipeline(config)

        assert pipeline.config['final_top_k'] == 5
        assert pipeline.config['temperature'] == 0.5
        assert pipeline.config['max_new_tokens'] == 4096

    def test_create_pipeline_factory_function(self):
        """Test create_pipeline factory function"""
        pipeline = create_pipeline()

        assert isinstance(pipeline, RAGPipeline)
        assert pipeline._initialized is False

    def test_pipeline_has_search_phases(self):
        """Test that pipeline config includes search phases"""
        pipeline = RAGPipeline()

        assert 'search_phases' in pipeline.config
        assert 'initial_scan' in pipeline.config['search_phases']
        assert 'focused_review' in pipeline.config['search_phases']


class TestPipelineConfig:
    """Test configuration management"""

    def test_update_config(self):
        """Test updating configuration"""
        pipeline = RAGPipeline()

        original_top_k = pipeline.config['final_top_k']
        pipeline.update_config(final_top_k=10)

        assert pipeline.config['final_top_k'] == 10
        assert pipeline.config['final_top_k'] != original_top_k

    def test_update_config_multiple(self):
        """Test updating multiple config values"""
        pipeline = RAGPipeline()

        pipeline.update_config(
            temperature=0.3,
            max_new_tokens=1024,
            final_top_k=7
        )

        assert pipeline.config['temperature'] == 0.3
        assert pipeline.config['max_new_tokens'] == 1024
        assert pipeline.config['final_top_k'] == 7

    def test_update_config_unknown_key(self):
        """Test updating with unknown config key (should warn but not fail)"""
        pipeline = RAGPipeline()

        # Should not raise exception
        pipeline.update_config(unknown_key='value')

        # Unknown key should not be added
        assert 'unknown_key' not in pipeline.config


class TestPipelineState:
    """Test pipeline state management"""

    def test_not_initialized_query(self):
        """Test querying without initialization returns error"""
        pipeline = RAGPipeline()

        result = pipeline.query("Test question")

        assert result['success'] is False
        assert 'not initialized' in result['error'].lower()
        assert result['answer'] == ''

    def test_get_pipeline_info_not_initialized(self):
        """Test getting info before initialization"""
        pipeline = RAGPipeline()

        info = pipeline.get_pipeline_info()

        assert info['initialized'] is False
        assert 'dataset_stats' not in info

    def test_shutdown_not_initialized(self):
        """Test shutdown when not initialized (should not fail)"""
        pipeline = RAGPipeline()

        # Should not raise exception
        pipeline.shutdown()

        assert pipeline._initialized is False


# =============================================================================
# INTEGRATION TESTS - Requires GPU and models
# =============================================================================

@pytest.mark.integration
class TestPipelineInitialization:
    """Test full pipeline initialization (requires GPU)"""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing"""
        return RAGPipeline()

    def test_initialize_success(self, pipeline):
        """Test successful initialization"""
        success = pipeline.initialize()

        assert success is True
        assert pipeline._initialized is True
        assert pipeline.embedding_model is not None
        assert pipeline.reranker_model is not None
        assert pipeline.data_loader is not None
        assert pipeline.orchestrator is not None
        assert pipeline.generation_engine is not None

        # Cleanup
        pipeline.shutdown()

    def test_initialize_with_progress_callback(self, pipeline):
        """Test initialization with progress callback"""
        progress_steps = []

        def callback(step_name, current, total):
            progress_steps.append((step_name, current, total))

        success = pipeline.initialize(progress_callback=callback)

        assert success is True
        assert len(progress_steps) == 5
        assert progress_steps[0][0] == "Loading models"
        assert progress_steps[-1][0] == "Finalizing"

        # Cleanup
        pipeline.shutdown()

    def test_initialize_twice(self, pipeline):
        """Test initializing twice (should warn but succeed)"""
        pipeline.initialize()
        success = pipeline.initialize()  # Second call

        assert success is True
        assert pipeline._initialized is True

        # Cleanup
        pipeline.shutdown()

    def test_get_pipeline_info_initialized(self, pipeline):
        """Test getting info after initialization"""
        pipeline.initialize()

        info = pipeline.get_pipeline_info()

        assert info['initialized'] is True
        assert 'dataset_stats' in info
        assert 'generation_info' in info
        assert info['initialization_time'] > 0

        # Cleanup
        pipeline.shutdown()


@pytest.mark.integration
class TestPipelineQuery:
    """Test query execution (requires GPU)"""

    @pytest.fixture
    def initialized_pipeline(self):
        """Create and initialize pipeline"""
        pipeline = RAGPipeline()
        pipeline.initialize()
        yield pipeline
        pipeline.shutdown()

    def test_query_basic(self, initialized_pipeline):
        """Test basic query execution"""
        result = initialized_pipeline.query(
            "Apa definisi pekerja menurut UU Ketenagakerjaan?"
        )

        assert result['success'] is True
        assert len(result['answer']) > 0
        assert 'metadata' in result
        assert result['metadata']['total_time'] > 0

    def test_query_specific_article(self, initialized_pipeline):
        """Test query for specific article"""
        result = initialized_pipeline.query(
            "Apa bunyi Pasal 1 UU Nomor 13 Tahun 2003?"
        )

        assert result['success'] is True
        assert len(result['answer']) > 0

    def test_query_procedural(self, initialized_pipeline):
        """Test procedural query"""
        result = initialized_pipeline.query(
            "Bagaimana prosedur pendirian PT menurut UU Perseroan Terbatas?"
        )

        assert result['success'] is True
        assert len(result['answer']) > 0

    def test_query_sanctions(self, initialized_pipeline):
        """Test sanctions query"""
        result = initialized_pipeline.query(
            "Apa sanksi pelanggaran UU Ketenagakerjaan?"
        )

        assert result['success'] is True
        assert len(result['answer']) > 0

    def test_query_with_history(self, initialized_pipeline):
        """Test query with conversation history"""
        history = [
            {'role': 'user', 'content': 'Apa itu UU Ketenagakerjaan?'},
            {'role': 'assistant', 'content': 'UU Ketenagakerjaan adalah undang-undang yang mengatur tentang ketenagakerjaan di Indonesia.'}
        ]

        result = initialized_pipeline.query(
            "Apa sanksinya?",
            conversation_history=history
        )

        assert result['success'] is True
        assert len(result['answer']) > 0

    def test_query_returns_sources(self, initialized_pipeline):
        """Test that query returns source documents"""
        result = initialized_pipeline.query(
            "Apa definisi pekerja menurut UU Ketenagakerjaan?"
        )

        assert result['success'] is True
        # Sources may or may not be present depending on generation engine
        assert 'sources' in result
        assert 'metadata' in result
        assert 'results_count' in result['metadata']

    def test_query_metadata(self, initialized_pipeline):
        """Test query metadata structure"""
        result = initialized_pipeline.query(
            "Apa definisi pekerja?"
        )

        assert result['success'] is True
        metadata = result['metadata']

        assert 'question' in metadata
        assert 'retrieval_time' in metadata
        assert 'generation_time' in metadata
        assert 'total_time' in metadata
        assert 'results_count' in metadata

        assert metadata['retrieval_time'] > 0
        assert metadata['total_time'] >= metadata['retrieval_time']


@pytest.mark.integration
class TestPipelineStreaming:
    """Test streaming response (requires GPU)"""

    @pytest.fixture
    def initialized_pipeline(self):
        """Create and initialize pipeline"""
        pipeline = RAGPipeline()
        pipeline.initialize()
        yield pipeline
        pipeline.shutdown()

    def test_streaming_basic(self, initialized_pipeline):
        """Test basic streaming"""
        chunks = list(initialized_pipeline.query(
            "Apa definisi pekerja?",
            stream=True
        ))

        assert len(chunks) > 0

        # Check we got tokens and completion
        token_chunks = [c for c in chunks if c.get('type') == 'token']
        complete_chunks = [c for c in chunks if c.get('type') == 'complete']

        assert len(complete_chunks) == 1
        assert complete_chunks[0]['done'] is True
        assert complete_chunks[0]['success'] is True

    def test_streaming_collects_full_answer(self, initialized_pipeline):
        """Test that streaming collects complete answer"""
        chunks = list(initialized_pipeline.query(
            "Apa definisi pekerja?",
            stream=True
        ))

        complete_chunk = next(c for c in chunks if c.get('type') == 'complete')

        assert len(complete_chunk['answer']) > 0
        assert 'metadata' in complete_chunk


@pytest.mark.integration
class TestPipelineShutdown:
    """Test pipeline shutdown and cleanup"""

    def test_shutdown_clears_resources(self):
        """Test that shutdown clears all resources"""
        pipeline = RAGPipeline()
        pipeline.initialize()

        assert pipeline._initialized is True
        assert pipeline.embedding_model is not None

        pipeline.shutdown()

        assert pipeline._initialized is False
        assert pipeline.embedding_model is None
        assert pipeline.reranker_model is None
        assert pipeline.data_loader is None
        assert pipeline.orchestrator is None
        assert pipeline.generation_engine is None

    def test_context_manager(self):
        """Test context manager usage"""
        with RAGPipeline() as pipeline:
            assert pipeline._initialized is True

            result = pipeline.query("Apa definisi pekerja?")
            assert result['success'] is True

        # After context exit, pipeline should be shut down
        assert pipeline._initialized is False


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePerformance:
    """Performance benchmarks (requires GPU)"""

    @pytest.fixture
    def initialized_pipeline(self):
        """Create and initialize pipeline"""
        pipeline = RAGPipeline()
        pipeline.initialize()
        yield pipeline
        pipeline.shutdown()

    def test_query_time_reasonable(self, initialized_pipeline):
        """Test that query completes in reasonable time"""
        import time

        start = time.time()
        result = initialized_pipeline.query("Apa definisi pekerja?")
        elapsed = time.time() - start

        assert result['success'] is True
        # Query should complete within 60 seconds
        assert elapsed < 60, f"Query took too long: {elapsed:.2f}s"

    def test_multiple_queries(self, initialized_pipeline):
        """Test multiple sequential queries"""
        queries = [
            "Apa definisi pekerja?",
            "Apa sanksi pelanggaran UU Ketenagakerjaan?",
            "Bagaimana prosedur PHK?"
        ]

        results = []
        for query in queries:
            result = initialized_pipeline.query(query)
            results.append(result)

        # All queries should succeed
        for result in results:
            assert result['success'] is True
            assert len(result['answer']) > 0


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run RAG Pipeline tests')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    pytest_args = ['pipeline/tests/test_rag_pipeline.py']

    if args.verbose:
        pytest_args.append('-v')

    if args.unit:
        pytest_args.extend(['-m', 'not integration'])
    elif args.integration:
        pytest_args.extend(['-m', 'integration'])
    elif args.all:
        pass  # Run all tests
    else:
        # Default: run unit tests only
        pytest_args.extend(['-m', 'not integration'])

    pytest.main(pytest_args)
