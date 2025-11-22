"""
Pytest Configuration and Shared Fixtures

This module provides shared fixtures for all tests in the project.
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def test_config():
    """Base test configuration - shared across all tests in session"""
    return {
        'embedding_model': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        'llm_model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'dataset_name': 'Azzindani/ID_REG_DB_2510',
        'max_results': 10,
        'semantic_weight': 0.7,
        'keyword_weight': 0.3,
    }


@pytest.fixture(scope='session')
def mock_config():
    """Mock configuration for unit tests - shared across session"""
    config = MagicMock()
    config.EMBEDDING_MODEL = 'test-embedding-model'
    config.LLM_MODEL = 'test-llm-model'
    config.DATASET_NAME = 'test-dataset'
    config.MAX_RESULTS = 5
    return config


# =============================================================================
# Shared Initialization Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def initialized_pipeline():
    """
    Initialize RAG pipeline once for all tests in session.
    Use for integration tests that need the full pipeline.
    """
    try:
        from pipeline import RAGPipeline
        pipeline = RAGPipeline()
        if pipeline.initialize():
            yield pipeline
            pipeline.shutdown()
        else:
            pytest.skip("Failed to initialize pipeline")
    except ImportError:
        pytest.skip("Pipeline dependencies not available")


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_document():
    """Sample legal document for testing"""
    return {
        'id': 'doc-001',
        'content': 'Pasal 1 ayat (1) UU No. 13 Tahun 2003 tentang Ketenagakerjaan menyatakan...',
        'regulation_type': 'Undang-Undang',
        'regulation_number': '13',
        'year': '2003',
        'about': 'Ketenagakerjaan',
        'chunk_id': 1
    }


@pytest.fixture
def sample_documents():
    """Multiple sample documents for testing"""
    return [
        {
            'id': f'doc-{i:03d}',
            'content': f'Test content for document {i}',
            'regulation_type': 'Undang-Undang',
            'regulation_number': str(i),
            'year': '2003',
            'about': f'Test Regulation {i}',
            'chunk_id': i
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "Apa sanksi pelanggaran UU Ketenagakerjaan?"


@pytest.fixture
def sample_queries():
    """Multiple sample queries for testing"""
    return [
        "Apa itu UU Ketenagakerjaan?",
        "Apa sanksi pelanggaran UU Perlindungan Konsumen?",
        "Bagaimana prosedur PHK menurut hukum?",
        "Apa hak karyawan kontrak?",
        "Berapa denda maksimal pelanggaran lingkungan?"
    ]


@pytest.fixture
def sample_session_data():
    """Sample session data for testing"""
    return {
        'id': 'test-session-123',
        'created_at': '2024-01-15T10:30:00',
        'updated_at': '2024-01-15T10:35:00',
        'turns': [
            {
                'turn_number': 1,
                'timestamp': '2024-01-15T10:30:00',
                'query': 'Apa sanksi pelanggaran UU Ketenagakerjaan?',
                'answer': 'Sanksi pelanggaran UU Ketenagakerjaan meliputi sanksi administratif dan pidana.',
                'metadata': {
                    'total_time': 5.2,
                    'tokens_generated': 150,
                    'query_type': 'sanctions',
                    'citations': [
                        {
                            'regulation_type': 'Undang-Undang',
                            'regulation_number': '13',
                            'year': '2003',
                            'about': 'Ketenagakerjaan'
                        }
                    ]
                }
            }
        ],
        'metadata': {
            'total_queries': 1,
            'total_tokens': 150,
            'total_time': 5.2,
            'regulations_cited': ['UU 13/2003']
        }
    }


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        {
            'document': {
                'id': 'doc-001',
                'content': 'Sanksi pidana berupa kurungan...',
                'regulation_type': 'Undang-Undang',
                'regulation_number': '13',
                'year': '2003',
                'about': 'Ketenagakerjaan'
            },
            'score': 0.95,
            'rank': 1
        },
        {
            'document': {
                'id': 'doc-002',
                'content': 'Denda administratif maksimal...',
                'regulation_type': 'Undang-Undang',
                'regulation_number': '13',
                'year': '2003',
                'about': 'Ketenagakerjaan'
            },
            'score': 0.87,
            'rank': 2
        }
    ]


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model"""
    model = MagicMock()
    model.encode.return_value = [[0.1] * 768]  # Fake embedding
    return model


@pytest.fixture
def mock_llm_model():
    """Mock LLM model"""
    model = MagicMock()
    model.generate.return_value = "This is a test response."
    return model


@pytest.fixture
def mock_dataset():
    """Mock dataset"""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=100)
    dataset.__getitem__ = MagicMock(return_value={
        'content': 'Test content',
        'regulation_type': 'UU',
        'regulation_number': '1',
        'year': '2020',
        'about': 'Test'
    })
    return dataset


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Temporary file for testing"""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Test content")
    return file_path


# =============================================================================
# Skip Markers
# =============================================================================

@pytest.fixture
def skip_without_gpu():
    """Skip test if GPU not available"""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.fixture
def skip_without_models():
    """Skip test if models not loaded"""
    # This would check if models are available
    # For now, always skip in CI
    if os.environ.get('CI'):
        pytest.skip("Skipping model tests in CI")


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture
def cleanup_gpu():
    """
    Cleanup GPU memory after test. Use explicitly for tests that load models.
    NOT autouse - only add to tests that need it.
    """
    yield
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: GPU required tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Auto-mark tests based on path
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "test_rag" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
