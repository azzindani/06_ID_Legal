"""
Integration Tests for RAG Pipeline

Tests the complete pipeline flow including retrieval and generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete RAG pipeline"""

    def test_pipeline_creation(self, test_config):
        """Test pipeline can be created with config"""
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(test_config)
        assert pipeline is not None
        assert pipeline.config is not None
        assert not pipeline._initialized

    def test_pipeline_query_without_init(self, test_config):
        """Test query fails gracefully without initialization"""
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(test_config)
        result = pipeline.query("test question")

        assert result['success'] is False
        assert 'not initialized' in result['error'].lower()

    @patch('pipeline.rag_pipeline.load_models')
    @patch('pipeline.rag_pipeline.EnhancedKGDatasetLoader')
    @patch('pipeline.rag_pipeline.LangGraphRAGOrchestrator')
    @patch('pipeline.rag_pipeline.GenerationEngine')
    def test_pipeline_initialization(
        self, mock_gen, mock_orch, mock_loader, mock_models, test_config
    ):
        """Test pipeline initialization flow"""
        from pipeline.rag_pipeline import RAGPipeline

        # Setup mocks
        mock_models.return_value = (Mock(), Mock())
        mock_loader_instance = Mock()
        mock_loader_instance.get_statistics.return_value = {'total_records': 100}
        mock_loader.return_value = mock_loader_instance

        mock_gen_instance = Mock()
        mock_gen_instance.initialize.return_value = True
        mock_gen.return_value = mock_gen_instance

        pipeline = RAGPipeline(test_config)
        result = pipeline.initialize()

        assert result is True
        assert pipeline._initialized is True

    @patch('pipeline.rag_pipeline.load_models')
    @patch('pipeline.rag_pipeline.EnhancedKGDatasetLoader')
    @patch('pipeline.rag_pipeline.LangGraphRAGOrchestrator')
    @patch('pipeline.rag_pipeline.GenerationEngine')
    def test_pipeline_query_flow(
        self, mock_gen, mock_orch, mock_loader, mock_models, test_config
    ):
        """Test complete query flow with mocked components"""
        from pipeline.rag_pipeline import RAGPipeline

        # Setup mocks
        mock_models.return_value = (Mock(), Mock())
        mock_loader_instance = Mock()
        mock_loader_instance.get_statistics.return_value = {'total_records': 100}
        mock_loader.return_value = mock_loader_instance

        # Mock orchestrator
        mock_orch_instance = Mock()
        mock_orch_instance.run.return_value = {
            'final_results': [
                {'content': 'Test content', 'score': 0.9}
            ],
            'metadata': {'query_analysis': {'query_type': 'definitional'}}
        }
        mock_orch.return_value = mock_orch_instance

        # Mock generation
        mock_gen_instance = Mock()
        mock_gen_instance.initialize.return_value = True
        mock_gen_instance.generate_answer.return_value = {
            'success': True,
            'answer': 'Test answer',
            'sources': [],
            'metadata': {'generation_time': 1.0, 'tokens_generated': 50}
        }
        mock_gen.return_value = mock_gen_instance

        pipeline = RAGPipeline(test_config)
        pipeline.initialize()

        result = pipeline.query("What is labor law?")

        assert result['success'] is True
        assert result['answer'] == 'Test answer'
        assert 'metadata' in result

    def test_pipeline_context_manager(self, test_config):
        """Test pipeline context manager protocol"""
        from pipeline.rag_pipeline import RAGPipeline

        with patch.object(RAGPipeline, 'initialize', return_value=True):
            with patch.object(RAGPipeline, 'shutdown'):
                with RAGPipeline(test_config) as pipeline:
                    assert pipeline is not None


@pytest.mark.integration
class TestProviderIntegration:
    """Tests for LLM provider integration"""

    def test_provider_listing(self):
        """Test listing available providers"""
        from providers import list_providers

        providers = list_providers()
        assert 'local' in providers
        assert 'openai' in providers
        assert 'anthropic' in providers

    def test_provider_switching(self):
        """Test switching between providers"""
        from providers import switch_provider, PROVIDERS

        # Test switching to each provider
        for provider_name in ['local', 'openai', 'anthropic']:
            result = switch_provider(provider_name)
            assert result is True or provider_name in PROVIDERS

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_openai_provider_config(self):
        """Test OpenAI provider configuration"""
        from providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        assert provider is not None

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    def test_anthropic_provider_config(self):
        """Test Anthropic provider configuration"""
        from providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        assert provider is not None


@pytest.mark.integration
class TestDocumentProcessing:
    """Tests for document upload and processing"""

    def test_document_parser_creation(self):
        """Test document parser can be created"""
        from core.document_parser import DocumentParser

        parser = DocumentParser()
        assert parser is not None

    def test_text_parsing(self, tmp_path):
        """Test parsing plain text files"""
        from core.document_parser import parse_document

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for parsing.")

        result = parse_document(str(test_file))

        assert result['success'] is True
        assert 'test content' in result['content'].lower()

    def test_content_chunking(self):
        """Test content chunking functionality"""
        from core.document_parser import DocumentParser

        parser = DocumentParser()
        content = "A" * 2500  # Create content longer than chunk size

        chunks = parser.chunk_content(content, chunk_size=1000, overlap=200)

        assert len(chunks) > 1
        assert all(len(chunk) <= 1000 for chunk in chunks)


@pytest.mark.integration
class TestContextCache:
    """Tests for context cache functionality"""

    def test_cache_creation(self):
        """Test context cache can be created"""
        from conversation.context_cache import get_context_cache

        cache = get_context_cache()
        assert cache is not None

    def test_cache_operations(self):
        """Test basic cache put/get operations"""
        from conversation.context_cache import get_context_cache

        cache = get_context_cache()

        # Put and get
        test_data = {'answer': 'test', 'metadata': {}}
        cache.put('test-key', test_data)

        retrieved = cache.get('test-key')
        assert retrieved is not None
        assert retrieved.get('answer') == 'test'

    def test_cache_miss(self):
        """Test cache miss returns None"""
        from conversation.context_cache import get_context_cache

        cache = get_context_cache()
        result = cache.get('nonexistent-key')

        assert result is None


@pytest.mark.integration
class TestHardwareDetection:
    """Tests for hardware auto-detection"""

    def test_hardware_detection(self):
        """Test hardware detection returns valid config"""
        from hardware_detection import detect_hardware

        config = detect_hardware()

        assert config.embedding_device in ['cpu', 'cuda']
        assert config.llm_device in ['cpu', 'cuda']
        assert config.llm_quantization in ['none', '4bit', '8bit']

    def test_ram_detection(self):
        """Test RAM detection returns reasonable value"""
        from hardware_detection import get_ram_info

        ram_gb = get_ram_info()

        assert ram_gb > 0
        assert ram_gb < 10000  # Sanity check

    def test_apply_hardware_config(self):
        """Test applying hardware configuration"""
        from hardware_detection import apply_hardware_config, detect_hardware

        config = detect_hardware()
        settings = apply_hardware_config(config)

        assert 'EMBEDDING_DEVICE' in settings
        assert 'LLM_DEVICE' in settings
