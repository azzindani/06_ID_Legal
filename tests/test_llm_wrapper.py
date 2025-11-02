# tests/test_llm_wrapper.py
"""
Unit tests for LLM wrapper system.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from core.models.base_llm import GenerationConfig, LLMResponse, LLMType
from core.models.local_llm import LocalLLM
from core.models.api_llm import APILLM, APIProvider
from core.models.llm_factory import LLMFactory
from utils.retry_handler import RetryHandler

class TestGenerationConfig:
    """Test generation configuration."""
    
    def test_valid_config(self):
        """Test creating valid config."""
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        assert config.max_new_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.9
    
    def test_invalid_temperature(self):
        """Test invalid temperature raises error."""
        with pytest.raises(ValueError):
            GenerationConfig(temperature=3.0)
    
    def test_invalid_top_p(self):
        """Test invalid top_p raises error."""
        with pytest.raises(ValueError):
            GenerationConfig(top_p=1.5)


class TestLLMResponse:
    """Test LLM response object."""
    
    def test_response_creation(self):
        """Test creating response."""
        response = LLMResponse(
            text="Generated text",
            finish_reason='stop',
            tokens_used=50,
            model='test-model'
        )
        
        assert response.text == "Generated text"
        assert response.finish_reason == 'stop'
        assert response.tokens_used == 50
        assert str(response) == "Generated text"


class TestRetryHandler:
    """Test retry logic."""
    
    def test_success_first_try(self):
        """Test successful execution on first try."""
        retry_handler = RetryHandler(max_retries=3)
        
        def success_func():
            return "success"
        
        result = retry_handler.execute(success_func)
        assert result == "success"
    
    def test_success_after_retries(self):
        """Test success after some failures."""
        retry_handler = RetryHandler(max_retries=3, initial_delay=0.1)
        
        attempt = {'count': 0}
        
        def eventually_succeeds():
            attempt['count'] += 1
            if attempt['count'] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = retry_handler.execute(eventually_succeeds)
        assert result == "success"
        assert attempt['count'] == 3
    
    def test_all_retries_fail(self):
        """Test all retries exhausted."""
        retry_handler = RetryHandler(max_retries=2, initial_delay=0.1)
        
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            retry_handler.execute(always_fails)


class TestLocalLLM:
    """Test local LLM wrapper."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.eval = Mock()
        model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        model.num_parameters = Mock(return_value=7000000000)
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="Generated text")
        tokenizer.eos_token_id = 0
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        return tokenizer
    
    @patch('core.models.local_llm.AutoModelForCausalLM')
    @patch('core.models.local_llm.AutoTokenizer')
    def test_load_model(self, mock_tokenizer_class, mock_model_class, mock_model, mock_tokenizer):
        """Test loading local model."""
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        llm = LocalLLM('test-model', {}, device='cpu')
        llm.load()
        
        assert llm._loaded
        assert llm.model is not None
        assert llm.tokenizer is not None
    
    def test_unload_model(self):
        """Test unloading model."""
        llm = LocalLLM('test-model', {}, device='cpu')
        llm.model = Mock()
        llm.tokenizer = Mock()
        llm._loaded = True
        
        llm.unload()
        
        assert llm.model is None
        assert llm.tokenizer is None
        assert not llm._loaded
    
    @patch('core.models.local_llm.AutoModelForCausalLM')
    @patch('core.models.local_llm.AutoTokenizer')
    def test_count_tokens(self, mock_tokenizer_class, mock_model_class, mock_tokenizer):
        """Test token counting."""
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = Mock()
        
        llm = LocalLLM('test-model', {}, device='cpu')
        llm.load()
        
        count = llm.count_tokens("test text")
        assert count == 5


class TestAPILLM:
    """Test API LLM wrapper."""
    
    def test_initialization(self):
        """Test API LLM initialization."""
        llm = APILLM(
            model_name='gpt-4',
            config={},
            api_key='test-key',
            provider=APIProvider.OPENAI
        )
        
        assert llm.llm_type == LLMType.OPENAI
        assert llm.api_key == 'test-key'
        assert llm.api_endpoint is not None
    
    @patch('core.models.api_llm.requests.post')
    def test_openai_generate(self, mock_post):
        """Test OpenAI generation."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {'content': 'Generated text'},
                'finish_reason': 'stop'
            }],
            'usage': {'total_tokens': 50}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        llm = APILLM(
            model_name='gpt-4',
            config={},
            api_key='test-key',
            provider=APIProvider.OPENAI
        )
        
        response = llm.generate("Test prompt")
        
        assert response.text == 'Generated text'
        assert response.finish_reason == 'stop'
        assert response.tokens_used == 50
    
    def test_count_tokens_estimate(self):
        """Test token count estimation."""
        llm = APILLM(
            model_name='gpt-4',
            config={},
            api_key='test-key',
            provider=APIProvider.OPENAI
        )
        
        # Should estimate ~5 tokens (20 chars / 4)
        count = llm.count_tokens("This is a test text.")
        assert count > 0


class TestLLMFactory:
    """Test LLM factory."""
    
    def test_create_local_llm(self):
        """Test creating local LLM via factory."""
        with patch('core.models.local_llm.AutoModelForCausalLM'), \
             patch('core.models.local_llm.AutoTokenizer'):
            
            llm = LLMFactory.create_llm(
                'local',
                'test-model',
                device='cpu',
                auto_load=False
            )
            
            assert isinstance(llm, LocalLLM)
            assert llm.device == 'cpu'
    
    def test_create_openai_llm(self):
        """Test creating OpenAI LLM via factory."""
        llm = LLMFactory.create_llm(
            'openai',
            'gpt-4',
            api_key='test-key'
        )
        
        assert isinstance(llm, APILLM)
        assert llm.provider == APIProvider.OPENAI
    
    def test_invalid_llm_type(self):
        """Test invalid LLM type raises error."""
        with pytest.raises(ValueError):
            LLMFactory.create_llm('invalid_type', 'model-name')


# Integration test
class TestLLMIntegration:
    """Integration tests."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    @pytest.mark.slow
    def test_local_llm_full_pipeline(self):
        """Test complete local LLM pipeline (requires GPU)."""
        # This would need a small test model
        # Skip in CI/CD unless test models are available
        pytest.skip("Integration test - requires test model")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--log-cli-level=INFO'])