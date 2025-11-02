# config/model_config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration with environment variable support."""
    
    # HuggingFace
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    
    # Dataset
    dataset_name: str = "Azzindani/ID_REG_KG_2510"
    
    # Embedding Model
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_max_length: int = 512
    embedding_batch_size: int = 32
    
    # Reranker Model
    reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_max_length: int = 32768
    
    # LLM Model
    llm_model_name: str = "Azzindani/Deepseek_ID_Legal_Preview"
    llm_max_length: int = 32768
    llm_use_api: bool = False  # Future: API inference toggle
    llm_api_endpoint: Optional[str] = os.getenv("LLM_API_ENDPOINT")
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    
    # Device
    device: str = "cuda"  # Will be auto-detected
    use_flash_attention: bool = True
    torch_dtype: str = "float16"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.llm_use_api and not self.llm_api_endpoint:
            raise ValueError("LLM API mode enabled but no endpoint provided")

# Global config instance
MODEL_CONFIG = ModelConfig()