"""
Model Configuration Module for Indonesian Legal RAG System

This module provides centralized configuration management for all models,
datasets, and system parameters. Supports both local and API inference modes.

Author: Azzindani Team
Version: 2.0
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

# Import logging utility (create this first - see utils/logging_config.py)
from utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# CORE MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """
    Comprehensive configuration for all models in the RAG system.
    
    Supports:
    - Local model inference (HuggingFace Transformers)
    - API-based inference (OpenAI, Anthropic, custom endpoints)
    - Environment variable overrides
    - Auto device detection (CUDA/CPU)
    
    Attributes:
        embedding_model_name: HuggingFace model name for embeddings
        reranker_model_name: HuggingFace model name for reranking
        llm_model_name: HuggingFace model name for generation
        dataset_name: HuggingFace dataset name for legal documents
        
        llm_use_api: Toggle for API-based LLM inference
        llm_api_endpoint: API endpoint URL (required if llm_use_api=True)
        llm_api_key: API key (reads from env var LLM_API_KEY)
        llm_api_model: Model identifier for API calls
        
        hf_token: HuggingFace token (reads from env var HF_TOKEN)
        device: Compute device ('cuda', 'cpu', or 'auto')
        
        max_length: Max token length for models
        batch_size: Batch size for inference
        embedding_dim: Embedding vector dimension
        
        cache_dir: Directory for cached models
        log_level: Logging verbosity level
    """
    
    # =========================================================================
    # MODEL IDENTIFIERS
    # =========================================================================
    
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    llm_model_name: str = "Azzindani/Deepseek_ID_Legal_Preview"
    dataset_name: str = "Azzindani/ID_REG_KG_2510"
    
    # =========================================================================
    # API INFERENCE CONFIGURATION (Future-Proof)
    # =========================================================================
    
    llm_use_api: bool = False
    llm_api_endpoint: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_api_model: Optional[str] = None
    llm_api_timeout: int = 120
    llm_api_max_retries: int = 3
    
    # =========================================================================
    # AUTHENTICATION & SECURITY
    # =========================================================================
    
    hf_token: Optional[str] = field(default=None)
    
    # =========================================================================
    # COMPUTE CONFIGURATION
    # =========================================================================
    
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    use_flash_attention: bool = True
    torch_dtype: str = "float16"  # 'float32', 'float16', 'bfloat16'
    
    # =========================================================================
    # MODEL PARAMETERS
    # =========================================================================
    
    max_length: int = 32768
    embedding_max_length: int = 512
    reranker_max_length: int = 8192
    
    batch_size: int = 2
    embedding_batch_size: int = 8
    reranker_batch_size: int = 2
    
    embedding_dim: int = 768  # Will be auto-detected
    
    # =========================================================================
    # GENERATION PARAMETERS
    # =========================================================================
    
    temperature: float = 0.7
    max_new_tokens: int = 2048
    top_p: float = 1.0
    top_k: int = 20
    min_p: float = 0.1
    
    # =========================================================================
    # SYSTEM CONFIGURATION
    # =========================================================================
    
    cache_dir: Optional[str] = None
    log_level: str = "INFO"
    enable_progress_bars: bool = True
    
    # =========================================================================
    # DATASET CONFIGURATION
    # =========================================================================
    
    dataset_split: str = "train"
    dataset_streaming: bool = False
    dataset_max_records: Optional[int] = None
    
    # =========================================================================
    # POST-INITIALIZATION VALIDATION
    # =========================================================================
    
    def __post_init__(self):
        """
        Post-initialization validation and setup.
        
        This method:
        1. Loads environment variables
        2. Validates configuration consistency
        3. Auto-detects compute device
        4. Logs configuration summary
        
        Raises:
            ValueError: If configuration is invalid
            EnvironmentError: If required environment variables are missing
        """
        logger.info("Initializing ModelConfig...")
        
        # Step 1: Load environment variables
        self._load_environment_variables()
        
        # Step 2: Validate API configuration
        if self.llm_use_api:
            self._validate_api_config()
        
        # Step 3: Auto-detect device
        self._detect_device()
        
        # Step 4: Validate paths
        self._validate_paths()
        
        # Step 5: Log configuration
        self._log_configuration_summary()
        
        logger.info("✅ ModelConfig initialized successfully")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        logger.debug("Loading environment variables...")
        
        # HuggingFace Token
        env_hf_token = os.getenv("HF_TOKEN")
        if env_hf_token:
            self.hf_token = env_hf_token
            logger.debug("✅ HF_TOKEN loaded from environment")
        elif self.hf_token is None:
            logger.warning("⚠️ HF_TOKEN not set - some models may be inaccessible")
        
        # API Key
        env_api_key = os.getenv("LLM_API_KEY")
        if env_api_key:
            self.llm_api_key = env_api_key
            logger.debug("✅ LLM_API_KEY loaded from environment")
        
        # Cache Directory
        env_cache_dir = os.getenv("TRANSFORMERS_CACHE")
        if env_cache_dir:
            self.cache_dir = env_cache_dir
            logger.debug(f"✅ Using cache directory: {self.cache_dir}")
        
        # Log Level
        env_log_level = os.getenv("LOG_LEVEL")
        if env_log_level:
            self.log_level = env_log_level.upper()
            logger.setLevel(getattr(logging, self.log_level))
            logger.debug(f"✅ Log level set to: {self.log_level}")
    
    def _validate_api_config(self):
        """Validate API inference configuration."""
        logger.debug("Validating API configuration...")
        
        if not self.llm_api_endpoint:
            raise ValueError(
                "❌ llm_api_endpoint is required when llm_use_api=True. "
                "Please set it in config or via environment variable."
            )
        
        if not self.llm_api_key:
            raise EnvironmentError(
                "❌ LLM_API_KEY environment variable is required for API inference. "
                "Set it with: export LLM_API_KEY=your_key_here"
            )
        
        if not self.llm_api_model:
            logger.warning(
                "⚠️ llm_api_model not set - using default from endpoint. "
                "Consider specifying explicitly for better control."
            )
        
        # Validate endpoint format
        if not self.llm_api_endpoint.startswith(('http://', 'https://')):
            raise ValueError(
                f"❌ Invalid API endpoint format: {self.llm_api_endpoint}. "
                "Must start with http:// or https://"
            )
        
        logger.info(f"✅ API configuration validated: {self.llm_api_endpoint}")
    
    def _detect_device(self):
        """Auto-detect optimal compute device."""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"✅ CUDA detected: {gpu_name} ({gpu_memory:.1f}GB)")
                else:
                    self.device = "cpu"
                    logger.warning("⚠️ CUDA not available - using CPU (will be slow)")
            except ImportError:
                self.device = "cpu"
                logger.error("❌ PyTorch not installed - defaulting to CPU")
        else:
            logger.info(f"📌 Using manually specified device: {self.device}")
    
    def _validate_paths(self):
        """Validate and create necessary directories."""
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
            if not cache_path.exists():
                logger.info(f"Creating cache directory: {cache_path}")
                cache_path.mkdir(parents=True, exist_ok=True)
            
            if not os.access(cache_path, os.W_OK):
                raise PermissionError(
                    f"❌ No write permission for cache directory: {cache_path}"
                )
            
            logger.debug(f"✅ Cache directory validated: {cache_path}")
    
    def _log_configuration_summary(self):
        """Log comprehensive configuration summary."""
        logger.info("=" * 80)
        logger.info("📋 MODEL CONFIGURATION SUMMARY")
        logger.info("=" * 80)
        
        logger.info("\n🤖 MODELS:")
        logger.info(f"   Embedding:  {self.embedding_model_name}")
        logger.info(f"   Reranker:   {self.reranker_model_name}")
        logger.info(f"   LLM:        {self.llm_model_name}")
        logger.info(f"   Dataset:    {self.dataset_name}")
        
        logger.info("\n🔧 INFERENCE MODE:")
        if self.llm_use_api:
            logger.info(f"   API Endpoint:  {self.llm_api_endpoint}")
            logger.info(f"   API Model:     {self.llm_api_model or 'Default'}")
            logger.info(f"   API Timeout:   {self.llm_api_timeout}s")
        else:
            logger.info(f"   Mode:          Local (Transformers)")
            logger.info(f"   Device:        {self.device}")
            logger.info(f"   Dtype:         {self.torch_dtype}")
            logger.info(f"   Flash Attn:    {self.use_flash_attention}")
        
        logger.info("\n📊 PARAMETERS:")
        logger.info(f"   Max Length:    {self.max_length}")
        logger.info(f"   Batch Size:    {self.batch_size}")
        logger.info(f"   Temperature:   {self.temperature}")
        logger.info(f"   Max Tokens:    {self.max_new_tokens}")
        
        logger.info("\n💾 SYSTEM:")
        logger.info(f"   Cache Dir:     {self.cache_dir or 'Default'}")
        logger.info(f"   Log Level:     {self.log_level}")
        
        logger.info("=" * 80)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            Dict containing all configuration parameters
        """
        return {
            'embedding_model_name': self.embedding_model_name,
            'reranker_model_name': self.reranker_model_name,
            'llm_model_name': self.llm_model_name,
            'dataset_name': self.dataset_name,
            'llm_use_api': self.llm_use_api,
            'device': self.device,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'cache_dir': self.cache_dir,
        }
    
    def get_model_kwargs(self, model_type: str = "llm") -> Dict[str, Any]:
        """
        Get model-specific initialization kwargs.
        
        Args:
            model_type: One of 'embedding', 'reranker', 'llm'
        
        Returns:
            Dictionary of kwargs for model initialization
        """
        base_kwargs = {
            'device_map': "auto" if self.device == "cuda" else None,
            'torch_dtype': getattr(__import__('torch'), self.torch_dtype),
            'token': self.hf_token,
            'cache_dir': self.cache_dir,
        }
        
        if self.use_flash_attention and self.device == "cuda":
            base_kwargs['attn_implementation'] = "flash_attention_2"
        
        if model_type == "llm":
            base_kwargs['max_length'] = self.max_length
        elif model_type == "embedding":
            base_kwargs['max_length'] = self.embedding_max_length
        elif model_type == "reranker":
            base_kwargs['max_length'] = self.reranker_max_length
        
        return base_kwargs
    
    def validate(self) -> bool:
        """
        Comprehensive configuration validation.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        issues = []
        
        # Check model names
        if not self.embedding_model_name:
            issues.append("Embedding model name is required")
        
        if not self.reranker_model_name:
            issues.append("Reranker model name is required")
        
        if not self.llm_model_name and not self.llm_use_api:
            issues.append("LLM model name is required for local inference")
        
        # Check API configuration
        if self.llm_use_api:
            if not self.llm_api_endpoint:
                issues.append("API endpoint is required when llm_use_api=True")
            if not self.llm_api_key:
                issues.append("API key is required when llm_use_api=True")
        
        # Check parameters
        if self.temperature < 0 or self.temperature > 2:
            issues.append("Temperature must be between 0 and 2")
        
        if self.max_new_tokens < 1:
            issues.append("max_new_tokens must be positive")
        
        if self.batch_size < 1:
            issues.append("batch_size must be positive")
        
        if issues:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join([f"- {i}" for i in issues]))
        
        return True


# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for multi-phase search strategy."""
    
    final_top_k: int = 3
    max_rounds: int = 5
    initial_quality: float = 0.95
    quality_degradation: float = 0.1
    min_quality: float = 0.5
    
    # Research team
    research_team_size: int = 4
    enable_cross_validation: bool = True
    enable_devil_advocate: bool = True
    consensus_threshold: float = 0.6
    
    # Search phases
    search_phases: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'initial_scan': {
            'candidates': 400,
            'semantic_threshold': 0.20,
            'keyword_threshold': 0.06,
            'enabled': True
        },
        'focused_review': {
            'candidates': 150,
            'semantic_threshold': 0.35,
            'keyword_threshold': 0.12,
            'enabled': True
        },
        'deep_analysis': {
            'candidates': 60,
            'semantic_threshold': 0.45,
            'keyword_threshold': 0.18,
            'enabled': True
        },
        'verification': {
            'candidates': 30,
            'semantic_threshold': 0.55,
            'keyword_threshold': 0.22,
            'enabled': True
        },
        'expert_review': {
            'candidates': 45,
            'semantic_threshold': 0.50,
            'keyword_threshold': 0.20,
            'enabled': False
        }
    })


# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

# Singleton instance - used throughout the application
MODEL_CONFIG = ModelConfig()
SEARCH_CONFIG = SearchConfig()

logger.info("✅ Global configuration instances created")


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def reload_config():
    """Reload configuration from environment variables."""
    global MODEL_CONFIG, SEARCH_CONFIG
    MODEL_CONFIG = ModelConfig()
    SEARCH_CONFIG = SearchConfig()
    logger.info("✅ Configuration reloaded")


def get_config_summary() -> str:
    """Get human-readable configuration summary."""
    return f"""
# Configuration Summary

## Models
- Embedding: {MODEL_CONFIG.embedding_model_name}
- Reranker: {MODEL_CONFIG.reranker_model_name}
- LLM: {MODEL_CONFIG.llm_model_name}
- Dataset: {MODEL_CONFIG.dataset_name}

## Inference
- Mode: {'API' if MODEL_CONFIG.llm_use_api else 'Local'}
- Device: {MODEL_CONFIG.device}

## Search
- Top K: {SEARCH_CONFIG.final_top_k}
- Team Size: {SEARCH_CONFIG.research_team_size}
- Consensus: {SEARCH_CONFIG.consensus_threshold:.0%}
"""


if __name__ == "__main__":
    # Test configuration
    print("Testing ModelConfig...")
    print(get_config_summary())
    
    # Test validation
    MODEL_CONFIG.validate()
    print("\n✅ Configuration validated successfully!")