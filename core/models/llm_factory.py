# core/models/llm_factory.py
"""
Factory pattern for creating LLM instances.
Simplifies LLM creation and configuration.
"""
from typing import Dict, Any, Optional
from pathlib import Path

from core.models.base_llm import BaseLLM
from core.models.local_llm import LocalLLM
from core.models.api_llm import APILLM, APIProvider
from utils.logging_config import get_logger

logger = get_logger(__name__)

class LLMFactory:
    """
    Factory for creating LLM instances.
    Handles configuration and initialization.
    """
    
    @staticmethod
    def create_llm(
        llm_type: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create LLM instance based on type.
        
        Args:
            llm_type: Type of LLM ('local', 'openai', 'anthropic', 'huggingface_api')
            model_name: Model identifier
            config: Optional configuration dictionary
            **kwargs: Additional arguments for specific LLM type
            
        Returns:
            Initialized LLM instance
            
        Raises:
            ValueError: If llm_type is invalid
        
        Examples:
            >>> # Local model
            >>> llm = LLMFactory.create_llm(
            ...     'local',
            ...     'Azzindani/Deepseek_ID_Legal_Preview',
            ...     device='cuda'
            ... )
            
            >>> # OpenAI API
            >>> llm = LLMFactory.create_llm(
            ...     'openai',
            ...     'gpt-4',
            ...     api_key='sk-...'
            ... )
        """
        config = config or {}
        llm_type = llm_type.lower()
        
        logger.info(f"Creating LLM: type={llm_type}, model={model_name}")
        
        try:
            if llm_type == 'local':
                return LLMFactory._create_local_llm(model_name, config, **kwargs)
            
            elif llm_type == 'openai':
                return LLMFactory._create_openai_llm(model_name, config, **kwargs)
            
            elif llm_type == 'anthropic':
                return LLMFactory._create_anthropic_llm(model_name, config, **kwargs)
            
            elif llm_type == 'huggingface_api':
                return LLMFactory._create_huggingface_api_llm(model_name, config, **kwargs)
            
            else:
                raise ValueError(
                    f"Unknown LLM type: {llm_type}. "
                    f"Supported types: local, openai, anthropic, huggingface_api"
                )
        
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}", exc_info=True)
            raise
    
    @staticmethod
    def _create_local_llm(
        model_name: str,
        config: Dict[str, Any],
        **kwargs
    ) -> LocalLLM:
        """Create local HuggingFace LLM."""
        device = kwargs.get('device', 'cuda')
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        auto_load = kwargs.get('auto_load', True)
        
        llm = LocalLLM(
            model_name=model_name,
            config=config,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        
        if auto_load:
            llm.load()
        
        return llm
    
    @staticmethod
    def _create_openai_llm(
        model_name: str,
        config: Dict[str, Any],
        **kwargs
    ) -> APILLM:
        """Create OpenAI API LLM."""
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("api_key is required for OpenAI LLM")
        
        api_endpoint = kwargs.get('api_endpoint')
        
        return APILLM(
            model_name=model_name,
            config=config,
            api_key=api_key,
            api_endpoint=api_endpoint,
            provider=APIProvider.OPENAI
        )
    
    @staticmethod
    def _create_anthropic_llm(
        model_name: str,
        config: Dict[str, Any],
        **kwargs
    ) -> APILLM:
        """Create Anthropic API LLM."""
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("api_key is required for Anthropic LLM")
        
        api_endpoint = kwargs.get('api_endpoint')
        
        return APILLM(
            model_name=model_name,
            config=config,
            api_key=api_key,
            api_endpoint=api_endpoint,
            provider=APIProvider.ANTHROPIC
        )
    
    @staticmethod
    def _create_huggingface_api_llm(
        model_name: str,
        config: Dict[str, Any],
        **kwargs
    ) -> APILLM:
        """Create HuggingFace Inference API LLM."""
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("api_key is required for HuggingFace API")
        
        api_endpoint = kwargs.get('api_endpoint')
        
        return APILLM(
            model_name=model_name,
            config=config,
            api_key=api_key,
            api_endpoint=api_endpoint,
            provider=APIProvider.HUGGINGFACE
        )
    
    @staticmethod
    def create_from_config_file(config_path: Path) -> BaseLLM:
        """
        Create LLM from configuration file.
        
        Args:
            config_path: Path to YAML or JSON config file
            
        Returns:
            Configured LLM instance
        """
        import json
        import yaml
        
        logger.info(f"Loading LLM config from: {config_path}")
        
        # Load config file
        if config_path.suffix == '.json':
            with open(config_path) as f:
                config = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Extract parameters
        llm_type = config.pop('type')
        model_name = config.pop('model_name')
        
        return LLMFactory.create_llm(llm_type, model_name, config=config, **config)