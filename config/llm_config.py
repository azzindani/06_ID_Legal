# config/llm_config.py
"""
Configuration presets for common LLM models.
"""
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class LLMPreset:
    """Preset configuration for specific models."""
    type: str
    model_name: str
    device: str = 'cuda'
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

# Common model presets
LLM_PRESETS = {
    # Local models
    'deepseek_id_legal': LLMPreset(
        type='local',
        model_name='Azzindani/Deepseek_ID_Legal_Preview',
        device='cuda',
        load_in_8bit=False
    ),
    
    'llama3_8b': LLMPreset(
        type='local',
        model_name='meta-llama/Meta-Llama-3-8B-Instruct',
        device='cuda',
        load_in_4bit=True  # 4-bit for efficiency
    ),
    
    'mistral_7b': LLMPreset(
        type='local',
        model_name='mistralai/Mistral-7B-Instruct-v0.2',
        device='cuda',
        load_in_4bit=True
    ),
    
    # API models
    'gpt4': LLMPreset(
        type='openai',
        model_name='gpt-4',
        api_key=os.getenv('OPENAI_API_KEY')
    ),
    
    'gpt35_turbo': LLMPreset(
        type='openai',
        model_name='gpt-3.5-turbo',
        api_key=os.getenv('OPENAI_API_KEY')
    ),
    
    'claude3_opus': LLMPreset(
        type='anthropic',
        model_name='claude-3-opus-20240229',
        api_key=os.getenv('ANTHROPIC_API_KEY')
    ),
    
    'claude3_sonnet': LLMPreset(
        type='anthropic',
        model_name='claude-3-sonnet-20240229',
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )
}

def get_preset(preset_name: str) -> LLMPreset:
    """
    Get preset configuration by name.
    
    Args:
        preset_name: Name of preset
        
    Returns:
        LLMPreset configuration
        
    Raises:
        ValueError: If preset not found
    """
    if preset_name not in LLM_PRESETS:
        available = ', '.join(LLM_PRESETS.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {available}"
        )
    
    return LLM_PRESETS[preset_name]