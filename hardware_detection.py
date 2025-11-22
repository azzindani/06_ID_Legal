"""
Hardware Detection Module

Auto-detects hardware capabilities and recommends optimal configuration
for embedding, reranker, and LLM models.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """Hardware configuration recommendation"""
    embedding_device: str
    reranker_device: str
    llm_device: str
    llm_quantization: str  # 'none', '4bit', '8bit'
    recommended_model: str
    vram_available: float  # GB
    ram_available: float   # GB


def get_gpu_info() -> Tuple[bool, float, str]:
    """
    Detect GPU and available VRAM.

    Returns:
        Tuple of (has_gpu, vram_gb, gpu_name)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            return True, vram_gb, gpu_name
    except ImportError:
        pass
    except Exception:
        pass

    return False, 0.0, "No GPU"


def get_ram_info() -> float:
    """Get available system RAM in GB"""
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
        return ram_bytes / (1024 ** 3)
    except ImportError:
        # Fallback for Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except:
            pass
    return 16.0  # Default assumption


def detect_hardware() -> HardwareConfig:
    """
    Auto-detect hardware and return optimal configuration.

    Decision logic:
    - VRAM < 4GB: CPU only, use API provider
    - VRAM 4-8GB: CPU for embedding/reranker, GPU for LLM (4-bit)
    - VRAM 8-12GB: CPU for embedding/reranker, GPU for LLM (8-bit)
    - VRAM 12-16GB: GPU for all, LLM 8-bit
    - VRAM > 16GB: GPU for all, LLM full precision
    """
    has_gpu, vram_gb, gpu_name = get_gpu_info()
    ram_gb = get_ram_info()

    if not has_gpu or vram_gb < 4:
        # No usable GPU - CPU only or API
        return HardwareConfig(
            embedding_device='cpu',
            reranker_device='cpu',
            llm_device='cpu',
            llm_quantization='4bit',  # Still use quantization for RAM
            recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
            vram_available=vram_gb,
            ram_available=ram_gb
        )

    elif vram_gb < 8:
        # Limited VRAM - embedding/reranker on CPU
        return HardwareConfig(
            embedding_device='cpu',
            reranker_device='cpu',
            llm_device='cuda',
            llm_quantization='4bit',
            recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
            vram_available=vram_gb,
            ram_available=ram_gb
        )

    elif vram_gb < 12:
        # Medium VRAM - 8B model with 4-bit
        return HardwareConfig(
            embedding_device='cpu',
            reranker_device='cpu',
            llm_device='cuda',
            llm_quantization='4bit',
            recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
            vram_available=vram_gb,
            ram_available=ram_gb
        )

    elif vram_gb < 16:
        # Good VRAM - 8B model with 8-bit
        return HardwareConfig(
            embedding_device='cuda',
            reranker_device='cpu',
            llm_device='cuda',
            llm_quantization='8bit',
            recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
            vram_available=vram_gb,
            ram_available=ram_gb
        )

    else:
        # High VRAM - full precision
        return HardwareConfig(
            embedding_device='cuda',
            reranker_device='cuda',
            llm_device='cuda',
            llm_quantization='none',
            recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
            vram_available=vram_gb,
            ram_available=ram_gb
        )


def apply_hardware_config(config: Optional[HardwareConfig] = None) -> Dict[str, Any]:
    """
    Apply hardware configuration to environment variables.

    Args:
        config: Hardware config to apply. If None, auto-detects.

    Returns:
        Dictionary of applied settings
    """
    if config is None:
        config = detect_hardware()

    settings = {
        'EMBEDDING_DEVICE': config.embedding_device,
        'RERANKER_DEVICE': config.reranker_device,
        'LLM_DEVICE': config.llm_device,
        'LLM_LOAD_IN_4BIT': str(config.llm_quantization == '4bit').lower(),
        'LLM_LOAD_IN_8BIT': str(config.llm_quantization == '8bit').lower(),
        'LLM_MODEL': config.recommended_model,
    }

    # Only set if not already set by user
    for key, value in settings.items():
        if not os.getenv(key):
            os.environ[key] = value

    return settings


def print_hardware_info():
    """Print detected hardware information"""
    config = detect_hardware()

    print("=" * 60)
    print("HARDWARE DETECTION")
    print("=" * 60)
    print(f"RAM Available: {config.ram_available:.1f} GB")
    print(f"VRAM Available: {config.vram_available:.1f} GB")
    print()
    print("Recommended Configuration:")
    print(f"  Embedding Device: {config.embedding_device}")
    print(f"  Reranker Device: {config.reranker_device}")
    print(f"  LLM Device: {config.llm_device}")
    print(f"  LLM Quantization: {config.llm_quantization}")
    print(f"  Recommended Model: {config.recommended_model}")
    print("=" * 60)

    return config


if __name__ == "__main__":
    config = print_hardware_info()

    print("\nApplying configuration...")
    settings = apply_hardware_config(config)

    print("\nEnvironment variables set:")
    for key, value in settings.items():
        print(f"  {key}={value}")
