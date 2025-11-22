"""
Hardware Detection Module

Auto-detects hardware capabilities and recommends optimal configuration
for embedding, reranker, and LLM models. Supports multi-GPU distribution.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    index: int
    name: str
    vram_gb: float
    compute_capability: Tuple[int, int] = (0, 0)


@dataclass
class HardwareConfig:
    """Hardware configuration recommendation"""
    embedding_device: str
    reranker_device: str
    llm_device: str
    llm_quantization: str  # 'none', '4bit', '8bit'
    recommended_model: str
    vram_available: float  # Total VRAM GB
    ram_available: float   # GB
    gpu_count: int = 1
    gpu_info: List[GPUInfo] = field(default_factory=list)
    device_map: Dict[str, int] = field(default_factory=dict)  # component -> gpu_index


def get_all_gpu_info() -> List[GPUInfo]:
    """
    Detect all GPUs and their VRAM.

    Returns:
        List of GPUInfo for each detected GPU
    """
    gpus = []

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()

            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                compute_cap = (props.major, props.minor)

                gpus.append(GPUInfo(
                    index=i,
                    name=name,
                    vram_gb=vram_gb,
                    compute_capability=compute_cap
                ))
    except ImportError:
        pass
    except Exception:
        pass

    return gpus


def get_gpu_info() -> Tuple[bool, float, str]:
    """
    Detect GPU and available VRAM (legacy compatibility).

    Returns:
        Tuple of (has_gpu, total_vram_gb, gpu_name)
    """
    gpus = get_all_gpu_info()

    if not gpus:
        return False, 0.0, "No GPU"

    total_vram = sum(g.vram_gb for g in gpus)
    names = ", ".join(g.name for g in gpus)

    return True, total_vram, names


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
    Supports multi-GPU distribution.

    Decision logic based on total VRAM:
    - VRAM < 4GB: CPU only
    - VRAM 4-8GB: CPU for embedding/reranker, GPU for LLM (4-bit)
    - VRAM 8-12GB: Distribute across GPUs, LLM 4-bit
    - VRAM 12-16GB: Distribute across GPUs, LLM 8-bit
    - VRAM 16-24GB: All on GPU, LLM 8-bit
    - VRAM > 24GB: All on GPU, LLM full precision
    """
    gpus = get_all_gpu_info()
    ram_gb = get_ram_info()

    gpu_count = len(gpus)
    total_vram = sum(g.vram_gb for g in gpus) if gpus else 0.0

    # Default device map
    device_map = {}

    if gpu_count == 0 or total_vram < 4:
        # No usable GPU - CPU only
        return HardwareConfig(
            embedding_device='cpu',
            reranker_device='cpu',
            llm_device='cpu',
            llm_quantization='4bit',
            recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
            vram_available=total_vram,
            ram_available=ram_gb,
            gpu_count=gpu_count,
            gpu_info=gpus,
            device_map={}
        )

    elif gpu_count == 1:
        # Single GPU - decide based on VRAM
        gpu = gpus[0]

        if gpu.vram_gb < 8:
            return HardwareConfig(
                embedding_device='cpu',
                reranker_device='cpu',
                llm_device='cuda:0',
                llm_quantization='4bit',
                recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
                vram_available=total_vram,
                ram_available=ram_gb,
                gpu_count=1,
                gpu_info=gpus,
                device_map={'llm': 0}
            )
        elif gpu.vram_gb < 12:
            return HardwareConfig(
                embedding_device='cpu',
                reranker_device='cpu',
                llm_device='cuda:0',
                llm_quantization='4bit',
                recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
                vram_available=total_vram,
                ram_available=ram_gb,
                gpu_count=1,
                gpu_info=gpus,
                device_map={'llm': 0}
            )
        elif gpu.vram_gb < 16:
            return HardwareConfig(
                embedding_device='cuda:0',
                reranker_device='cpu',
                llm_device='cuda:0',
                llm_quantization='8bit',
                recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
                vram_available=total_vram,
                ram_available=ram_gb,
                gpu_count=1,
                gpu_info=gpus,
                device_map={'embedding': 0, 'llm': 0}
            )
        else:
            return HardwareConfig(
                embedding_device='cuda:0',
                reranker_device='cuda:0',
                llm_device='cuda:0',
                llm_quantization='none',
                recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
                vram_available=total_vram,
                ram_available=ram_gb,
                gpu_count=1,
                gpu_info=gpus,
                device_map={'embedding': 0, 'reranker': 0, 'llm': 0}
            )

    else:
        # Multi-GPU - distribute workloads
        # Sort GPUs by VRAM (largest first for LLM)
        sorted_gpus = sorted(gpus, key=lambda g: g.vram_gb, reverse=True)

        # Assign LLM to largest GPU
        llm_gpu = sorted_gpus[0].index

        # Assign embedding to second GPU if available
        embedding_gpu = sorted_gpus[1].index if len(sorted_gpus) > 1 else llm_gpu

        # Assign reranker to third GPU or share with embedding
        if len(sorted_gpus) > 2:
            reranker_gpu = sorted_gpus[2].index
        else:
            reranker_gpu = embedding_gpu

        device_map = {
            'embedding': embedding_gpu,
            'reranker': reranker_gpu,
            'llm': llm_gpu
        }

        # Determine quantization based on LLM GPU's VRAM
        llm_vram = sorted_gpus[0].vram_gb

        if total_vram >= 48:  # e.g., 2x24GB or 4x16GB
            quantization = 'none'
        elif total_vram >= 32:  # e.g., 2x16GB
            quantization = 'none'
        elif llm_vram >= 16:
            quantization = '8bit'
        elif llm_vram >= 12:
            quantization = '8bit'
        else:
            quantization = '4bit'

        return HardwareConfig(
            embedding_device=f'cuda:{embedding_gpu}',
            reranker_device=f'cuda:{reranker_gpu}',
            llm_device=f'cuda:{llm_gpu}',
            llm_quantization=quantization,
            recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
            vram_available=total_vram,
            ram_available=ram_gb,
            gpu_count=gpu_count,
            gpu_info=gpus,
            device_map=device_map
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
        'GPU_COUNT': str(config.gpu_count),
    }

    # Only set if not already set by user
    for key, value in settings.items():
        if not os.getenv(key):
            os.environ[key] = value

    return settings


def print_hardware_info():
    """Print detected hardware information"""
    gpus = get_all_gpu_info()
    config = detect_hardware()

    print("=" * 60)
    print("HARDWARE DETECTION")
    print("=" * 60)

    print(f"\nSystem RAM: {config.ram_available:.1f} GB")
    print(f"GPU Count: {config.gpu_count}")
    print(f"Total VRAM: {config.vram_available:.1f} GB")

    if gpus:
        print("\nGPU Details:")
        for gpu in gpus:
            print(f"  [{gpu.index}] {gpu.name}")
            print(f"      VRAM: {gpu.vram_gb:.1f} GB")
            print(f"      Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")

    print("\nRecommended Configuration:")
    print(f"  Embedding Device: {config.embedding_device}")
    print(f"  Reranker Device: {config.reranker_device}")
    print(f"  LLM Device: {config.llm_device}")
    print(f"  LLM Quantization: {config.llm_quantization}")
    print(f"  Recommended Model: {config.recommended_model}")

    if config.device_map:
        print("\nDevice Map:")
        for component, gpu_idx in config.device_map.items():
            gpu_name = next((g.name for g in gpus if g.index == gpu_idx), "Unknown")
            print(f"  {component}: GPU {gpu_idx} ({gpu_name})")

    print("=" * 60)

    return config


if __name__ == "__main__":
    config = print_hardware_info()

    print("\nApplying configuration...")
    settings = apply_hardware_config(config)

    print("\nEnvironment variables set:")
    for key, value in settings.items():
        print(f"  {key}={value}")
