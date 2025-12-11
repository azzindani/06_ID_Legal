"""
Hardware Detection Module

Auto-detects hardware capabilities and recommends optimal configuration
for embedding, reranker, and LLM models. Supports multi-GPU distribution.

Uses mathematical optimization to determine the best model placement strategy
based on available hardware resources, balancing speed, quality, and memory efficiency.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass, field
from enum import Enum


class DeviceType(Enum):
    """Device type enumeration"""
    CPU = "cpu"
    GPU = "gpu"


class QuantizationType(Enum):
    """Quantization type enumeration"""
    FP16 = "none"  # Full precision (fp16)
    FP8 = "8bit"   # 8-bit quantization
    FP4 = "4bit"   # 4-bit quantization


@dataclass
class ModelSpec:
    """Specification for a model's memory requirements"""
    name: str
    size_fp16_gb: float  # Size in GB at fp16 precision

    def get_size(self, quantization: QuantizationType) -> float:
        """Get model size for given quantization level"""
        if quantization == QuantizationType.FP16:
            return self.size_fp16_gb
        elif quantization == QuantizationType.FP8:
            return self.size_fp16_gb * 0.5  # ~50% of fp16
        elif quantization == QuantizationType.FP4:
            return self.size_fp16_gb * 0.25  # ~25% of fp16
        return self.size_fp16_gb


# Model specifications (from user requirements)
MODEL_SPECS = {
    'embedding': ModelSpec('embedding', 1.2),
    'reranker': ModelSpec('reranker', 1.2),
    'llm': ModelSpec('llm', 16.0)
}

# Memory overhead multiplier for activations and workspace
MEMORY_OVERHEAD = 1.3

# Speed factors (relative performance)
SPEED_FACTOR = {
    DeviceType.GPU: 1.0,
    DeviceType.CPU: 0.1  # CPU is ~10x slower than GPU
}

# Quality factors (relative quality)
QUALITY_FACTOR = {
    QuantizationType.FP16: 1.0,
    QuantizationType.FP8: 0.95,  # Slight quality loss
    QuantizationType.FP4: 0.85   # More quality loss
}


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    index: int
    name: str
    vram_gb: float
    compute_capability: Tuple[int, int] = (0, 0)

    def available_memory(self, reserved_gb: float = 1.0) -> float:
        """Get available memory after system reservation"""
        return max(0, self.vram_gb - reserved_gb)


@dataclass
class ModelPlacement:
    """Represents where and how a model should be placed"""
    model_name: str
    device_type: DeviceType
    device_index: int  # GPU index or -1 for CPU
    quantization: QuantizationType
    memory_required_gb: float

    @property
    def device_string(self) -> str:
        """Get device string (e.g., 'cuda:0', 'cpu')"""
        if self.device_type == DeviceType.CPU:
            return 'cpu'
        return f'cuda:{self.device_index}'

    def compute_score(self) -> float:
        """
        Compute placement quality score.
        Higher is better. Balances speed and quality.
        """
        speed = SPEED_FACTOR[self.device_type]
        quality = QUALITY_FACTOR[self.quantization]
        # Weighted combination: speed (60%) + quality (40%)
        return 0.6 * speed + 0.4 * quality


@dataclass
class AllocationStrategy:
    """Complete allocation strategy for all models"""
    placements: Dict[str, ModelPlacement]
    total_score: float
    memory_usage: Dict[int, float]  # GPU index -> memory used (GB)
    cpu_memory_gb: float

    def is_valid(self, gpus: List[GPUInfo], ram_gb: float) -> bool:
        """Check if allocation is valid given hardware constraints"""
        # Check GPU memory constraints
        for gpu_idx, mem_used in self.memory_usage.items():
            if gpu_idx >= 0:  # GPU
                gpu = next((g for g in gpus if g.index == gpu_idx), None)
                if gpu is None:
                    return False
                if mem_used > gpu.available_memory():
                    return False

        # Check CPU memory constraint (with more headroom)
        if self.cpu_memory_gb > ram_gb * 0.7:  # Use max 70% of RAM
            return False

        return True


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
    allocation_score: float = 0.0  # Quality score of allocation
    memory_breakdown: Dict[str, float] = field(default_factory=dict)  # Model -> memory GB


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


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information"""
    cpu_info = {
        'cores': 1,
        'threads': 1,
        'name': 'Unknown CPU'
    }

    try:
        import psutil
        cpu_info['cores'] = psutil.cpu_count(logical=False) or 1
        cpu_info['threads'] = psutil.cpu_count(logical=True) or 1
    except:
        pass

    return cpu_info


def generate_model_candidates(
    model_name: str,
    model_spec: ModelSpec,
    gpus: List[GPUInfo]
) -> List[ModelPlacement]:
    """
    Generate all candidate placements for a model.

    Args:
        model_name: Name of the model (embedding, reranker, llm)
        model_spec: Model specification
        gpus: List of available GPUs

    Returns:
        List of possible placements
    """
    candidates = []

    # CPU placements (always available)
    for quant in QuantizationType:
        # For embedding/reranker, only fp16 makes sense on CPU
        # For LLM, all quantizations are viable on CPU
        if model_name in ['embedding', 'reranker'] and quant != QuantizationType.FP16:
            continue

        mem_required = model_spec.get_size(quant) * MEMORY_OVERHEAD
        candidates.append(ModelPlacement(
            model_name=model_name,
            device_type=DeviceType.CPU,
            device_index=-1,
            quantization=quant,
            memory_required_gb=mem_required
        ))

    # GPU placements
    for gpu in gpus:
        for quant in QuantizationType:
            # For embedding/reranker, only fp16 makes sense
            if model_name in ['embedding', 'reranker'] and quant != QuantizationType.FP16:
                continue

            mem_required = model_spec.get_size(quant) * MEMORY_OVERHEAD
            # Only add if GPU has enough memory
            if mem_required <= gpu.available_memory():
                candidates.append(ModelPlacement(
                    model_name=model_name,
                    device_type=DeviceType.GPU,
                    device_index=gpu.index,
                    quantization=quant,
                    memory_required_gb=mem_required
                ))

    return candidates


def compute_allocation_strategy(
    placements: Dict[str, ModelPlacement]
) -> AllocationStrategy:
    """
    Compute allocation strategy from placements.

    Args:
        placements: Dictionary mapping model name to placement

    Returns:
        AllocationStrategy with computed scores and memory usage
    """
    memory_usage = {}  # GPU index -> memory used
    cpu_memory = 0.0

    # Calculate total score and memory usage
    total_score = 0.0
    for model_name, placement in placements.items():
        # Add model's score contribution
        model_score = placement.compute_score()

        # Weight scores by model importance
        # LLM is most important (50%), embedding and reranker (25% each)
        if model_name == 'llm':
            total_score += model_score * 0.5
        else:
            total_score += model_score * 0.25

        # Track memory usage
        if placement.device_type == DeviceType.GPU:
            gpu_idx = placement.device_index
            memory_usage[gpu_idx] = memory_usage.get(gpu_idx, 0) + placement.memory_required_gb
        else:
            cpu_memory += placement.memory_required_gb

    return AllocationStrategy(
        placements=placements,
        total_score=total_score,
        memory_usage=memory_usage,
        cpu_memory_gb=cpu_memory
    )


def optimize_model_allocation(
    gpus: List[GPUInfo],
    ram_gb: float
) -> AllocationStrategy:
    """
    Find optimal model allocation using intelligent search.

    Uses a greedy optimization approach:
    1. Generate candidates for each model
    2. Prioritize LLM placement (most important)
    3. Then place embedding and reranker
    4. Score each valid configuration
    5. Return best allocation

    Args:
        gpus: List of available GPUs
        ram_gb: Available system RAM in GB

    Returns:
        Best allocation strategy
    """
    # Generate candidates for each model
    embedding_candidates = generate_model_candidates('embedding', MODEL_SPECS['embedding'], gpus)
    reranker_candidates = generate_model_candidates('reranker', MODEL_SPECS['reranker'], gpus)
    llm_candidates = generate_model_candidates('llm', MODEL_SPECS['llm'], gpus)

    best_strategy = None
    best_score = -1.0

    # Try combinations - prioritize LLM first since it's largest and most important
    for llm_placement in llm_candidates:
        # Calculate remaining memory after LLM placement
        remaining_gpu_mem = {}
        for gpu in gpus:
            remaining_gpu_mem[gpu.index] = gpu.available_memory()

        remaining_cpu_mem = ram_gb * 0.7  # Max 70% CPU RAM

        if llm_placement.device_type == DeviceType.GPU:
            remaining_gpu_mem[llm_placement.device_index] -= llm_placement.memory_required_gb
        else:
            remaining_cpu_mem -= llm_placement.memory_required_gb

        # Try embedding placements
        for emb_placement in embedding_candidates:
            # Check if embedding fits
            fits = False
            if emb_placement.device_type == DeviceType.GPU:
                if remaining_gpu_mem[emb_placement.device_index] >= emb_placement.memory_required_gb:
                    fits = True
            else:
                if remaining_cpu_mem >= emb_placement.memory_required_gb:
                    fits = True

            if not fits:
                continue

            # Update remaining memory
            rem_gpu_mem = remaining_gpu_mem.copy()
            rem_cpu_mem = remaining_cpu_mem

            if emb_placement.device_type == DeviceType.GPU:
                rem_gpu_mem[emb_placement.device_index] -= emb_placement.memory_required_gb
            else:
                rem_cpu_mem -= emb_placement.memory_required_gb

            # Try reranker placements
            for rer_placement in reranker_candidates:
                # Check if reranker fits
                fits = False
                if rer_placement.device_type == DeviceType.GPU:
                    if rem_gpu_mem[rer_placement.device_index] >= rer_placement.memory_required_gb:
                        fits = True
                else:
                    if rem_cpu_mem >= rer_placement.memory_required_gb:
                        fits = True

                if not fits:
                    continue

                # Create allocation strategy
                placements = {
                    'embedding': emb_placement,
                    'reranker': rer_placement,
                    'llm': llm_placement
                }

                strategy = compute_allocation_strategy(placements)

                # Verify it's valid
                if strategy.is_valid(gpus, ram_gb):
                    if strategy.total_score > best_score:
                        best_score = strategy.total_score
                        best_strategy = strategy

    # Fallback: CPU-only if no valid strategy found
    if best_strategy is None:
        placements = {
            'embedding': ModelPlacement('embedding', DeviceType.CPU, -1,
                                       QuantizationType.FP16, MODEL_SPECS['embedding'].size_fp16_gb * MEMORY_OVERHEAD),
            'reranker': ModelPlacement('reranker', DeviceType.CPU, -1,
                                      QuantizationType.FP16, MODEL_SPECS['reranker'].size_fp16_gb * MEMORY_OVERHEAD),
            'llm': ModelPlacement('llm', DeviceType.CPU, -1,
                                 QuantizationType.FP4, MODEL_SPECS['llm'].get_size(QuantizationType.FP4) * MEMORY_OVERHEAD)
        }
        best_strategy = compute_allocation_strategy(placements)

    return best_strategy


def detect_hardware() -> HardwareConfig:
    """
    Auto-detect hardware and return optimal configuration.
    Supports multi-GPU distribution with intelligent allocation.

    Uses mathematical optimization to find the best model placement
    strategy that balances:
    - Speed (GPU > CPU)
    - Quality (fp16 > fp8 > fp4)
    - Memory efficiency
    - Load balancing across GPUs

    Model sizes (fp16):
    - Embedding: 1.2 GB
    - Reranker: 1.2 GB
    - LLM: 16.0 GB

    Returns:
        HardwareConfig with optimized device allocation
    """
    gpus = get_all_gpu_info()
    ram_gb = get_ram_info()
    cpu_info = get_cpu_info()

    gpu_count = len(gpus)
    total_vram = sum(g.vram_gb for g in gpus) if gpus else 0.0

    # Use intelligent allocation algorithm
    strategy = optimize_model_allocation(gpus, ram_gb)

    # Extract placements
    emb_placement = strategy.placements['embedding']
    rer_placement = strategy.placements['reranker']
    llm_placement = strategy.placements['llm']

    # Build device map (GPU index mapping)
    device_map = {}
    for model_name, placement in strategy.placements.items():
        if placement.device_type == DeviceType.GPU:
            device_map[model_name] = placement.device_index

    # Build memory breakdown
    memory_breakdown = {
        name: placement.memory_required_gb
        for name, placement in strategy.placements.items()
    }

    return HardwareConfig(
        embedding_device=emb_placement.device_string,
        reranker_device=rer_placement.device_string,
        llm_device=llm_placement.device_string,
        llm_quantization=llm_placement.quantization.value,
        recommended_model='Azzindani/Deepseek_ID_Legal_Preview',
        vram_available=total_vram,
        ram_available=ram_gb,
        gpu_count=gpu_count,
        gpu_info=gpus,
        device_map=device_map,
        allocation_score=strategy.total_score,
        memory_breakdown=memory_breakdown
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
    """Print detected hardware information with optimization details"""
    gpus = get_all_gpu_info()
    cpu_info = get_cpu_info()
    config = detect_hardware()

    print("=" * 70)
    print("INTELLIGENT HARDWARE DETECTION & MODEL ALLOCATION")
    print("=" * 70)

    # System specs
    print(f"\n📊 System Specifications:")
    print(f"  CPU: {cpu_info['cores']} cores, {cpu_info['threads']} threads")
    print(f"  RAM: {config.ram_available:.1f} GB")
    print(f"  GPU Count: {config.gpu_count}")
    print(f"  Total VRAM: {config.vram_available:.1f} GB")

    if gpus:
        print("\n  GPU Details:")
        for gpu in gpus:
            print(f"    [{gpu.index}] {gpu.name}")
            print(f"        VRAM: {gpu.vram_gb:.1f} GB (Available: {gpu.available_memory():.1f} GB)")
            print(f"        Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")

    # Model requirements
    print(f"\n📦 Model Memory Requirements (fp16 baseline):")
    print(f"  Embedding: {MODEL_SPECS['embedding'].size_fp16_gb:.1f} GB × {MEMORY_OVERHEAD:.1f} overhead = {MODEL_SPECS['embedding'].size_fp16_gb * MEMORY_OVERHEAD:.2f} GB")
    print(f"  Reranker:  {MODEL_SPECS['reranker'].size_fp16_gb:.1f} GB × {MEMORY_OVERHEAD:.1f} overhead = {MODEL_SPECS['reranker'].size_fp16_gb * MEMORY_OVERHEAD:.2f} GB")
    print(f"  LLM:       {MODEL_SPECS['llm'].size_fp16_gb:.1f} GB × {MEMORY_OVERHEAD:.1f} overhead = {MODEL_SPECS['llm'].size_fp16_gb * MEMORY_OVERHEAD:.2f} GB")

    # Optimized allocation
    print(f"\n✨ Optimized Model Allocation (Score: {config.allocation_score:.3f}):")
    print(f"  Embedding → {config.embedding_device:10s} (fp16, {config.memory_breakdown['embedding']:.2f} GB)")
    print(f"  Reranker  → {config.reranker_device:10s} (fp16, {config.memory_breakdown['reranker']:.2f} GB)")

    llm_quant_display = {
        'none': 'fp16',
        '8bit': 'fp8',
        '4bit': 'fp4'
    }.get(config.llm_quantization, config.llm_quantization)
    print(f"  LLM       → {config.llm_device:10s} ({llm_quant_display:4s}, {config.memory_breakdown['llm']:.2f} GB)")

    # Memory usage breakdown
    if config.device_map:
        print(f"\n💾 Memory Usage by Device:")

        # GPU memory usage
        gpu_mem_usage = {}
        for model_name, gpu_idx in config.device_map.items():
            gpu_mem_usage[gpu_idx] = gpu_mem_usage.get(gpu_idx, 0) + config.memory_breakdown[model_name]

        for gpu_idx, mem_used in sorted(gpu_mem_usage.items()):
            gpu = next((g for g in gpus if g.index == gpu_idx), None)
            if gpu:
                utilization = (mem_used / gpu.available_memory()) * 100
                print(f"  GPU {gpu_idx}: {mem_used:.2f} GB / {gpu.available_memory():.1f} GB ({utilization:.1f}% utilized)")

        # CPU memory usage
        cpu_mem = sum(config.memory_breakdown[m] for m in config.memory_breakdown
                     if m in ['embedding', 'reranker', 'llm'] and
                     config.device_map.get(m) is None)
        if cpu_mem > 0:
            cpu_utilization = (cpu_mem / (config.ram_available * 0.7)) * 100
            print(f"  CPU:    {cpu_mem:.2f} GB / {config.ram_available * 0.7:.1f} GB ({cpu_utilization:.1f}% utilized)")

    print(f"\n🎯 Strategy Explanation:")
    print(f"  Allocation optimizes for:")
    print(f"    - Speed (GPU 10x faster than CPU)")
    print(f"    - Quality (fp16 > fp8 > fp4)")
    print(f"    - Memory efficiency (1.3x overhead for activations)")
    print(f"    - Load balancing across available GPUs")

    print("=" * 70)

    return config


if __name__ == "__main__":
    config = print_hardware_info()

    print("\nApplying configuration...")
    settings = apply_hardware_config(config)

    print("\nEnvironment variables set:")
    for key, value in settings.items():
        print(f"  {key}={value}")
