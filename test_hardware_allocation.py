"""
Test script to demonstrate intelligent hardware allocation.

This script simulates different hardware scenarios and shows
how the optimization algorithm allocates models.
"""

from hardware_detection import (
    GPUInfo, detect_hardware, optimize_model_allocation,
    MODEL_SPECS, MEMORY_OVERHEAD, print_hardware_info
)


def simulate_scenario(scenario_name: str, gpus: list, ram_gb: float):
    """Simulate a hardware scenario and show allocation"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    if gpus:
        print(f"\nGPUs:")
        for gpu in gpus:
            print(f"  [{gpu.index}] {gpu.name}: {gpu.vram_gb:.1f} GB VRAM")
    else:
        print(f"\nNo GPUs detected")

    print(f"RAM: {ram_gb:.1f} GB")

    # Run optimization
    strategy = optimize_model_allocation(gpus, ram_gb)

    print(f"\n✨ Optimized Allocation (Score: {strategy.total_score:.3f}):")
    for model_name, placement in strategy.placements.items():
        quant_display = {
            'none': 'fp16',
            '8bit': 'fp8',
            '4bit': 'fp4'
        }.get(placement.quantization.value, placement.quantization.value)

        print(f"  {model_name:10s} → {placement.device_string:10s} "
              f"({quant_display:4s}, {placement.memory_required_gb:.2f} GB)")

    print(f"\n💾 Memory Usage:")
    for gpu_idx, mem_used in sorted(strategy.memory_usage.items()):
        if gpu_idx >= 0:
            gpu = next((g for g in gpus if g.index == gpu_idx), None)
            if gpu:
                util = (mem_used / gpu.available_memory()) * 100
                print(f"  GPU {gpu_idx}: {mem_used:.2f} GB / {gpu.available_memory():.1f} GB ({util:.1f}%)")

    if strategy.cpu_memory_gb > 0:
        cpu_util = (strategy.cpu_memory_gb / (ram_gb * 0.7)) * 100
        print(f"  CPU:    {strategy.cpu_memory_gb:.2f} GB / {ram_gb * 0.7:.1f} GB ({cpu_util:.1f}%)")


def main():
    print("=" * 70)
    print("INTELLIGENT HARDWARE ALLOCATION - TEST SCENARIOS")
    print("=" * 70)

    # Scenario 1: Single GPU with 8GB VRAM (tight fit)
    simulate_scenario(
        "Single GPU 8GB (Low VRAM)",
        gpus=[GPUInfo(0, "GeForce GTX 1070", 8.0)],
        ram_gb=16.0
    )

    # Scenario 2: Single GPU with 12GB VRAM (medium)
    simulate_scenario(
        "Single GPU 12GB (Medium VRAM)",
        gpus=[GPUInfo(0, "GeForce RTX 3060", 12.0)],
        ram_gb=16.0
    )

    # Scenario 3: Single GPU with 24GB VRAM (plenty)
    simulate_scenario(
        "Single GPU 24GB (High VRAM)",
        gpus=[GPUInfo(0, "GeForce RTX 3090", 24.0)],
        ram_gb=32.0
    )

    # Scenario 4: Dual GPU 16GB each (your target scenario)
    simulate_scenario(
        "Dual GPU 16GB each (Balanced Multi-GPU)",
        gpus=[
            GPUInfo(0, "Tesla P100", 16.0),
            GPUInfo(1, "Tesla P100", 16.0)
        ],
        ram_gb=32.0
    )

    # Scenario 5: Triple GPU (3x 16GB)
    simulate_scenario(
        "Triple GPU 16GB each (Rich Multi-GPU)",
        gpus=[
            GPUInfo(0, "Tesla V100", 16.0),
            GPUInfo(1, "Tesla V100", 16.0),
            GPUInfo(2, "Tesla V100", 16.0)
        ],
        ram_gb=64.0
    )

    # Scenario 6: Mixed GPU sizes
    simulate_scenario(
        "Mixed GPUs (24GB + 12GB + 8GB)",
        gpus=[
            GPUInfo(0, "RTX 3090", 24.0),
            GPUInfo(1, "RTX 3060", 12.0),
            GPUInfo(2, "GTX 1070", 8.0)
        ],
        ram_gb=32.0
    )

    # Scenario 7: CPU only
    simulate_scenario(
        "CPU Only (No GPU)",
        gpus=[],
        ram_gb=32.0
    )

    # Scenario 8: Low-end GPU (not enough for anything)
    simulate_scenario(
        "Low-end GPU 4GB (Insufficient VRAM)",
        gpus=[GPUInfo(0, "GeForce GTX 1050", 4.0)],
        ram_gb=8.0
    )

    print(f"\n{'='*70}")
    print("REAL HARDWARE DETECTION")
    print(f"{'='*70}")
    print_hardware_info()


if __name__ == "__main__":
    main()
