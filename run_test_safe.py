#!/usr/bin/env python3
"""
Safe Test Runner with Automatic Memory Management
Automatically configures thinking mode and max_new_tokens based on available GPU memory

Usage:
    python run_test_safe.py <test_file> [extra_args]

Examples:
    python run_test_safe.py tests/integration/test_complete_output.py
    python run_test_safe.py tests/integration/test_conversational.py --verbose
    python run_test_safe.py tests/integration/test_stress_single.py --export
"""

import sys
import os
import subprocess


def get_available_gpu_memory():
    """Get available GPU memory in GB"""
    try:
        import torch
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0]
            return free_mem / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def determine_safe_settings(gpu_mem_gb):
    """
    Suggest thinking mode based on available GPU memory.
    Does NOT limit max_new_tokens - uses configured value.

    Returns:
        tuple: (thinking_mode_flag, advice)
    """
    if gpu_mem_gb < 3.0:
        return (
            "--low",
            "⚠️  Very limited GPU memory detected.\n"
            "   Suggestion: Use --low thinking mode\n"
            "   If OOM occurs: export MAX_NEW_TOKENS=512"
        )
    elif gpu_mem_gb < 4.0:
        return (
            "--low",
            "⚠️  Limited GPU memory detected.\n"
            "   Suggestion: Use --low thinking mode"
        )
    elif gpu_mem_gb < 5.0:
        return (
            "--low",
            "✅ Moderate GPU memory. --low or --medium should work."
        )
    elif gpu_mem_gb < 6.0:
        return (
            "--medium",
            "✅ Good GPU memory. --medium thinking mode recommended."
        )
    elif gpu_mem_gb < 8.0:
        return (
            "--medium",
            "✅ Very good GPU memory. --medium or --high should work."
        )
    else:
        return (
            "--high",
            "🚀 Excellent GPU memory. All thinking modes available."
        )


def main():
    print("=" * 60)
    print("SAFE TEST RUNNER - AUTOMATIC MEMORY MANAGEMENT")
    print("=" * 60)
    print()

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python run_test_safe.py <test_file> [extra_args]")
        print()
        print("Examples:")
        print("  python run_test_safe.py tests/integration/test_complete_output.py")
        print("  python run_test_safe.py tests/integration/test_conversational.py --verbose")
        print("  python run_test_safe.py tests/integration/test_stress_single.py --export")
        sys.exit(1)

    test_file = sys.argv[1]
    extra_args = sys.argv[2:]

    # Verify test file exists
    if not os.path.exists(test_file):
        print(f"❌ Error: Test file not found: {test_file}")
        sys.exit(1)

    # Detect GPU memory
    print("🔍 Detecting GPU memory...")
    gpu_mem = get_available_gpu_memory()
    print(f"   Available GPU Memory: {gpu_mem:.2f} GB")
    print()

    # Determine settings
    thinking_mode, advice = determine_safe_settings(gpu_mem)

    print(advice)
    print()
    print("📋 Suggested settings:")
    print(f"   Thinking Mode: {thinking_mode}")
    print(f"   MAX_NEW_TOKENS: Using configured value from config.py")
    print()

    # Set environment variables
    # Note: We don't override MAX_NEW_TOKENS - user's GPU can handle it
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Build command
    cmd = [sys.executable, test_file, thinking_mode] + extra_args

    print("🚀 Running test with auto-configured settings...")
    print()
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print()

    # Run test
    try:
        result = subprocess.run(cmd)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    if exit_code == 0:
        print("✅ Test completed successfully!")
    else:
        print(f"❌ Test failed with exit code: {exit_code}")
        print()
        print("💡 Troubleshooting tips:")
        print("   1. Clear GPU cache first:")
        print("      python -c 'import torch; torch.cuda.empty_cache()'")
        print("   2. Try with even lower settings:")
        print("      export MAX_NEW_TOKENS=384")
        print("      python", test_file, "--low")
        print("   3. Use quick mode for stress tests:")
        print("      python", test_file, "--quick --low")
        print("   4. Check GPU usage:")
        print("      nvidia-smi")
    print("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
