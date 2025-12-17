#!/bin/bash
# Automatic Memory-Aware Test Runner
# Detects available GPU memory and sets appropriate parameters

echo "=========================================="
echo "MEMORY-AWARE TEST RUNNER"
echo "=========================================="
echo

# Check if Python and PyTorch are available
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ Error: PyTorch not found. Please install PyTorch."
    exit 1
fi

# Get available GPU memory
echo "🔍 Detecting GPU memory..."
GPU_MEM=$(python3 << 'EOF'
import torch
if torch.cuda.is_available():
    free_mem_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"{free_mem_gb:.2f}")
else:
    print("0")
EOF
)

echo "   Available GPU Memory: ${GPU_MEM} GB"
echo

# Determine optimal settings based on available memory
if (( $(echo "$GPU_MEM < 3.0" | bc -l) )); then
    # Very limited memory
    THINKING_MODE="--low"
    MAX_NEW_TOKENS=512
    ADVICE="⚠️  Very limited GPU memory. Using minimal settings."
    echo "$ADVICE"
    echo "   Recommended: Use CPU inference or cloud GPU"
elif (( $(echo "$GPU_MEM < 4.0" | bc -l) )); then
    # Limited memory
    THINKING_MODE="--low"
    MAX_NEW_TOKENS=768
    ADVICE="⚠️  Limited GPU memory. Using low thinking mode."
    echo "$ADVICE"
elif (( $(echo "$GPU_MEM < 5.0" | bc -l) )); then
    # Moderate memory
    THINKING_MODE="--low"
    MAX_NEW_TOKENS=1024
    ADVICE="✅ Moderate GPU memory. Low thinking mode recommended."
    echo "$ADVICE"
elif (( $(echo "$GPU_MEM < 6.0" | bc -l) )); then
    # Good memory
    THINKING_MODE="--medium"
    MAX_NEW_TOKENS=1024
    ADVICE="✅ Good GPU memory. Medium thinking mode available."
    echo "$ADVICE"
elif (( $(echo "$GPU_MEM < 8.0" | bc -l) )); then
    # Very good memory
    THINKING_MODE="--medium"
    MAX_NEW_TOKENS=1536
    ADVICE="✅ Very good GPU memory. Medium thinking mode recommended."
    echo "$ADVICE"
else
    # Excellent memory
    THINKING_MODE="--high"
    MAX_NEW_TOKENS=2048
    ADVICE="🚀 Excellent GPU memory. All thinking modes available."
    echo "$ADVICE"
fi

echo
echo "📋 Auto-configured settings:"
echo "   Thinking Mode: $THINKING_MODE"
echo "   MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo

# Export environment variable
export MAX_NEW_TOKENS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Parse command line arguments
TEST_FILE=""
EXTRA_ARGS=""

# Check if test file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <test_file> [extra_args]"
    echo
    echo "Examples:"
    echo "  $0 tests/integration/test_complete_output.py"
    echo "  $0 tests/integration/test_conversational.py --verbose"
    echo "  $0 tests/integration/test_stress_single.py --export"
    exit 1
fi

TEST_FILE=$1
shift
EXTRA_ARGS="$@"

# Verify test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

echo "🚀 Running test with auto-configured settings..."
echo
echo "Command: python $TEST_FILE $THINKING_MODE $EXTRA_ARGS"
echo "=========================================="
echo

# Run the test
python "$TEST_FILE" $THINKING_MODE $EXTRA_ARGS

EXIT_CODE=$?

echo
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
else
    echo "❌ Test failed with exit code: $EXIT_CODE"
    echo
    echo "💡 Troubleshooting tips:"
    echo "   1. Clear GPU cache: python -c 'import torch; torch.cuda.empty_cache()'"
    echo "   2. Try lower thinking mode: --low"
    echo "   3. Reduce tokens manually: export MAX_NEW_TOKENS=512"
    echo "   4. Use quick mode: --quick (for stress tests)"
fi
echo "=========================================="

exit $EXIT_CODE
