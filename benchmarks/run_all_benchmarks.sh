#!/bin/bash
################################################################################
# Run All vLLM Benchmarks for Blog Series
# Generates comprehensive performance data for blog posts
################################################################################

set -e

echo "================================================================================"
echo "🔬 vLLM Comprehensive Benchmark Suite"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p results

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. GPU required for benchmarks."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "🎮 GPU: $GPU_NAME"
echo "💾 VRAM: ${GPU_MEMORY}MB"
echo ""

# Check Python environment
if [ ! -d "../venv" ]; then
    echo "❌ Virtual environment not found. Run setup_real_vllm.sh first"
    exit 1
fi

source ../venv/bin/activate

# Check vLLM installation
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed. Run setup_real_vllm.sh first"
    exit 1
fi

# Install transformers for comparison (if not already installed)
echo "Checking transformers installation..."
if ! python -c "import transformers" 2>/dev/null; then
    echo "Installing transformers for comparison..."
    pip install transformers>=4.37.0 accelerate
else
    echo "✅ Transformers already installed"
fi

echo "✅ Environment ready"
echo ""

# Benchmark 1: Throughput Comparison
echo "================================================================================"
echo "Benchmark 1: Throughput Comparison (vLLM vs Transformers)"
echo "================================================================================"
python 01_throughput_comparison.py
echo ""

# Benchmark 2: Memory Efficiency
echo "================================================================================"
echo "Benchmark 2: Memory Efficiency (PagedAttention benefits)"
echo "================================================================================"
if [ -f "02_memory_efficiency.py" ]; then
    python 02_memory_efficiency.py
else
    echo "⚠️  02_memory_efficiency.py not found, skipping..."
fi
echo ""

# Benchmark 3: Cost Analysis
echo "================================================================================"
echo "Benchmark 3: Cost Analysis"
echo "================================================================================"
if [ -f "03_cost_analysis.py" ]; then
    python 03_cost_analysis.py
else
    echo "⚠️  03_cost_analysis.py not found, skipping..."
fi
echo ""

# Benchmark 4: Latency Distribution
echo "================================================================================"
echo "Benchmark 4: Latency Distribution (P50/P95/P99)"
echo "================================================================================"
if [ -f "04_latency_distribution.py" ]; then
    python 04_latency_distribution.py
else
    echo "⚠️  04_latency_distribution.py not found, skipping..."
fi
echo ""

# Generate summary report
echo "================================================================================"
echo "📊 Generating Summary Report"
echo "================================================================================"

cat > results/benchmark_summary.txt << EOF
vLLM Benchmark Summary
======================
Date: $(date)
GPU: $GPU_NAME
GPU Memory: ${GPU_MEMORY}MB

Benchmark Results:
------------------
$(ls -1 results/*.json 2>/dev/null | wc -l) benchmark files generated

Files:
$(ls -1 results/ 2>/dev/null || echo "No results yet")

Next Steps:
-----------
1. Review results/throughput_results.csv
2. Check results/*.json for detailed metrics
3. Use data in blog posts
4. Generate visualizations with plot_benchmarks.py

EOF

cat results/benchmark_summary.txt

echo ""
echo "================================================================================"
echo "✅ All Benchmarks Complete!"
echo "================================================================================"
echo ""
echo "📁 Results saved to: benchmarks/results/"
echo ""
echo "💡 Next steps:"
echo "  1. Review benchmark_summary.txt"
echo "  2. Analyze CSV files for blog metrics"
echo "  3. Generate charts: python plot_benchmarks.py"
echo "  4. Include results in blog post"
echo ""
