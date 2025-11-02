#!/bin/bash
################################################################################
# Setup Real vLLM on GCP VM with L4 GPU
# This script installs vLLM and downloads models for testing
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🚀 vLLM Setup Script for GCP L4 GPU"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Step 1: Check GPU
echo "Step 1: Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. GPU drivers not installed!"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

print_success "GPU detected: $GPU_NAME"
print_success "GPU memory: ${GPU_MEMORY}MB"

if [[ $GPU_MEMORY -lt 20000 ]]; then
    print_error "GPU memory < 20GB. vLLM works best with ≥20GB VRAM"
    exit 1
fi

# Step 2: Check Python
echo ""
echo "Step 2: Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python version: $PYTHON_VERSION"

# Check if venv exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
print_success "Virtual environment activated"

# Step 3: Install vLLM
echo ""
echo "Step 3: Installing vLLM..."

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install vLLM
print_info "Installing vLLM (this may take several minutes)..."
pip install vllm

# Verify installation
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
    print_success "vLLM installed successfully: version $VLLM_VERSION"
else
    print_error "vLLM installation failed!"
    exit 1
fi

# Step 4: Install dependencies
echo ""
echo "Step 4: Installing additional dependencies..."
pip install \
    torch \
    transformers \
    accelerate \
    huggingface-hub \
    fastapi \
    uvicorn \
    pytest \
    pytest-cov \
    pandas \
    numpy \
    matplotlib \
    seaborn

print_success "Dependencies installed"

# Step 5: Setup HuggingFace cache
echo ""
echo "Step 5: Setting up HuggingFace cache..."
mkdir -p ~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

print_success "HuggingFace cache directory created"

# Step 6: Download models
echo ""
echo "Step 6: Downloading models..."
print_info "This will download ~5GB of models"

# Create models directory
mkdir -p models

# Function to download model
download_model() {
    local model_name=$1
    local local_dir=$2

    print_info "Downloading $model_name..."
    python3 << EOF
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        repo_id="$model_name",
        local_dir="$local_dir",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✅ Downloaded $model_name")
except Exception as e:
    print(f"❌ Failed to download $model_name: {e}")
    exit(1)
EOF
}

# Download Phi-2 (2.7B) - Small and fast
print_info "Downloading Phi-2 (2.7B)..."
download_model "microsoft/phi-2" "./models/phi-2"

# Download Mistral-7B (optional, uncomment if needed)
# print_info "Downloading Mistral-7B-Instruct-v0.2..."
# download_model "mistralai/Mistral-7B-Instruct-v0.2" "./models/mistral-7b"

print_success "Models downloaded"

# Step 7: Test vLLM
echo ""
echo "Step 7: Testing vLLM installation..."

python3 << 'EOTEST'
import torch
from vllm import LLM, SamplingParams

print("Testing vLLM with Phi-2...")

# Initialize vLLM
llm = LLM(
    model="microsoft/phi-2",
    gpu_memory_utilization=0.8,
    max_model_len=2048,
    trust_remote_code=True
)

# Test generation
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated text: {output.outputs[0].text}")

print("\n✅ vLLM test successful!")
EOTEST

if [ $? -eq 0 ]; then
    print_success "vLLM is working correctly!"
else
    print_error "vLLM test failed!"
    exit 1
fi

# Step 8: Create test script
echo ""
echo "Step 8: Creating test scripts..."

cat > test_vllm_quick.py << 'EOF'
#!/usr/bin/env python3
"""Quick vLLM test script"""

from vllm import LLM, SamplingParams
import time

print("Quick vLLM Test")
print("="*60)

# Load model
print("Loading Phi-2...")
start = time.time()
llm = LLM(
    model="microsoft/phi-2",
    gpu_memory_utilization=0.8,
    max_model_len=2048,
    trust_remote_code=True
)
load_time = time.time() - start
print(f"Model loaded in {load_time:.2f}s")

# Test generation
prompts = [
    "What is vLLM?",
    "Explain PagedAttention.",
    "Why is vLLM fast?"
]

print(f"\nGenerating responses for {len(prompts)} prompts...")
start = time.time()
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
gen_time = time.time() - start

print(f"Generation took {gen_time:.2f}s")
print(f"Throughput: {len(prompts)/gen_time:.2f} req/sec")

for i, output in enumerate(outputs):
    print(f"\nPrompt {i+1}: {output.prompt}")
    print(f"Response: {output.outputs[0].text[:100]}...")

print("\n✅ Test complete!")
EOF

chmod +x test_vllm_quick.py
print_success "Test script created: test_vllm_quick.py"

# Step 9: Update model_loader.py
echo ""
echo "Step 9: Updating model_loader.py for real vLLM..."

if [ -f "src/model_loader.py" ]; then
    print_info "Backing up model_loader.py..."
    cp src/model_loader.py src/model_loader.py.bak
    print_success "Backup created: src/model_loader.py.bak"
    print_info "You can now update model_loader.py to use real vLLM"
else
    print_error "src/model_loader.py not found"
fi

# Step 10: Summary
echo ""
echo "================================================================================"
echo "✅ vLLM Setup Complete!"
echo "================================================================================"
echo ""
echo "📋 Summary:"
echo "  - GPU: $GPU_NAME ($GPU_MEMORY MB)"
echo "  - Python: $PYTHON_VERSION"
echo "  - vLLM: $VLLM_VERSION"
echo "  - Models: Phi-2 (2.7B)"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Test vLLM:"
echo "   python test_vllm_quick.py"
echo ""
echo "2. Run examples:"
echo "   python examples/01_basic_vllm.py"
echo ""
echo "3. Run benchmarks:"
echo "   cd benchmarks && ./run_all_benchmarks.sh"
echo ""
echo "4. Update model_loader.py to use real vLLM instead of mocks"
echo ""
echo "5. Run tests with real vLLM:"
echo "   ./run_all_tests.sh"
echo ""
echo "================================================================================"

# Save environment info
cat > vllm_setup_info.txt << EOF
vLLM Setup Information
======================
Date: $(date)
GPU: $GPU_NAME
GPU Memory: ${GPU_MEMORY}MB
Python: $PYTHON_VERSION
vLLM: $VLLM_VERSION
Models: microsoft/phi-2
EOF

print_success "Setup info saved to vllm_setup_info.txt"
