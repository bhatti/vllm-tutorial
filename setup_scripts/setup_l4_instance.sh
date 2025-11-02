#!/bin/bash
# Complete setup script for your GCP L4 instance
# Run this after SSHing into your VM

set -e  # Exit on error

echo "================================================"
echo "vLLM Setup Script for GCP L4 Instance"
echo "================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: System Update and GPU Check
print_status "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

print_status "Checking GPU status..."
if nvidia-smi > /dev/null 2>&1; then
    print_status "GPU detected successfully!"
    nvidia-smi
else
    print_error "GPU not detected. Installing NVIDIA drivers..."

    # Install NVIDIA drivers
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall

    print_warning "Rebooting to load drivers. Please re-run this script after reboot."
    sudo reboot
fi

# Step 2: Install Python 3.10 and essential packages
print_status "Step 2: Installing Python 3.10 and dependencies..."
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    htop \
    nvtop \
    build-essential \
    libssl-dev \
    libffi-dev

# Step 3: Install Docker (for containerized deployment)
print_status "Step 3: Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    print_warning "Docker installed. You may need to logout and login for group changes."
fi

# Install NVIDIA Container Toolkit
print_status "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Step 4: Create project directory structure
print_status "Step 4: Creating project structure..."
cd ~
mkdir -p vllm-project/{models,scripts,logs,data,configs}
cd vllm-project

# Step 5: Create Python virtual environment
print_status "Step 5: Setting up Python environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Step 6: Install vLLM and dependencies
print_status "Step 6: Installing vLLM and dependencies..."
pip install vllm==0.5.4  # Stable version

# Install additional dependencies
pip install \
    torch==2.1.2 \
    transformers==4.36.2 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pandas==2.1.4 \
    numpy==1.24.3 \
    pydantic==2.5.0 \
    python-multipart \
    prometheus-client==0.19.0 \
    psutil==5.9.6 \
    gputil==1.4.0 \
    huggingface-hub==0.20.1 \
    sentencepiece==0.1.99 \
    protobuf==4.25.1 \
    accelerate==0.25.0

# Install observability tools
print_status "Installing observability packages..."
pip install \
    langfuse==2.20.0 \
    phoenix-arize==2.0.0 \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0 \
    plotly==5.18.0

# Step 7: Download models suitable for L4 (24GB VRAM)
print_status "Step 7: Downloading models..."

# Login to Hugging Face (optional, for gated models)
print_warning "Setting up Hugging Face access..."
echo "Please enter your Hugging Face token (press Enter to skip for public models):"
read -s HF_TOKEN

if [ ! -z "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
    print_status "Logged in to Hugging Face"
else
    print_warning "Skipping Hugging Face login. Only public models will be available."
fi

# Download models
print_status "Downloading Phi-2 (2.7B - fastest)..."
huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2

print_status "Downloading Mistral-7B-Instruct (best for general use)..."
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir ./models/mistral-7b

# Optional: Download Llama-2-7B (requires HF login and approval)
if [ ! -z "$HF_TOKEN" ]; then
    print_status "Attempting to download Llama-2-7B-chat..."
    huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/llama-7b || print_warning "Llama-2 download failed. You may need to request access at https://huggingface.co/meta-llama"
fi

# Step 8: Create test scripts
print_status "Step 8: Creating test scripts..."

# Create basic vLLM test
cat > test_vllm.py << 'EOF'
#!/usr/bin/env python3
"""
Basic vLLM test script to verify installation
"""
import torch
from vllm import LLM, SamplingParams
import time

def test_vllm():
    print("=" * 60)
    print("vLLM Installation Test")
    print("=" * 60)

    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test with Phi-2 (smallest model)
    print("\n" + "=" * 60)
    print("Testing inference with Phi-2...")
    print("=" * 60)

    try:
        llm = LLM(
            model="./models/phi-2",
            gpu_memory_utilization=0.9,
            max_model_len=2048
        )

        prompts = [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers."
        ]

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=100
        )

        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        for i, output in enumerate(outputs):
            print(f"\nPrompt {i+1}: {output.prompt[:50]}...")
            print(f"Response: {output.outputs[0].text[:200]}...")
            print(f"Tokens generated: {len(output.outputs[0].token_ids)}")

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"\n" + "=" * 60)
        print(f"✓ Test completed successfully!")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Total tokens: {total_tokens}")
        print(f"Throughput: {total_tokens / (end_time - start_time):.1f} tokens/sec")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    test_vllm()
EOF

chmod +x test_vllm.py

# Create API server script
cat > vllm_server.py << 'EOF'
#!/usr/bin/env python3
"""
vLLM API Server with multiple models
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from vllm import LLM, SamplingParams
import time
import psutil
import GPUtil

app = FastAPI(title="vLLM Server", version="1.0.0")

# Global model storage
models = {}

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "phi-2"
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False

class CompletionResponse(BaseModel):
    text: str
    model: str
    tokens: int
    latency_ms: float

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Loading models...")

    # Load Phi-2 (fast, small)
    models["phi-2"] = LLM(
        model="./models/phi-2",
        gpu_memory_utilization=0.4,  # Use only 40% for this model
        max_model_len=2048
    )
    print("✓ Phi-2 loaded")

    # Load Mistral-7B (better quality)
    models["mistral-7b"] = LLM(
        model="./models/mistral-7b",
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )
    print("✓ Mistral-7B loaded")

    print("All models loaded successfully!")

@app.get("/")
async def root():
    return {
        "message": "vLLM Server Running",
        "available_models": list(models.keys()),
        "endpoints": {
            "/complete": "POST - Generate completion",
            "/models": "GET - List available models",
            "/health": "GET - Health check"
        }
    }

@app.get("/models")
async def list_models():
    return {"models": list(models.keys())}

@app.get("/health")
async def health_check():
    gpus = GPUtil.getGPUs()
    return {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_utilization": gpus[0].load * 100 if gpus else 0,
        "gpu_memory_used": gpus[0].memoryUsed if gpus else 0,
        "models_loaded": list(models.keys())
    }

@app.post("/complete", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available")

    start_time = time.time()

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )

    outputs = models[request.model].generate([request.prompt], sampling_params)

    generated_text = outputs[0].outputs[0].text
    token_count = len(outputs[0].outputs[0].token_ids)
    latency = (time.time() - start_time) * 1000

    return CompletionResponse(
        text=generated_text,
        model=request.model,
        tokens=token_count,
        latency_ms=latency
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

chmod +x vllm_server.py

# Create benchmark script
cat > benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive vLLM Benchmark Script
"""
import time
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np

def benchmark_vllm(model_path, prompts, max_tokens=256):
    """Benchmark vLLM performance"""
    print(f"\nBenchmarking vLLM with {model_path}...")

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens
    )

    # Warmup
    _ = llm.generate(prompts[:1], sampling_params)

    # Actual benchmark
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_time = end_time - start_time

    return {
        "framework": "vLLM",
        "model": model_path,
        "total_time": total_time,
        "prompts": len(prompts),
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time,
        "avg_latency_per_prompt": total_time / len(prompts) * 1000  # ms
    }

def benchmark_transformers(model_path, prompts, max_tokens=256):
    """Benchmark HuggingFace Transformers (for comparison)"""
    print(f"\nBenchmarking Transformers with {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_tokens = 0
    start_time = time.time()

    for prompt in prompts[:3]:  # Only do 3 for Transformers (slower)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        total_tokens += outputs.shape[1] - inputs['input_ids'].shape[1]

    end_time = time.time()
    total_time = end_time - start_time

    return {
        "framework": "Transformers",
        "model": model_path,
        "total_time": total_time,
        "prompts": 3,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time,
        "avg_latency_per_prompt": total_time / 3 * 1000  # ms
    }

def main():
    # Test prompts
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to sort a list of numbers.",
        "What are the main differences between CPU and GPU?",
        "Describe the process of photosynthesis.",
        "Create a simple REST API endpoint in Python.",
        "Explain quantum computing to a 10-year-old.",
        "What are the best practices for writing clean code?",
        "Describe the water cycle in nature.",
        "Write a SQL query to find duplicate records.",
        "Explain the concept of blockchain technology."
    ]

    results = []

    # Benchmark vLLM
    vllm_result = benchmark_vllm("./models/phi-2", prompts)
    results.append(vllm_result)

    # Benchmark Transformers (for comparison)
    try:
        transformers_result = benchmark_transformers("./models/phi-2", prompts)
        results.append(transformers_result)
    except Exception as e:
        print(f"Transformers benchmark failed: {e}")

    # Display results
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)

    if len(results) == 2:
        speedup = results[0]["tokens_per_second"] / results[1]["tokens_per_second"]
        print(f"\n🚀 vLLM is {speedup:.1f}x faster than Transformers!")

    # Save results
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
EOF

chmod +x benchmark.py

# Step 9: Create systemd service for auto-start
print_status "Step 9: Creating systemd service..."
sudo tee /etc/systemd/system/vllm.service << EOF
[Unit]
Description=vLLM API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/vllm-project
Environment="PATH=/home/$USER/vllm-project/venv/bin"
ExecStart=/home/$USER/vllm-project/venv/bin/python /home/$USER/vllm-project/vllm_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
# Don't start automatically yet
# sudo systemctl enable vllm.service

# Step 10: Setup firewall rules
print_status "Step 10: Configuring firewall..."
sudo ufw allow 8080/tcp  # vLLM API
sudo ufw allow 22/tcp    # SSH
sudo ufw --force enable

# Step 11: Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
# Simple monitoring script
while true; do
    clear
    echo "=== vLLM Instance Monitor ==="
    echo "Time: $(date)"
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
    echo "=== CPU & Memory ==="
    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')%"
    echo "Memory: $(free -h | grep "^Mem" | awk '{print $3 " / " $2}')"
    echo ""
    echo "=== Disk Usage ==="
    df -h / | tail -1
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
EOF
chmod +x monitor.sh

# Final summary
echo ""
print_status "=========================================="
print_status "Setup Complete!"
print_status "=========================================="
echo ""
echo "Next steps:"
echo "1. Test vLLM installation:"
echo "   source venv/bin/activate"
echo "   python test_vllm.py"
echo ""
echo "2. Run benchmark:"
echo "   python benchmark.py"
echo ""
echo "3. Start API server:"
echo "   python vllm_server.py"
echo "   (Server will run on http://0.0.0.0:8080)"
echo ""
echo "4. Monitor system:"
echo "   ./monitor.sh"
echo ""
echo "5. For production deployment:"
echo "   sudo systemctl start vllm.service"
echo "   sudo systemctl enable vllm.service"
echo ""
print_warning "Remember to configure your firewall if you want external access!"
print_status "Happy LLM serving with vLLM! 🚀"