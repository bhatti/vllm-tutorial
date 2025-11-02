# vLLM Production Deployment Guide

**Simple Docker deployment for real-world production use**

---

## 📋 Prerequisites

### Required
- NVIDIA GPU with 24GB+ VRAM (L4, A10, A100, etc.)
- Docker 20.10+ with NVIDIA Container Toolkit
- 50GB+ disk space (for model + Docker image)

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Quick Start (2 minutes)

### 1. Clone & Configure

```bash
# Clone repository
git clone https://github.com/yourusername/vllm-proj.git
cd vllm-proj/deployment

# Create environment file
cp .env.example .env

# Edit configuration (optional)
vi .env  # Set MODEL_NAME, GPU_MEMORY_UTILIZATION, etc.
```

### 2. Build & Run

```bash
# Build Docker image
docker-compose build

# Start service
docker-compose up -d

# Check logs
docker-compose logs -f vllm-server
```

### 3. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain vLLM in simple terms",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Streaming generation
curl -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about AI",
    "max_tokens": 100
  }'

# Check metrics
curl http://localhost:8000/metrics
```

---

## 🔧 Configuration

### Environment Variables

Edit `deployment/.env` or set environment variables:

```bash
# Model Configuration
MODEL_NAME=microsoft/phi-2              # HuggingFace model
GPU_MEMORY_UTILIZATION=0.9             # 0.0 to 1.0
MAX_MODEL_LEN=2048                     # Maximum sequence length
TRUST_REMOTE_CODE=True                 # Required for Phi-2

# Server Configuration
HOST=0.0.0.0                           # Listen address
PORT=8000                              # Listen port
WORKERS=1                              # Number of workers
LOG_LEVEL=info                         # debug, info, warning, error

# Cost Budgeting (Optional)
DAILY_BUDGET=50.00                     # Daily cost limit
INPUT_COST_PER_1M=0.10                # Cost per 1M input tokens
OUTPUT_COST_PER_1M=0.10               # Cost per 1M output tokens
GPU_COST_PER_HOUR=0.45                # GPU rental cost
```

### Supported Models

Test with smaller models first:

```bash
# Tiny (2-3B params, 6-8GB VRAM)
MODEL_NAME=microsoft/phi-2

# Small (7B params, 14-16GB VRAM)
MODEL_NAME=mistralai/Mistral-7B-v0.1

# Medium (13B params, 26-30GB VRAM - needs A100)
MODEL_NAME=meta-llama/Llama-2-13b-hf
```

---

## 📊 API Endpoints

### POST /generate
Generate text completion.

**Request:**
```json
{
  "prompt": "Your prompt here",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**Response:**
```json
{
  "text": "Generated text...",
  "tokens": 87,
  "finish_reason": "stop",
  "model": "microsoft/phi-2"
}
```

### POST /generate/stream
Streaming text generation (Server-Sent Events).

**Request:** Same as /generate

**Response:** Stream of SSE events:
```
data: {"text": "chunk1", "token_id": 123}
data: {"text": "chunk2", "token_id": 456}
data: [DONE]
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "microsoft/phi-2",
  "gpu_memory_used_gb": 19.3
}
```

### GET /metrics
Prometheus-style metrics.

**Response:**
```json
{
  "total_requests": 1234,
  "successful_requests": 1200,
  "failed_requests": 34,
  "avg_latency_ms": 125.5
}
```

---

## 🐳 Docker Commands

### Build

```bash
# Build image
docker-compose build

# Or with Docker directly
docker build -t vllm-enterprise:latest -f deployment/Dockerfile .
```

### Run

```bash
# Start in background
docker-compose up -d

# Start in foreground (see logs)
docker-compose up

# Stop
docker-compose down

# Restart
docker-compose restart
```

### Logs & Debugging

```bash
# View logs
docker-compose logs -f vllm-server

# Last 100 lines
docker-compose logs --tail=100 vllm-server

# Execute command in container
docker-compose exec vllm-server bash

# Check GPU inside container
docker-compose exec vllm-server nvidia-smi

# Check Python environment
docker-compose exec vllm-server python3 -c "import vllm; print(vllm.__version__)"
```

### Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes model cache)
docker-compose down -v

# Remove image
docker rmi vllm-enterprise:latest

# Full cleanup
docker-compose down -v
docker system prune -a
```

---

## 🎯 Production Deployment

### GCP Compute Engine

```bash
# Create VM with GPU
gcloud compute instances create vllm-production \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-l4,count=1 \
  --image-family=common-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE

# SSH to VM
gcloud compute ssh vllm-production

# Clone and deploy
git clone <your-repo>
cd vllm-proj/deployment
cp .env.example .env
docker-compose up -d
```

### AWS EC2

```bash
# Launch g5.xlarge or g5.2xlarge instance
# With Deep Learning AMI (Ubuntu)

# SSH to instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Install Docker + NVIDIA toolkit
# (see Prerequisites above)

# Deploy
git clone <your-repo>
cd vllm-proj/deployment
cp .env.example .env
docker-compose up -d
```

### Azure

```bash
# Create NC-series VM with GPU
az vm create \
  --resource-group vllm-rg \
  --name vllm-production \
  --size Standard_NC6s_v3 \
  --image Canonical:UbuntuServer:18.04-LTS:latest

# Deploy same as above
```

---

## 📈 Monitoring

### Basic Health Monitoring

```bash
# Check health every 30 seconds
watch -n 30 'curl -s http://localhost:8000/health | jq'

# Monitor metrics
watch -n 5 'curl -s http://localhost:8000/metrics | jq'

# GPU monitoring
watch -n 1 nvidia-smi
```

### Log Monitoring

```bash
# Follow logs
docker-compose logs -f vllm-server

# Filter errors
docker-compose logs vllm-server | grep ERROR

# Save logs to file
docker-compose logs vllm-server > vllm.log
```

### Resource Usage

```bash
# Container stats
docker stats vllm-production

# Detailed GPU usage
nvidia-smi dmon -s pucvmet -d 1
```

---

## 🚨 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs vllm-server

# Common issues:
# - GPU not available: Check nvidia-smi
# - OOM: Reduce GPU_MEMORY_UTILIZATION
# - Model download failed: Check network/disk space
```

### CUDA Out of Memory

```bash
# Reduce GPU memory utilization
GPU_MEMORY_UTILIZATION=0.7 docker-compose up

# Or use smaller model
MODEL_NAME=microsoft/phi-2 docker-compose up
```

### Model Download Slow/Failed

```bash
# Pre-download model
docker-compose exec vllm-server python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'microsoft/phi-2'
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
"

# Or mount pre-downloaded models
# Edit docker-compose.yml:
# volumes:
#   - /path/to/models:/app/models
```

### Port Already in Use

```bash
# Change port in .env
PORT=8001

# Or in docker-compose.yml:
# ports:
#   - "8001:8000"
```

### Health Check Failing

```bash
# Check if service is running
curl http://localhost:8000/health

# Check if model is loaded
docker-compose logs vllm-server | grep "Loading model"

# Increase health check timeout in docker-compose.yml:
# healthcheck:
#   start_period: 120s  # Give more time for model loading
```

---

## ⚡ Performance Tuning

### GPU Memory Optimization

```bash
# More memory for KV cache (better throughput)
GPU_MEMORY_UTILIZATION=0.95

# Less memory usage (more stability)
GPU_MEMORY_UTILIZATION=0.8
```

### Batch Size Tuning

vLLM automatically optimizes batching. Monitor with:

```bash
# Check throughput
curl http://localhost:8000/metrics | jq '.tokens_per_second'

# Optimal batch size depends on:
# - Model size
# - Sequence length
# - GPU VRAM
```

### Multi-GPU (Future)

```bash
# Edit .env:
TENSOR_PARALLEL_SIZE=2  # For 2 GPUs

# Update docker-compose.yml:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 2  # Number of GPUs
```

---

## 🔒 Security

### API Key Authentication (Optional)

```bash
# Add to .env:
API_KEY=your-secret-key

# Modify src/api_server.py to check API key
# (See security examples in code)
```

### CORS Configuration

```bash
# Add to .env:
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### HTTPS/TLS

Use reverse proxy (nginx, Caddy, Traefik):

```nginx
# nginx example
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 📚 Next Steps

After basic deployment:

1. ✅ **Load Testing** - Use `benchmarks/01_throughput_comparison.py`
2. ✅ **Monitor Metrics** - Track latency, throughput, costs
3. ⏳ **Add Prometheus/Grafana** - For production monitoring (Blog 3)
4. ⏳ **Implement Auto-scaling** - Based on request volume (Blog 3)
5. ⏳ **Add Quantization** - AWQ/GPTQ for larger models (Blog 2)

---

## 💡 Tips

- **Start small:** Test with phi-2 before deploying larger models
- **Monitor GPU:** Keep GPU utilization >80% for cost efficiency
- **Pre-download models:** Avoid cold-start delays in production
- **Use spot instances:** Save 60-80% on cloud GPU costs
- **Enable logging:** Essential for debugging production issues
- **Set resource limits:** Prevent OOM crashes
- **Test health checks:** Ensure monitoring catches failures

---

## 📞 Support

Issues? Check:
1. Docker logs: `docker-compose logs`
2. GPU status: `nvidia-smi`
3. Health endpoint: `curl http://localhost:8000/health`
4. [GitHub Issues](https://github.com/yourusername/vllm-proj/issues)
