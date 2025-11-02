# vLLM Tutorial: LLM Serving for Enterprise

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-latest-green.svg)](https://github.com/vllm-project/vllm)

**An implementation of vLLM for high-performance, cost-effective LLM serving in enterprise applications.**

Reduce your LLM infrastructure costs while achieving throughput improvement with example code, benchmarks, and deployment configurations.

---

## 📊 Real Performance Metrics

All metrics from **GCP L4 GPU** running **Phi-2** model:

### Throughput & Latency
```
Batch Size 1:   41.5 tokens/sec,  987ms latency
Batch Size 32:  934.4 tokens/sec,  99ms latency
Improvement:    22.5x throughput, 9.9x latency reduction
```

### Memory Efficiency (PagedAttention)
```
Base model:     19.3 GB
Batch 8:        19.38 GB (+0.5% overhead)
PagedAttention: Near-zero KV cache growth
```

### Cost Optimization
```
Configuration                   Cost/Month    Savings vs Baseline
─────────────────────────────────────────────────────────────────
Baseline (Phi-2, FP16)          $324          -
+ FP8 Quantization              $162          50%
+ AWQ Quantization (W4A16)      $87           73%
+ Prefix Caching (RAG)          $65           80%
+ Intelligent Routing           $45           86%

vs OpenAI GPT-4:                $666          93% cheaper with all optimizations
```

---

## 🚀 Quick Start (2 Minutes)

### Prerequisites
- NVIDIA GPU with 16GB+ VRAM (L4, T4, A10, or better)
- CUDA 11.8+
- Python 3.10+
- Docker & Docker Compose (for deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/bhatti/vllm-tutorial.git
cd vllm-tutorial

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run End-to-End Demo

```bash
# This demonstrates the stack for :
# - Intelligent routing between 3 models
# - Budget tracking ($10/day limit)
# - Prefix caching (80% cost savings)
# - Full observability metrics

./scripts/run_end_to_end_demo.sh
```

**Expected output:**
```
🎉 All validation checks passed!
  ✅ Budget not exceeded
  ✅ At least 80% success rate
  ✅ Average latency < 2000ms
  ✅ Cache hit rate > 0%
  ✅ Multiple models used

Total cost: $0.002543 (Budget remaining: $9.997457)
Cache hit rate: 72.7%
Average latency: 487.3ms
```

---

## 💡 Key Features

### 1. Intelligent Routing
Route requests to appropriate models based on complexity and cost:

```python
from src.intelligent_router import IntelligentRouter

router = IntelligentRouter(daily_budget_usd=100.0)

# Simple query → Phi-2 (cheapest)
response = router.route_request("What is EBITDA?")

# Complex analysis → Llama-3-8B (most capable)
response = router.route_request("Analyze Microsoft's risk factors and impact on earnings")
```

**Savings:** 30% cost reduction through smart routing

### 2. Prefix Caching
Massive cost savings for RAG and template-based use cases:

```python
from vllm import LLM

llm = LLM(
    model="microsoft/phi-2",
    enable_prefix_caching=True,  # Enable caching!
)

# System prompt cached after first request
system_prompt = "You are a financial analyst..."

# Each query reuses cached system prompt → 80% faster + cheaper
for query in user_queries:
    response = llm.generate([system_prompt + query])
```

**Savings:** 50-80% cost reduction for RAG use cases

### 3. Quantization
Reduce memory and cost with minimal quality loss:

```python
from vllm import LLM

# FP8 quantization - 2x compression
llm = LLM(
    model="microsoft/phi-2",
    quantization="fp8",
    gpu_memory_utilization=0.9,
)

# AWQ quantization - 3.7x compression
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",
)
```

**Savings:** 50-73% cost reduction, 2-3.7x memory reduction

### 4. Observability

**Prometheus + Grafana:**
- Request rate, success/error rates
- Latency (P50, P95, P99)
- TTFT (Time to First Token) and ITL (Inter-Token Latency)
- GPU memory usage
- Cost tracking

**LLM-Specific Observability:**
- **Langfuse:** Cost tracking, prompt versioning, user analytics
- **Arize Phoenix:** Embedding analysis, drift detection, open-source alternative

```python
from examples.llm_observability import track_with_langfuse

# Track all LLM interactions for compliance
trace = track_with_langfuse(
    prompt="Analyze earnings report",
    model="phi-2",
    user_id="user_123"
)
```

### 5. Error Handling

```python
from examples.advanced_error_handling import (
    retry_with_backoff,
    CircuitBreaker,
    RateLimiter
)

# Retry with exponential backoff
@retry_with_backoff(max_retries=3)
def generate_text(prompt):
    return llm.generate(prompt)

# Circuit breaker for fault tolerance
circuit_breaker = CircuitBreaker(failure_threshold=5)
response = circuit_breaker.call(llm.generate, prompt)

# Rate limiting
rate_limiter = RateLimiter(max_requests=100, time_window=1.0)
if rate_limiter.acquire():
    response = llm.generate(prompt)
```

---

## 🐳 Deployment

### Single Instance (2 minutes)

```bash
cd deployment
cp .env.example .env

# Edit .env to configure:
# - MODEL_NAME=microsoft/phi-2
# - QUANTIZATION=fp8 (already default!)
# - GPU_MEMORY_UTILIZATION=0.9

docker-compose up -d

# Test
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is EBITDA?", "max_tokens": 100}'
```

### With Monitoring Stack (5 minutes)

```bash
# Start vLLM + Prometheus + Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
open http://localhost:9090   # Prometheus
open http://localhost:3000   # Grafana (admin/admin)

# Import pre-built dashboard: monitoring/grafana/dashboards/vllm_overview.json
```

**Grafana Dashboard includes:**
- Request rate and success/error rates
- Latency percentiles (P50, P95, P99)
- Token generation metrics
- GPU memory usage
- Cost tracking
- Model distribution

### With Langfuse (LLM Observability)

```bash
# Start with Langfuse for compliance tracking
docker-compose -f docker-compose.monitoring.yml --profile langfuse up -d

# Access Langfuse
open http://localhost:3001

# Configure in your code
export LANGFUSE_PUBLIC_KEY=your_key
export LANGFUSE_SECRET_KEY=your_secret

python examples/05_llm_observability.py
```

### High Availability Setup (Reference)

```bash
# Requires 2+ GPUs
# Shown as reference implementation

docker-compose -f docker-compose.ha.yml up -d

# Access via Nginx load balancer
curl http://localhost/generate

# Features:
# - 2+ vLLM instances
# - Nginx load balancing (least_conn)
# - Health checks
# - Automatic failover
```

**Note:** HA setup tested as reference on single GPU. Requires multi-GPU infrastructure for production use.

---

## 📈 Running Benchmarks

All benchmarks run on **GCP L4 GPU** with real models:

```bash
# Activate venv
source venv/bin/activate

# Run all benchmarks
cd benchmarks
bash scripts/run_all_benchmarks.sh

# Or run individually
python 01_throughput_comparison.py   # Batch processing speedup
python 02_memory_efficiency.py       # PagedAttention efficiency
python 03_cost_analysis.py           # Cost vs OpenAI
python 04_quantization_comparison.py # Quantization impact

# Results saved to benchmarks/results/
ls -la results/
# throughput_results.csv
# memory_results.csv
# cost_analysis.csv
# quantization_results.csv
```

---

## 🧪 Running Examples

```bash
# Basic examples
python examples/01_basic_vllm.py                    # Introduction
python examples/02_intelligent_routing_production.py # Routing + budgets
python examples/03_observability_quick.py           # Metrics

# Advanced optimizations
python examples/04_quantized_deployment.py          # FP8, AWQ
python examples/06_prefix_caching.py                # 80% savings!

# Production patterns
python examples/07_advanced_error_handling.py       # Resilience
python examples/08_long_context_handling.py         # >2K tokens

# Observability (requires Langfuse/Phoenix)
python examples/05_llm_observability.py
```

---

## 🧰 Running Tests

```bash
# API server tests (31 tests)
pytest tests/api/ -v

# End-to-end integration test
./run_end_to_end_demo.sh

# Or run directly
python tests/integration/test_end_to_end_fintech.py

# Results saved to test_results/
```

###End-to-End Integration Test Results

The E2E test validates the complete production stack with **11 sample queries**:

| Metric | Value |
|--------|-------|
| Success Rate | 100% (11/11) |
| Cache Hit Rate | 91% |
| Total Cost | $0.000100 |
| Prefix Caching Savings | $0.000401 (80%) |
| Models Used | 3 (Phi-2: 54.5%, Llama-3-8B: 27.3%, Mistral-7B: 18.2%) |

**What it validates:**
- ✅ Intelligent routing correctly classifies query complexity
- ✅ Budget tracking prevents cost overruns
- ✅ Prefix caching provides 80% cost reduction
- ✅ All observability metrics collected successfully
- ✅ Error handling gracefully manages failures
- ✅ Multi-model support works across 3 models

**See `E2E_TEST_RESULTS_ANALYSIS.md` for detailed analysis and blog-ready soundbites.**

---

## 📊 Cost Analysis: Real Numbers

Based on **10,000 requests/day** with **average 100 tokens/request**:

### Monthly Cost Breakdown

| Configuration | GPU | Cost/Month | vs Baseline | vs GPT-4 |
|--------------|-----|------------|-------------|----------|
| **OpenAI GPT-4** | N/A | $666 | +105% | - |
| **OpenAI GPT-3.5** | N/A | $15 | -95% | -98% |
| **vLLM Phi-2 (FP16)** | L4 | $324 | Baseline | -51% |
| **+ FP8 Quantization** | L4 | $162 | -50% | -76% |
| **+ AWQ Quantization** | L4 | $87 | -73% | -87% |
| **+ Prefix Caching** | L4 | $65 | -80% | -90% |
| **+ All Optimizations** | L4 | **$45** | **-86%** | **-93%** |

**All optimizations:**
- AWQ quantization (3.7x compression)
- Prefix caching (80% reduction for RAG)
- Intelligent routing (30% reduction)

### GPU Costs
- **GCP L4 GPU:** $0.45/hour = $324/month (730 hours)
- **AWS G5 (A10G):** $1.006/hour = $735/month
- **Azure NCasT4_v3:** $0.526/hour = $384/month

---

## 🎓 Learning Path

### 1. Start with Basics
- Read: `examples/01_basic_vllm.py`
- Run: Basic benchmark `python benchmarks/01_throughput_comparison.py`
- Understand: PagedAttention, continuous batching

### 2. Deploy Single Instance
- Follow: `deployment/DEPLOYMENT_GUIDE.md`
- Deploy: `docker-compose up -d`
- Test: API endpoints

### 3. Add Optimizations
- Implement: Quantization (`examples/04_quantized_deployment.py`)
- Enable: Prefix caching (`examples/06_prefix_caching.py`)
- Add: Intelligent routing (`examples/02_intelligent_routing_production.py`)

### 4. Production Monitoring
- Deploy: Monitoring stack (`docker-compose.monitoring.yml`)
- Configure: Alerts (`monitoring/prometheus/alert_rules.yml`)
- Dashboard: Grafana (`monitoring/grafana/dashboards/vllm_overview.json`)

### 5. Advanced Patterns
- Error handling: `examples/07_advanced_error_handling.py`
- Long context: `examples/08_long_context_handling.py`
- Observability: `examples/05_llm_observability.py`

### 6. Scale to Production
- High availability: `docker-compose.ha.yml`
- Load balancing: `deployment/nginx/load_balancer.conf`
- Run integration test: `./run_end_to_end_demo.sh`

---

## 🔧 Configuration Reference

### Environment Variables (.env)

```bash
# Model Configuration
MODEL_NAME=microsoft/phi-2              # Or mistralai/Mistral-7B-Instruct-v0.2
QUANTIZATION=fp8                        # none, fp8, awq, gptq
DTYPE=auto                              # auto, float16, bfloat16

# GPU Settings
GPU_MEMORY_UTILIZATION=0.9              # 0.0-1.0 (0.9 recommended)
TENSOR_PARALLEL_SIZE=1                  # Number of GPUs for tensor parallelism

# Context & Performance
MAX_MODEL_LEN=2048                      # Max context length
ENABLE_PREFIX_CACHING=True              # Enable prefix caching (recommended!)
MAX_NUM_SEQS=256                        # Max concurrent sequences

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Observability (optional)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=http://localhost:3001

# Budget (for intelligent routing)
DAILY_BUDGET_USD=100.0
MONTHLY_BUDGET_USD=3000.0
```

## 📚 Documentation

- **[DEPLOYMENT_GUIDE.md](deployment/DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[COMPREHENSIVE_BLOG_COMPLETE.md](COMPREHENSIVE_BLOG_COMPLETE.md)** - Feature summary and blog outline
- **[QUANTIZATION_ADDED.md](docs/QUANTIZATION_ADDED.md)** - Quantization guide
- **[VLLM_TALK_INSIGHTS.md](docs/VLLM_TALK_INSIGHTS.md)** - Research insights

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🎯 Quick Links

- [vLLM Documentation](https://docs.vllm.ai/)
- [Hugging Face Models](https://huggingface.co/models)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Langfuse](https://langfuse.com/)
- [Arize Phoenix](https://phoenix.arize.com/)

