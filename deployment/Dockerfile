FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install vLLM (separate for better caching)
RUN pip3 install --no-cache-dir vllm==0.5.4

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/

# Create directories
RUN mkdir -p /app/models /app/logs

# Environment variables with defaults
ENV MODEL_NAME=${MODEL_NAME:-microsoft/phi-2}
ENV MODEL_PATH=${MODEL_PATH:-/app/models}
ENV GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
ENV MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
ENV QUANTIZATION=${QUANTIZATION:-fp8}
ENV DTYPE=${DTYPE:-auto}
ENV HOST=${HOST:-0.0.0.0}
ENV PORT=${PORT:-8000}
ENV TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["python3", "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
