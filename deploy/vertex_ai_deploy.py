#!/usr/bin/env python3
"""
Production deployment script for vLLM on Vertex AI
Complete automation for deploying vLLM models to Google Cloud Vertex AI
"""

import os
import json
import time
import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging

from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import artifactregistry
from google.cloud.exceptions import NotFound
import docker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DeploymentConfig:
    """Configuration for Vertex AI deployment"""
    project_id: str
    region: str
    model_name: str
    model_path: str
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    min_replicas: int = 1
    max_replicas: int = 3
    container_image_uri: Optional[str] = None
    service_account: Optional[str] = None
    enable_autoscaling: bool = True
    spot_instances: bool = True  # Use spot for cost savings


class VertexAIDeployer:
    """Handles deployment of vLLM models to Vertex AI"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.storage_client = storage.Client(project=config.project_id)

        # Initialize Vertex AI
        aiplatform.init(
            project=config.project_id,
            location=config.region
        )

        # Docker client for building images
        self.docker_client = docker.from_env()

    def create_dockerfile(self, model_id: str) -> str:
        """Create Dockerfile for vLLM container"""
        dockerfile_content = f"""
# vLLM Production Container for Vertex AI
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install vLLM
RUN pip3 install vllm==0.5.4

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Download or copy model
ENV MODEL_ID={model_id}
ENV MODEL_PATH=/models

# Copy model files if local, otherwise download at runtime
COPY models/ $MODEL_PATH/

# Vertex AI environment variables
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_MODE=PREDICTION

# vLLM configuration
ENV TENSOR_PARALLEL_SIZE=1
ENV MAX_MODEL_LEN=2048
ENV GPU_MEMORY_UTILIZATION=0.95

# Copy serving script
COPY deploy/serve_vertex.py /app/serve.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start vLLM server
CMD ["python3", "/app/serve.py"]
"""
        return dockerfile_content

    def create_serving_script(self) -> str:
        """Create the serving script for Vertex AI"""
        return '''#!/usr/bin/env python3
"""
vLLM Serving Script for Vertex AI
Handles prediction requests and health checks
"""

import os
import json
import time
import logging
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter("vllm_requests_total", "Total requests")
request_latency = Histogram("vllm_request_latency_seconds", "Request latency")
active_requests = Gauge("vllm_active_requests", "Active requests")

# Global model instance
llm_engine = None


class PredictionRequest(BaseModel):
    """Vertex AI prediction request format"""
    instances: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = {}


class PredictionResponse(BaseModel):
    """Vertex AI prediction response format"""
    predictions: List[Dict[str, Any]]
    model_version: str = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global llm_engine

    # Startup
    logger.info("Starting vLLM engine...")
    model_path = os.getenv("MODEL_PATH", "/models")
    model_id = os.getenv("MODEL_ID", "microsoft/phi-2")

    # Use local model if available, otherwise download
    if os.path.exists(model_path):
        model_to_load = model_path
    else:
        model_to_load = model_id

    llm_engine = LLM(
        model=model_to_load,
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "2048")),
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        trust_remote_code=True
    )
    logger.info(f"vLLM engine started with model: {model_to_load}")

    yield

    # Shutdown
    logger.info("Shutting down vLLM engine...")
    del llm_engine


app = FastAPI(
    title="vLLM Vertex AI Server",
    version="1.0.0",
    lifespan=lifespan
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health_check():
    """Health check endpoint for Vertex AI"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": llm_engine is not None,
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint for Vertex AI"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with active_requests.track_inprogress():
        start_time = time.time()

        try:
            # Extract prompts from instances
            prompts = []
            for instance in request.instances:
                if "prompt" in instance:
                    prompts.append(instance["prompt"])
                elif "text" in instance:
                    prompts.append(instance["text"])
                else:
                    prompts.append(str(instance))

            # Get sampling parameters
            params = request.parameters or {}
            sampling_params = SamplingParams(
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.95),
                max_tokens=params.get("max_tokens", 256),
                top_k=params.get("top_k", -1)
            )

            # Generate responses
            outputs = llm_engine.generate(prompts, sampling_params)

            # Format predictions
            predictions = []
            for output in outputs:
                predictions.append({
                    "text": output.outputs[0].text,
                    "tokens": len(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason
                })

            # Record metrics
            request_count.inc()
            request_latency.observe(time.time() - start_time)

            return PredictionResponse(predictions=predictions)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "vLLM Vertex AI",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Main prediction endpoint",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics"
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("AIP_HTTP_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
'''

    def build_container_image(self, image_name: str) -> str:
        """Build Docker container for deployment"""
        console.print("[bold blue]Building container image...[/bold blue]")

        # Create build directory
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)

        # Write Dockerfile
        dockerfile_path = build_dir / "Dockerfile"
        dockerfile_path.write_text(
            self.create_dockerfile(self.config.model_name)
        )

        # Write serving script
        serve_script_path = build_dir / "serve_vertex.py"
        serve_script_path.write_text(self.create_serving_script())

        # Copy necessary files
        (build_dir / "requirements.txt").write_text(
            Path("requirements.txt").read_text()
        )

        # Build image
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building Docker image...", total=None)

            image, logs = self.docker_client.images.build(
                path=str(build_dir),
                tag=image_name,
                rm=True,
                pull=True
            )

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Image built: {image_name}")
        return image_name

    def push_to_artifact_registry(self, image_name: str) -> str:
        """Push container to Google Artifact Registry"""
        console.print("[bold blue]Pushing to Artifact Registry...[/bold blue]")

        # Create repository if doesn't exist
        ar_client = artifactregistry.ArtifactRegistryClient()
        repository_name = f"projects/{self.config.project_id}/locations/{self.config.region}/repositories/vllm-models"

        try:
            ar_client.get_repository(name=repository_name)
        except NotFound:
            console.print("Creating Artifact Registry repository...")
            # Create repository
            operation = ar_client.create_repository(
                parent=f"projects/{self.config.project_id}/locations/{self.config.region}",
                repository_id="vllm-models",
                repository=artifactregistry.Repository(
                    format_=artifactregistry.Repository.Format.DOCKER
                )
            )
            operation.result()

        # Tag and push image
        registry_url = f"{self.config.region}-docker.pkg.dev/{self.config.project_id}/vllm-models/{image_name}"

        self.docker_client.images.get(image_name).tag(registry_url)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Pushing image...", total=None)
            self.docker_client.images.push(registry_url)
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Image pushed: {registry_url}")
        return registry_url

    def deploy_model(self) -> aiplatform.Endpoint:
        """Deploy model to Vertex AI"""
        console.print("[bold blue]Deploying to Vertex AI...[/bold blue]")

        # Build and push container if not provided
        if not self.config.container_image_uri:
            image_name = f"vllm-{self.config.model_name.replace('/', '-')}"
            local_image = self.build_container_image(image_name)
            self.config.container_image_uri = self.push_to_artifact_registry(local_image)

        # Upload model to Vertex AI
        console.print("Uploading model to Vertex AI...")
        model = aiplatform.Model.upload(
            display_name=f"vllm-{self.config.model_name}",
            serving_container_image_uri=self.config.container_image_uri,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            serving_container_environment_variables={
                "MODEL_ID": self.config.model_name,
                "TENSOR_PARALLEL_SIZE": str(self.config.accelerator_count),
                "MAX_MODEL_LEN": "2048",
                "GPU_MEMORY_UTILIZATION": "0.95"
            }
        )
        console.print(f"[green]✓[/green] Model uploaded: {model.display_name}")

        # Create endpoint
        console.print("Creating endpoint...")
        endpoint = aiplatform.Endpoint.create(
            display_name=f"vllm-{self.config.model_name}-endpoint",
            description=f"vLLM endpoint for {self.config.model_name}"
        )
        console.print(f"[green]✓[/green] Endpoint created: {endpoint.display_name}")

        # Deploy model to endpoint
        console.print("Deploying model to endpoint...")

        # Machine specs based on spot instance preference
        if self.config.spot_instances:
            machine_spec = {
                "machine_type": self.config.machine_type,
                "accelerator_type": self.config.accelerator_type,
                "accelerator_count": self.config.accelerator_count,
            }
            # Note: Spot instances configuration may vary by region
        else:
            machine_spec = {
                "machine_type": self.config.machine_type,
                "accelerator_type": self.config.accelerator_type,
                "accelerator_count": self.config.accelerator_count,
            }

        deployed_model = endpoint.deploy(
            model=model,
            deployed_model_display_name=f"vllm-{self.config.model_name}-deployed",
            machine_type=self.config.machine_type,
            accelerator_type=self.config.accelerator_type,
            accelerator_count=self.config.accelerator_count,
            min_replica_count=self.config.min_replicas,
            max_replica_count=self.config.max_replicas,
            traffic_percentage=100,
            enable_access_logging=True,
            service_account=self.config.service_account
        )

        console.print(f"[green]✓[/green] Model deployed successfully!")

        # Display deployment information
        self._display_deployment_info(endpoint)

        return endpoint

    def _display_deployment_info(self, endpoint: aiplatform.Endpoint):
        """Display deployment information in a table"""
        table = Table(title="Deployment Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Endpoint Name", endpoint.display_name)
        table.add_row("Endpoint ID", endpoint.name)
        table.add_row("Region", self.config.region)
        table.add_row("Model", self.config.model_name)
        table.add_row("Machine Type", self.config.machine_type)
        table.add_row("GPU Type", self.config.accelerator_type)
        table.add_row("GPU Count", str(self.config.accelerator_count))
        table.add_row("Min Replicas", str(self.config.min_replicas))
        table.add_row("Max Replicas", str(self.config.max_replicas))
        table.add_row("Spot Instances", "Yes" if self.config.spot_instances else "No")

        # Estimate costs
        hourly_cost = self._estimate_hourly_cost()
        table.add_row("Estimated Cost/Hour", f"${hourly_cost:.2f}")
        table.add_row("Estimated Cost/Month", f"${hourly_cost * 730:.2f}")

        console.print(table)

    def _estimate_hourly_cost(self) -> float:
        """Estimate hourly cost based on configuration"""
        # Simplified cost estimation (actual costs may vary)
        gpu_costs = {
            "NVIDIA_TESLA_T4": 0.35 if not self.config.spot_instances else 0.11,
            "NVIDIA_TESLA_V100": 2.48 if not self.config.spot_instances else 0.74,
            "NVIDIA_TESLA_A100": 3.67 if not self.config.spot_instances else 1.10,
            "NVIDIA_L4": 0.65 if not self.config.spot_instances else 0.22,
        }

        machine_costs = {
            "n1-standard-4": 0.19,
            "n1-standard-8": 0.38,
            "n1-standard-16": 0.76,
            "a2-highgpu-1g": 0.85,
        }

        gpu_cost = gpu_costs.get(self.config.accelerator_type, 0.35)
        machine_cost = machine_costs.get(self.config.machine_type, 0.38)

        total_per_instance = (gpu_cost * self.config.accelerator_count) + machine_cost
        return total_per_instance * self.config.min_replicas


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy vLLM to Vertex AI")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--model-name", required=True, help="Model name or path")
    parser.add_argument("--model-path", required=True, help="Path to model files")
    parser.add_argument("--machine-type", default="n1-standard-8", help="Machine type")
    parser.add_argument("--gpu-type", default="NVIDIA_TESLA_T4", help="GPU type")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--min-replicas", type=int, default=1, help="Minimum replicas")
    parser.add_argument("--max-replicas", type=int, default=3, help="Maximum replicas")
    parser.add_argument("--use-spot", action="store_true", help="Use spot instances")
    parser.add_argument("--service-account", help="Service account email")

    args = parser.parse_args()

    # Create configuration
    config = DeploymentConfig(
        project_id=args.project_id,
        region=args.region,
        model_name=args.model_name,
        model_path=args.model_path,
        machine_type=args.machine_type,
        accelerator_type=args.gpu_type,
        accelerator_count=args.gpu_count,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        spot_instances=args.use_spot,
        service_account=args.service_account
    )

    # Deploy
    deployer = VertexAIDeployer(config)

    try:
        endpoint = deployer.deploy_model()
        console.print("\n[bold green]Deployment completed successfully![/bold green]")
        console.print(f"Endpoint URL: {endpoint.resource_name}")

        # Save deployment information
        deployment_info = {
            "endpoint_name": endpoint.display_name,
            "endpoint_id": endpoint.name,
            "region": config.region,
            "model": config.model_name,
            "timestamp": time.time()
        }

        with open("deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)

        console.print("\nDeployment info saved to deployment_info.json")

    except Exception as e:
        console.print(f"[bold red]Deployment failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()