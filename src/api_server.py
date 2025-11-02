#!/usr/bin/env python3
"""
Production FastAPI server for intelligent vLLM routing
Handles real-world Enterprise requests with optimal model selection
"""

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
import time
import uuid
import logging
from enum import Enum

# Import our routing system
from intelligent_router_simple import (
    IntelligentRouter,
    RoutingRequest,
    ModelConfig,
    ModelTier,
    RequestComplexity
)

# Import model loader for real inference
from model_loader import get_model_manager, ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API version
API_VERSION = "1.0.0"


# Global router instance, model manager, and startup time
router: Optional[IntelligentRouter] = None
model_manager: Optional[ModelManager] = None
startup_time = time.time()


# Dependency injection for model manager
def get_model_manager_instance() -> ModelManager:
    """Get the model manager instance - can be overridden in tests"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    global startup_time, model_manager
    startup_time = time.time()
    initialize_router()

    # Check if we're in test mode OR model_manager is already set
    import os
    if os.getenv("TESTING") != "true":
        # In production, initialize model_manager if not set
        if model_manager is None:
            model_manager = get_model_manager()
            logger.info("API server started successfully")
        else:
            logger.info("API server started - model_manager already initialized")
    else:
        # In test mode, model_manager should already be set by tests - don't touch it
        logger.info(f"API server started in test mode - model_manager: {type(model_manager)}")

    yield
    # Shutdown
    if model_manager:
        model_manager.cleanup_all()
    logger.info("API server shutting down")


# Create FastAPI app with lifespan
app = FastAPI(
    title="vLLM Intelligent Routing API",
    description="Production API for intelligent model routing with vLLM",
    version=API_VERSION,
    lifespan=lifespan
)


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for generation endpoint"""
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt")
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    max_tokens: Optional[int] = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    max_cost: Optional[float] = Field(None, description="Maximum cost in USD")
    timeout_ms: Optional[int] = Field(30000, description="Request timeout in milliseconds")
    force_model: Optional[str] = Field(None, description="Force specific model")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v or v.isspace():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()


class GenerateResponse(BaseModel):
    """Response model for generation endpoint"""
    request_id: str
    response: str
    model_used: str
    complexity: str
    latency_ms: float
    tokens_used: int
    cost_estimate: float
    metadata: Optional[Dict[str, Any]] = {}


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    tier: str
    max_complexity: str
    capabilities: List[str]
    status: str
    cost_per_1k_tokens: float
    avg_latency_ms: float


# Request metrics
request_metrics = {
    "total_requests": 0,
    "requests_per_model": {},
    "total_latency_ms": 0,
    "requests_per_complexity": {
        "SIMPLE": 0,
        "MODERATE": 0,
        "COMPLEX": 0,
        "CRITICAL": 0
    }
}


def initialize_router():
    """Initialize the intelligent router with model configurations"""
    global router

    # Define available models (mock for now, replace with actual models)
    model_configs = [
        ModelConfig(
            name="tiny-model",
            tier=ModelTier.TINY,
            model_path="facebook/opt-125m",
            max_complexity=RequestComplexity.SIMPLE,
            cost_per_1k_tokens=0.0001,
            avg_latency_ms=50,
            capabilities=["general", "support"]
        ),
        ModelConfig(
            name="small-model",
            tier=ModelTier.SMALL,
            model_path="facebook/opt-1.3b",
            max_complexity=RequestComplexity.MODERATE,
            cost_per_1k_tokens=0.001,
            avg_latency_ms=100,
            capabilities=["general", "analysis", "support"]
        ),
        ModelConfig(
            name="medium-model",
            tier=ModelTier.MEDIUM,
            model_path="facebook/opt-6.7b",
            max_complexity=RequestComplexity.COMPLEX,
            cost_per_1k_tokens=0.01,
            avg_latency_ms=300,
            capabilities=["general", "analysis", "financial"]
        ),
        ModelConfig(
            name="large-model",
            tier=ModelTier.LARGE,
            model_path="meta-llama/Llama-2-13b-hf",
            max_complexity=RequestComplexity.CRITICAL,
            cost_per_1k_tokens=0.1,
            avg_latency_ms=1000,
            capabilities=["general", "analysis", "financial", "trading", "risk"]
        )
    ]

    router = IntelligentRouter(model_configs)
    logger.info(f"Router initialized with {len(model_configs)} models")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": API_VERSION
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    # Initialize router if not already done (for testing)
    if router is None:
        initialize_router()

    return {
        "ready": router is not None,
        "models_loaded": len(router.model_configs) if router else 0,
        "router_initialized": router is not None
    }


@app.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {"alive": True}


@app.get("/models")
async def list_models():
    """List available models"""
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    models = []
    for config in router.model_configs:
        models.append(ModelInfo(
            name=config.name,
            tier=config.tier.value,
            max_complexity=config.max_complexity.name,
            capabilities=config.capabilities,
            status="healthy" if config.is_healthy else "unhealthy",
            cost_per_1k_tokens=config.cost_per_1k_tokens,
            avg_latency_ms=config.avg_latency_ms
        ))

    return {"models": models}


@app.get("/models/{model_name}")
async def get_model_details(model_name: str):
    """Get specific model details"""
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    for config in router.model_configs:
        if config.name == model_name:
            return ModelInfo(
                name=config.name,
                tier=config.tier.value,
                max_complexity=config.max_complexity.name,
                capabilities=config.capabilities,
                status="healthy" if config.is_healthy else "unhealthy",
                cost_per_1k_tokens=config.cost_per_1k_tokens,
                avg_latency_ms=config.avg_latency_ms
            )

    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    manager: ModelManager = Depends(get_model_manager_instance)
):
    """Main generation endpoint with intelligent routing"""
    global request_metrics

    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Create routing request
    routing_request = RoutingRequest(
        id=request_id,
        content=request.prompt,
        user_id=request.user_id,
        max_latency_ms=request.timeout_ms,
        max_cost=request.max_cost,
        metadata=request.metadata or {}
    )

    try:
        # Get routing decision
        decision = router.route(routing_request)

        # Use real model inference if available, otherwise mock
        if manager:
            # Real vLLM inference
            try:
                result = manager.generate(
                    model_name=decision.model_name,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens or 256,
                    temperature=request.temperature or 0.7,
                    stream=request.stream
                )
                generated_text = result["generated_text"]
                tokens_used = result["total_tokens"]
                inference_time = result["generation_time"]
            except Exception as e:
                logger.warning(f"Model inference failed, using mock: {str(e)}")
                # Fallback to mock
                generated_text = f"[Mock response from {decision.model_name}] This is a simulated response to: {request.prompt[:50]}..."
                tokens_used = len(request.prompt.split()) + len(generated_text.split())
                inference_time = 0.1
        else:
            # Mock generation
            generated_text = f"[Mock response from {decision.model_name}] This is a simulated response to: {request.prompt[:50]}..."
            tokens_used = len(request.prompt.split()) + len(generated_text.split())
            inference_time = 0.1

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        cost_estimate = (tokens_used / 1000) * router.models[decision.model_name].cost_per_1k_tokens

        # Update metrics
        request_metrics["total_requests"] += 1
        request_metrics["requests_per_model"][decision.model_name] = \
            request_metrics["requests_per_model"].get(decision.model_name, 0) + 1
        request_metrics["total_latency_ms"] += latency_ms
        request_metrics["requests_per_complexity"][decision.complexity.name] += 1

        return GenerateResponse(
            request_id=request_id,
            response=generated_text,
            model_used=decision.model_name,
            complexity=decision.complexity.name,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            metadata={
                "confidence_score": decision.confidence_score,
                "fallback": decision.fallback,
                "reasoning": decision.reasoning
            }
        )

    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = []

    # Total requests
    metrics.append(f"# HELP requests_total Total number of requests")
    metrics.append(f"# TYPE requests_total counter")
    metrics.append(f"requests_total {request_metrics['total_requests']}")

    # Requests per model
    metrics.append(f"# HELP requests_per_model Requests per model")
    metrics.append(f"# TYPE requests_per_model counter")
    for model, count in request_metrics["requests_per_model"].items():
        metrics.append(f'requests_per_model{{model="{model}"}} {count}')

    # Average latency
    avg_latency = (request_metrics["total_latency_ms"] / max(1, request_metrics["total_requests"]))
    metrics.append(f"# HELP average_latency_ms Average request latency")
    metrics.append(f"# TYPE average_latency_ms gauge")
    metrics.append(f"average_latency_ms {avg_latency:.2f}")

    return Response(content="\n".join(metrics), media_type="text/plain")


@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    global request_metrics

    avg_latency = 0
    if request_metrics["total_requests"] > 0:
        avg_latency = request_metrics["total_latency_ms"] / request_metrics["total_requests"]

    return {
        "total_requests": request_metrics["total_requests"],
        "requests_per_model": request_metrics["requests_per_model"],
        "average_latency_ms": avg_latency,
        "requests_per_complexity": request_metrics["requests_per_complexity"],
        "uptime_seconds": time.time() - startup_time
    }


@app.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    force_reload: bool = False,
    manager: ModelManager = Depends(get_model_manager_instance)
):
    """Load a specific model into memory"""
    try:
        model = manager.load_model(model_name, force_reload=force_reload)
        return {
            "status": "success",
            "model_name": model.name,
            "is_loaded": model.is_loaded,
            "load_time": model.load_time
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    manager: ModelManager = Depends(get_model_manager_instance)
):
    """Unload a specific model from memory"""
    manager.unload_model(model_name)
    return {
        "status": "success",
        "message": f"Model {model_name} unloaded"
    }


@app.get("/models/loaded")
async def get_loaded_models(manager: ModelManager = Depends(get_model_manager_instance)):
    """Get list of currently loaded models"""
    loaded = manager.get_loaded_models()
    models_info = []

    for model_name in loaded:
        stats = manager.get_model_stats(model_name)
        if stats:
            models_info.append(stats)

    return {
        "loaded_models": models_info,
        "total_loaded": len(loaded),
        "max_models": manager.max_models_in_memory
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
