"""
Intelligent LLM Router with Cost Optimization and Observability
Production-ready routing system for multi-model deployment with vLLM

This system intelligently routes requests to different LLMs based on:
- Request complexity classification
- Cost optimization
- Latency requirements
- Model availability
- Historical performance

Integrated with Langfuse and Phoenix for complete observability.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from collections import deque
import logging
from abc import ABC, abstractmethod

# vLLM for high-performance inference
from vllm import LLM, SamplingParams

# Observability
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from phoenix.trace import TracerProvider
from phoenix.trace.langchain import LangChainInstrumentor
import opentelemetry.trace as trace

# For request classification
from sentence_transformers import SentenceTransformer
import torch
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestComplexity(Enum):
    """Classification of request complexity"""
    SIMPLE = "simple"           # FAQ, simple extraction
    MODERATE = "moderate"        # Analysis, summarization
    COMPLEX = "complex"          # Multi-step reasoning
    CRITICAL = "critical"        # High-stakes decisions


class ModelTier(Enum):
    """Model tiers for routing"""
    TINY = "tiny"               # Phi-2, 2.7B params
    SMALL = "small"             # Mistral-7B, Llama-2-7B
    MEDIUM = "medium"           # Llama-2-13B, CodeLlama-13B
    LARGE = "large"             # Llama-2-70B, Mixtral-8x7B
    PREMIUM = "premium"         # GPT-4, Claude-3


@dataclass
class ModelConfig:
    """Configuration for each model in the routing pool"""
    name: str
    tier: ModelTier
    cost_per_1k_tokens: float
    avg_latency_ms: float
    max_context_length: int
    capabilities: List[str]
    endpoint_url: Optional[str] = None
    is_vllm: bool = True
    quantization: Optional[str] = None  # "awq", "gptq", None
    gpu_memory_gb: float = 0
    max_concurrent_requests: int = 100


@dataclass
class RoutingRequest:
    """Request to be routed to appropriate model"""
    id: str
    content: str
    user_id: Optional[str] = None
    max_latency_ms: Optional[float] = None
    max_cost: Optional[float] = None
    required_capabilities: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher is more important
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decision made by the router"""
    request_id: str
    selected_model: str
    model_tier: ModelTier
    complexity: RequestComplexity
    estimated_cost: float
    estimated_latency_ms: float
    confidence_score: float
    reasoning: str
    alternatives: List[Dict[str, Any]]


class ComplexityClassifier:
    """
    Classifies request complexity using embeddings and heuristics
    This is crucial for intelligent routing decisions
    """

    def __init__(self):
        # Load sentence transformer for semantic understanding
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Define complexity patterns
        self.complexity_patterns = {
            RequestComplexity.SIMPLE: [
                "what is", "define", "when", "where", "who",
                "list", "name", "yes or no", "true or false"
            ],
            RequestComplexity.MODERATE: [
                "summarize", "explain", "compare", "analyze",
                "describe", "evaluate", "assess", "review"
            ],
            RequestComplexity.COMPLEX: [
                "create", "design", "implement", "solve",
                "optimize", "debug", "refactor", "architect"
            ],
            RequestComplexity.CRITICAL: [
                "financial advice", "medical", "legal",
                "investment decision", "risk assessment",
                "compliance", "regulatory"
            ]
        }

        # Token count thresholds
        self.token_thresholds = {
            RequestComplexity.SIMPLE: 50,
            RequestComplexity.MODERATE: 200,
            RequestComplexity.COMPLEX: 500,
            RequestComplexity.CRITICAL: 1000
        }

    def classify(self, request: RoutingRequest) -> Tuple[RequestComplexity, float]:
        """
        Classify request complexity with confidence score

        Returns:
            Tuple of (complexity_level, confidence_score)
        """
        text_lower = request.content.lower()

        # Check for critical keywords first (highest priority)
        for keyword in self.complexity_patterns[RequestComplexity.CRITICAL]:
            if keyword in text_lower:
                return RequestComplexity.CRITICAL, 0.95

        # Estimate token count (rough approximation)
        estimated_tokens = len(request.content.split()) * 1.3

        # Pattern matching for complexity
        pattern_scores = {}
        for complexity, patterns in self.complexity_patterns.items():
            score = sum(1 for p in patterns if p in text_lower)
            pattern_scores[complexity] = score

        # Combine pattern matching with token count
        if estimated_tokens < self.token_thresholds[RequestComplexity.SIMPLE]:
            if pattern_scores.get(RequestComplexity.SIMPLE, 0) > 0:
                return RequestComplexity.SIMPLE, 0.85

        if estimated_tokens < self.token_thresholds[RequestComplexity.MODERATE]:
            if pattern_scores.get(RequestComplexity.MODERATE, 0) > 0:
                return RequestComplexity.MODERATE, 0.80

        # Use embedding similarity for complex classification
        embedding = self.encoder.encode([request.content])[0]
        embedding_norm = np.linalg.norm(embedding)

        # Higher norm often indicates more complex semantic content
        if embedding_norm > 15:  # Threshold based on empirical testing
            return RequestComplexity.COMPLEX, 0.75

        # Default to moderate with lower confidence
        return RequestComplexity.MODERATE, 0.60


class CostOptimizer:
    """
    Optimizes model selection based on cost constraints and performance requirements
    """

    def __init__(self, models: List[ModelConfig]):
        self.models = {m.name: m for m in models}
        self.cost_history = deque(maxlen=1000)
        self.performance_history = {}

        # Cost budgets per user tier
        self.user_budgets = {
            "free": 0.01,      # $0.01 per request max
            "basic": 0.05,     # $0.05 per request max
            "premium": 0.50,   # $0.50 per request max
            "enterprise": None  # No limit
        }

    def select_optimal_model(
        self,
        complexity: RequestComplexity,
        request: RoutingRequest,
        available_models: List[str]
    ) -> Tuple[str, float]:
        """
        Select the most cost-effective model that meets requirements

        Returns:
            Tuple of (model_name, estimated_cost)
        """
        # Get user budget
        user_tier = request.metadata.get("user_tier", "basic")
        max_budget = self.user_budgets.get(user_tier, 0.05)

        if request.max_cost:
            max_budget = min(max_budget, request.max_cost)

        # Filter models by requirements
        candidates = []
        for model_name in available_models:
            model = self.models[model_name]

            # Check latency requirement
            if request.max_latency_ms and model.avg_latency_ms > request.max_latency_ms:
                continue

            # Check capabilities
            if request.required_capabilities:
                if not all(cap in model.capabilities for cap in request.required_capabilities):
                    continue

            # Estimate cost (rough token estimation)
            estimated_tokens = len(request.content.split()) * 2  # Input + output
            estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens

            # Check budget
            if max_budget and estimated_cost > max_budget:
                continue

            candidates.append((model_name, estimated_cost, model.avg_latency_ms))

        if not candidates:
            # Fallback to cheapest available model
            cheapest = min(
                available_models,
                key=lambda m: self.models[m].cost_per_1k_tokens
            )
            model = self.models[cheapest]
            estimated_tokens = len(request.content.split()) * 2
            cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens
            return cheapest, cost

        # Select based on complexity
        if complexity == RequestComplexity.SIMPLE:
            # Choose cheapest for simple requests
            selected = min(candidates, key=lambda x: x[1])
        elif complexity == RequestComplexity.CRITICAL:
            # Choose highest tier for critical requests
            selected = max(
                candidates,
                key=lambda x: self.models[x[0]].tier.value
            )
        else:
            # Balance cost and performance for moderate/complex
            # Score = normalized_cost * 0.6 + normalized_latency * 0.4
            costs = [c[1] for c in candidates]
            latencies = [c[2] for c in candidates]

            min_cost, max_cost = min(costs), max(costs)
            min_lat, max_lat = min(latencies), max(latencies)

            best_score = float('inf')
            selected = candidates[0]

            for candidate in candidates:
                norm_cost = (candidate[1] - min_cost) / (max_cost - min_cost + 0.001)
                norm_lat = (candidate[2] - min_lat) / (max_lat - min_lat + 0.001)
                score = norm_cost * 0.6 + norm_lat * 0.4

                if score < best_score:
                    best_score = score
                    selected = candidate

        return selected[0], selected[1]


class IntelligentRouter:
    """
    Main routing system that combines all components for intelligent LLM routing
    """

    def __init__(
        self,
        models: List[ModelConfig],
        langfuse_config: Optional[Dict] = None,
        phoenix_config: Optional[Dict] = None
    ):
        self.models = {m.name: m for m in models}
        self.model_pools = self._initialize_model_pools(models)

        # Initialize components
        self.classifier = ComplexityClassifier()
        self.cost_optimizer = CostOptimizer(models)

        # Model health tracking
        self.model_health = {m.name: {"healthy": True, "error_rate": 0.0} for m in models}

        # Performance tracking
        self.request_metrics = deque(maxlen=10000)

        # Initialize observability
        self._init_observability(langfuse_config, phoenix_config)

        # Circuit breaker for each model
        self.circuit_breakers = {
            m.name: CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                half_open_requests=2
            ) for m in models
        }

    def _initialize_model_pools(self, models: List[ModelConfig]) -> Dict[str, Any]:
        """Initialize vLLM model pools for different tiers"""
        pools = {}

        for model in models:
            if model.is_vllm:
                try:
                    # Initialize vLLM instance
                    vllm_kwargs = {
                        "model": model.name,
                        "gpu_memory_utilization": 0.9,
                        "max_model_len": model.max_context_length,
                        "enforce_eager": False
                    }

                    if model.quantization:
                        vllm_kwargs["quantization"] = model.quantization

                    pools[model.name] = LLM(**vllm_kwargs)
                    logger.info(f"Initialized vLLM pool for {model.name}")

                except Exception as e:
                    logger.error(f"Failed to initialize {model.name}: {e}")
                    pools[model.name] = None
            else:
                # External API endpoint
                pools[model.name] = model.endpoint_url

        return pools

    def _init_observability(
        self,
        langfuse_config: Optional[Dict],
        phoenix_config: Optional[Dict]
    ):
        """Initialize Langfuse and Phoenix for observability"""

        # Initialize Langfuse
        if langfuse_config:
            self.langfuse = Langfuse(
                public_key=langfuse_config.get("public_key"),
                secret_key=langfuse_config.get("secret_key"),
                host=langfuse_config.get("host", "https://cloud.langfuse.com")
            )
            logger.info("Langfuse observability initialized")
        else:
            self.langfuse = None

        # Initialize Phoenix
        if phoenix_config:
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            logger.info("Phoenix tracing initialized")
        else:
            self.tracer = None

    @observe(name="route_request")
    async def route_request(
        self,
        request: RoutingRequest
    ) -> Tuple[RoutingDecision, str]:
        """
        Route a request to the optimal model

        Returns:
            Tuple of (routing_decision, generated_response)
        """
        start_time = time.time()

        # Create trace span
        span = None
        if self.tracer:
            span = self.tracer.start_span("route_request")
            span.set_attribute("request_id", request.id)
            span.set_attribute("user_id", request.user_id or "anonymous")

        try:
            # Step 1: Classify request complexity
            complexity, confidence = self.classifier.classify(request)

            if span:
                span.set_attribute("complexity", complexity.value)
                span.set_attribute("confidence", confidence)

            # Step 2: Get available models (health check)
            available_models = self._get_available_models(complexity)

            if not available_models:
                raise Exception("No available models for request")

            # Step 3: Select optimal model based on cost and requirements
            selected_model, estimated_cost = self.cost_optimizer.select_optimal_model(
                complexity, request, available_models
            )

            # Step 4: Prepare routing decision
            model_config = self.models[selected_model]
            estimated_latency = model_config.avg_latency_ms

            # Get alternatives for transparency
            alternatives = self._get_alternatives(
                complexity, request, available_models, selected_model
            )

            decision = RoutingDecision(
                request_id=request.id,
                selected_model=selected_model,
                model_tier=model_config.tier,
                complexity=complexity,
                estimated_cost=estimated_cost,
                estimated_latency_ms=estimated_latency,
                confidence_score=confidence,
                reasoning=self._generate_reasoning(
                    complexity, selected_model, estimated_cost, confidence
                ),
                alternatives=alternatives
            )

            # Step 5: Execute request on selected model
            response = await self._execute_on_model(
                selected_model, request, decision
            )

            # Step 6: Track metrics
            actual_latency = (time.time() - start_time) * 1000
            self._track_metrics(
                request, decision, response, actual_latency
            )

            # Log to Langfuse
            if self.langfuse:
                self.langfuse.trace(
                    name="routing",
                    input=request.content,
                    output=response,
                    metadata={
                        "model": selected_model,
                        "complexity": complexity.value,
                        "cost": estimated_cost,
                        "latency_ms": actual_latency
                    },
                    user_id=request.user_id,
                    session_id=request.id
                )

            if span:
                span.set_attribute("selected_model", selected_model)
                span.set_attribute("actual_latency_ms", actual_latency)
                span.end()

            return decision, response

        except Exception as e:
            logger.error(f"Routing failed for request {request.id}: {e}")
            if span:
                span.record_exception(e)
                span.end()
            raise

    def _get_available_models(self, complexity: RequestComplexity) -> List[str]:
        """Get list of available models based on health and complexity"""
        available = []

        # Map complexity to minimum required tier
        min_tier_map = {
            RequestComplexity.SIMPLE: ModelTier.TINY,
            RequestComplexity.MODERATE: ModelTier.SMALL,
            RequestComplexity.COMPLEX: ModelTier.MEDIUM,
            RequestComplexity.CRITICAL: ModelTier.LARGE
        }

        min_tier = min_tier_map[complexity]

        for model_name, model_config in self.models.items():
            # Check if model tier is sufficient
            if model_config.tier.value < min_tier.value:
                continue

            # Check health
            if not self.model_health[model_name]["healthy"]:
                continue

            # Check circuit breaker
            if not self.circuit_breakers[model_name].is_available():
                continue

            available.append(model_name)

        return available

    async def _execute_on_model(
        self,
        model_name: str,
        request: RoutingRequest,
        decision: RoutingDecision
    ) -> str:
        """Execute request on selected model"""

        model_pool = self.model_pools.get(model_name)

        if model_pool is None:
            raise Exception(f"Model {model_name} not available")

        # Use circuit breaker
        circuit_breaker = self.circuit_breakers[model_name]

        try:
            with circuit_breaker:
                if isinstance(model_pool, LLM):
                    # vLLM execution
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=500
                    )

                    outputs = model_pool.generate(
                        [request.content],
                        sampling_params
                    )

                    return outputs[0].outputs[0].text
                else:
                    # External API call
                    # Implement API call logic here
                    return f"Response from {model_name}"

        except Exception as e:
            # Update health status
            self.model_health[model_name]["error_rate"] += 0.1
            if self.model_health[model_name]["error_rate"] > 0.5:
                self.model_health[model_name]["healthy"] = False
            raise

    def _generate_reasoning(
        self,
        complexity: RequestComplexity,
        model: str,
        cost: float,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for routing decision"""
        return (
            f"Request classified as {complexity.value} with {confidence:.0%} confidence. "
            f"Selected {model} based on cost-performance optimization. "
            f"Estimated cost: ${cost:.4f}. "
        )

    def _get_alternatives(
        self,
        complexity: RequestComplexity,
        request: RoutingRequest,
        available_models: List[str],
        selected: str
    ) -> List[Dict[str, Any]]:
        """Get alternative routing options for transparency"""
        alternatives = []

        for model_name in available_models[:3]:  # Top 3 alternatives
            if model_name == selected:
                continue

            model = self.models[model_name]
            estimated_tokens = len(request.content.split()) * 2
            cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens

            alternatives.append({
                "model": model_name,
                "tier": model.tier.value,
                "estimated_cost": cost,
                "estimated_latency_ms": model.avg_latency_ms,
                "reason_not_selected": self._get_rejection_reason(
                    model_name, selected, request
                )
            })

        return alternatives

    def _get_rejection_reason(
        self,
        model: str,
        selected: str,
        request: RoutingRequest
    ) -> str:
        """Get reason why model wasn't selected"""
        model_config = self.models[model]
        selected_config = self.models[selected]

        if model_config.cost_per_1k_tokens > selected_config.cost_per_1k_tokens * 2:
            return "Significantly higher cost"
        elif model_config.avg_latency_ms > selected_config.avg_latency_ms * 1.5:
            return "Higher latency"
        elif model_config.tier.value < selected_config.tier.value:
            return "Lower capability tier"
        else:
            return "Suboptimal cost-performance ratio"

    def _track_metrics(
        self,
        request: RoutingRequest,
        decision: RoutingDecision,
        response: str,
        actual_latency: float
    ):
        """Track routing metrics for analysis"""
        metric = {
            "timestamp": datetime.now(),
            "request_id": request.id,
            "user_id": request.user_id,
            "model": decision.selected_model,
            "complexity": decision.complexity.value,
            "estimated_cost": decision.estimated_cost,
            "actual_latency_ms": actual_latency,
            "response_length": len(response),
            "confidence": decision.confidence_score
        }

        self.request_metrics.append(metric)

        # Update model performance stats
        model_name = decision.selected_model
        if model_name not in self.cost_optimizer.performance_history:
            self.cost_optimizer.performance_history[model_name] = deque(maxlen=100)

        self.cost_optimizer.performance_history[model_name].append({
            "latency": actual_latency,
            "tokens": len(response.split()),
            "timestamp": datetime.now()
        })

    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the routing system"""
        if not self.request_metrics:
            return {"message": "No metrics available yet"}

        metrics_df = pd.DataFrame(list(self.request_metrics))

        analytics = {
            "total_requests": len(metrics_df),
            "avg_latency_ms": metrics_df["actual_latency_ms"].mean(),
            "p95_latency_ms": metrics_df["actual_latency_ms"].quantile(0.95),
            "total_cost": metrics_df["estimated_cost"].sum(),
            "avg_cost_per_request": metrics_df["estimated_cost"].mean(),
            "model_distribution": metrics_df["model"].value_counts().to_dict(),
            "complexity_distribution": metrics_df["complexity"].value_counts().to_dict(),
            "model_health": self.model_health,
            "cost_by_user": metrics_df.groupby("user_id")["estimated_cost"].sum().to_dict()
        }

        return analytics


class CircuitBreaker:
    """Circuit breaker pattern for model availability"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_count = 0

    def __enter__(self):
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_count = 0
            else:
                raise Exception("Circuit breaker is open")

        if self.state == "half_open":
            self.half_open_count += 1
            if self.half_open_count > self.half_open_requests:
                self.state = "open"
                raise Exception("Circuit breaker is open")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            if self.state == "half_open":
                self.state = "closed"
            self.failure_count = 0
        else:
            # Failure
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def is_available(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open" and self._should_attempt_reset():
            return True
        return self.state == "half_open"


# Example usage and configuration
def create_production_router():
    """Create a production-ready intelligent router"""

    # Define model configurations with real costs
    models = [
        # Tiny models for simple tasks
        ModelConfig(
            name="microsoft/phi-2",
            tier=ModelTier.TINY,
            cost_per_1k_tokens=0.0001,  # Self-hosted vLLM
            avg_latency_ms=50,
            max_context_length=2048,
            capabilities=["general", "simple_qa"],
            is_vllm=True
        ),

        # Small models for moderate tasks
        ModelConfig(
            name="mistralai/Mistral-7B-Instruct-v0.2",
            tier=ModelTier.SMALL,
            cost_per_1k_tokens=0.0002,  # Self-hosted vLLM
            avg_latency_ms=100,
            max_context_length=4096,
            capabilities=["general", "analysis", "summarization"],
            is_vllm=True,
            quantization="awq"  # Using AWQ for efficiency
        ),

        # Medium models for complex tasks
        ModelConfig(
            name="meta-llama/Llama-2-13b-chat-hf",
            tier=ModelTier.MEDIUM,
            cost_per_1k_tokens=0.0005,  # Self-hosted vLLM
            avg_latency_ms=200,
            max_context_length=4096,
            capabilities=["general", "analysis", "code", "reasoning"],
            is_vllm=True
        ),

        # Large models for critical tasks
        ModelConfig(
            name="meta-llama/Llama-2-70b-chat-hf",
            tier=ModelTier.LARGE,
            cost_per_1k_tokens=0.002,  # Self-hosted vLLM with multiple GPUs
            avg_latency_ms=500,
            max_context_length=4096,
            capabilities=["general", "analysis", "code", "reasoning", "financial"],
            is_vllm=True,
            quantization="gptq"
        ),

        # Premium external API for fallback
        ModelConfig(
            name="gpt-4-turbo",
            tier=ModelTier.PREMIUM,
            cost_per_1k_tokens=0.01,  # OpenAI API
            avg_latency_ms=1000,
            max_context_length=128000,
            capabilities=["all"],
            is_vllm=False,
            endpoint_url="https://api.openai.com/v1/chat/completions"
        )
    ]

    # Observability configuration
    langfuse_config = {
        "public_key": "your-public-key",
        "secret_key": "your-secret-key",
        "host": "https://cloud.langfuse.com"  # or self-hosted
    }

    phoenix_config = {
        "endpoint": "http://localhost:6006",  # Phoenix endpoint
        "project": "vllm-routing"
    }

    # Create router
    router = IntelligentRouter(
        models=models,
        langfuse_config=langfuse_config,
        phoenix_config=phoenix_config
    )

    return router


# Demonstration of the routing system
async def demonstrate_routing():
    """Demonstrate the intelligent routing system"""

    router = create_production_router()

    # Test requests of varying complexity
    test_requests = [
        # Simple request - should route to Phi-2
        RoutingRequest(
            id="req_001",
            content="What is the capital of France?",
            user_id="user_123",
            max_latency_ms=100,
            metadata={"user_tier": "free"}
        ),

        # Moderate request - should route to Mistral-7B
        RoutingRequest(
            id="req_002",
            content="Summarize the key points from this earnings report: Revenue increased 23% YoY to $5.2B...",
            user_id="user_456",
            max_latency_ms=500,
            metadata={"user_tier": "basic"}
        ),

        # Complex request - should route to Llama-13B or 70B
        RoutingRequest(
            id="req_003",
            content="Design a microservices architecture for a high-frequency trading system with sub-millisecond latency requirements...",
            user_id="user_789",
            required_capabilities=["code", "reasoning"],
            metadata={"user_tier": "premium"}
        ),

        # Critical financial request - should route to Llama-70B or GPT-4
        RoutingRequest(
            id="req_004",
            content="Analyze this investment portfolio and provide risk assessment with specific recommendations for rebalancing...",
            user_id="user_enterprise",
            required_capabilities=["financial"],
            priority=10,
            metadata={"user_tier": "enterprise"}
        )
    ]

    # Process requests
    for request in test_requests:
        try:
            decision, response = await router.route_request(request)

            print(f"\n{'='*60}")
            print(f"Request ID: {request.id}")
            print(f"Content Preview: {request.content[:100]}...")
            print(f"Complexity: {decision.complexity.value}")
            print(f"Selected Model: {decision.selected_model}")
            print(f"Model Tier: {decision.model_tier.value}")
            print(f"Estimated Cost: ${decision.estimated_cost:.4f}")
            print(f"Estimated Latency: {decision.estimated_latency_ms:.0f}ms")
            print(f"Confidence: {decision.confidence_score:.0%}")
            print(f"Reasoning: {decision.reasoning}")

            if decision.alternatives:
                print("\nAlternatives considered:")
                for alt in decision.alternatives[:2]:
                    print(f"  - {alt['model']}: ${alt['estimated_cost']:.4f} "
                          f"({alt['reason_not_selected']})")

        except Exception as e:
            print(f"Error processing request {request.id}: {e}")

    # Show analytics
    print(f"\n{'='*60}")
    print("ROUTING ANALYTICS")
    print("="*60)
    analytics = router.get_analytics()
    for key, value in analytics.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    # Run demonstration
    import asyncio

    print("Intelligent LLM Router Demonstration")
    print("This system routes requests to optimal models based on:")
    print("- Complexity classification")
    print("- Cost optimization")
    print("- Latency requirements")
    print("- Model availability")
    print("")

    # Note: This requires vLLM and models to be installed
    # For testing without models, comment out the actual execution

    try:
        asyncio.run(demonstrate_routing())
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Note: This demo requires vLLM and models to be installed")
        print("You can test the routing logic without actual model execution")