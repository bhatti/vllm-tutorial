#!/usr/bin/env python3
"""
Simplified Intelligent Router for vLLM
Routes requests to appropriate models based on complexity and cost
Works without external observability dependencies
"""

import re
import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RequestComplexity(Enum):
    """Complexity levels for requests"""
    SIMPLE = 1      # Quick responses, basic Q&A
    MODERATE = 2    # Explanations, summaries, simple analysis
    COMPLEX = 3     # Deep analysis, multi-step reasoning
    CRITICAL = 4    # Production/urgent requests requiring highest quality


class ModelTier(Enum):
    """Model tiers for routing"""
    TINY = "tiny"               # Small models like OPT-125M
    SMALL = "small"             # Models like OPT-1.3B
    MEDIUM = "medium"           # Models like OPT-6.7B
    LARGE = "large"             # Models like Llama-13B
    PREMIUM = "premium"         # Large models like Llama-70B


@dataclass
class ModelConfig:
    """Configuration for a model endpoint"""
    name: str
    tier: ModelTier
    model_path: str
    max_complexity: RequestComplexity
    cost_per_1k_tokens: float
    avg_latency_ms: float = 100
    max_context_length: int = 2048
    capabilities: List[str] = None
    is_healthy: bool = True
    failure_count: int = 0
    last_failure: Optional[float] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ["general"]


@dataclass
class RoutingRequest:
    """Request to be routed"""
    id: str
    content: str
    user_id: Optional[str] = None
    max_latency_ms: Optional[float] = None
    max_cost: Optional[float] = None
    required_capabilities: List[str] = None
    priority: int = 5
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.required_capabilities is None:
            self.required_capabilities = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Decision made by the router"""
    request_id: str
    model_name: str
    model_path: str
    model_tier: ModelTier
    complexity: RequestComplexity
    estimated_tokens: int
    estimated_cost: float
    estimated_latency_ms: float
    confidence_score: float
    reasoning: str
    fallback: bool = False
    alternatives: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class ComplexityClassifier:
    """Classifies request complexity based on various factors"""

    def __init__(self):
        # Keywords indicating complexity levels
        self.critical_keywords = [
            "[PRODUCTION]", "[URGENT]", "[CRITICAL]", "[HIGH-PRIORITY]",
            "[RISK]", "[COMPLIANCE]", "[REGULATORY]",
            "investment decision", "financial advice", "medical", "legal"
        ]

        self.complex_keywords = [
            "comprehensive", "detailed analysis", "in-depth", "extensive",
            "design and implement", "complete system", "architecture",
            "multi-step", "end-to-end", "full stack", "create", "build"
        ]

        self.moderate_keywords = [
            "explain", "analyze", "summarize", "compare", "evaluate",
            "write a function", "implement", "calculate", "describe",
            "assess", "review"
        ]

        self.simple_keywords = [
            "what is", "define", "translate", "hello", "hi",
            "yes or no", "true or false", "list", "name", "when", "where"
        ]

        # FinTech-specific patterns
        self.fintech_patterns = {
            "earnings": RequestComplexity.MODERATE,
            "risk assessment": RequestComplexity.COMPLEX,
            "var calculation": RequestComplexity.CRITICAL,
            "trading signal": RequestComplexity.CRITICAL,
            "fraud detection": RequestComplexity.CRITICAL,
            "compliance report": RequestComplexity.CRITICAL,
            "portfolio optimization": RequestComplexity.COMPLEX,
            "technical analysis": RequestComplexity.COMPLEX,
            "account balance": RequestComplexity.SIMPLE,
            "transaction history": RequestComplexity.SIMPLE
        }

    def classify(self, request: RoutingRequest) -> tuple:
        """
        Classify the complexity of a request

        Args:
            request: The routing request

        Returns:
            Tuple of (RequestComplexity, confidence_score)
        """
        content_lower = request.content.lower()
        confidence = 0.8  # Default confidence

        # Check for critical keywords first (highest priority)
        for keyword in self.critical_keywords:
            if keyword.lower() in content_lower:
                return RequestComplexity.CRITICAL, 0.95

        # Check FinTech-specific patterns
        for pattern, complexity in self.fintech_patterns.items():
            if pattern in content_lower:
                # Upgrade complexity for production/urgent
                if any(kw.lower() in content_lower for kw in self.critical_keywords[:4]):
                    return RequestComplexity.CRITICAL, 0.95
                return complexity, 0.85

        # Estimate based on content length
        word_count = len(request.content.split())
        estimated_tokens = word_count * 1.3  # Rough token estimate

        # Very long requests are likely complex
        if estimated_tokens > 1000:
            return RequestComplexity.CRITICAL, 0.75
        elif estimated_tokens > 500:
            return RequestComplexity.COMPLEX, 0.75

        # Check for complex keywords
        for keyword in self.complex_keywords:
            if keyword in content_lower:
                return RequestComplexity.COMPLEX, 0.85

        # Check for moderate keywords
        for keyword in self.moderate_keywords:
            if keyword in content_lower:
                return RequestComplexity.MODERATE, 0.85

        # Check for simple keywords
        for keyword in self.simple_keywords:
            if keyword in content_lower:
                return RequestComplexity.SIMPLE, 0.90

        # Default based on length
        if word_count < 10:
            return RequestComplexity.SIMPLE, 0.70
        elif word_count < 50:
            return RequestComplexity.MODERATE, 0.70
        elif word_count < 200:
            return RequestComplexity.COMPLEX, 0.70
        else:
            return RequestComplexity.CRITICAL, 0.70


class CostOptimizer:
    """Optimizes model selection based on cost and requirements"""

    def __init__(self, models: List[ModelConfig]):
        self.models = {m.name: m for m in models}
        self.user_budgets = {
            "free": 0.001,     # $0.001 per request max
            "basic": 0.01,     # $0.01 per request max
            "premium": 0.10,   # $0.10 per request max
            "enterprise": None  # No limit
        }

    def select_optimal_model(
        self,
        complexity: RequestComplexity,
        request: RoutingRequest,
        available_models: List[str]
    ) -> tuple:
        """
        Select the most cost-effective model that meets requirements

        Returns:
            Tuple of (model_name, estimated_cost)
        """
        # Get user budget
        user_tier = request.metadata.get("user_tier", "basic")
        max_budget = self.user_budgets.get(user_tier, 0.01)

        if request.max_cost:
            max_budget = min(max_budget, request.max_cost) if max_budget else request.max_cost

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

            # Estimate cost
            estimated_tokens = len(request.content.split()) * 2  # Input + output estimate
            estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens

            # Check budget
            if max_budget and estimated_cost > max_budget:
                continue

            candidates.append((model_name, estimated_cost, model.avg_latency_ms))

        if not candidates:
            # Fallback to cheapest available model
            if not available_models:
                # No models available at all
                return None, float('inf')

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
            # Choose most capable for critical requests
            selected = max(
                candidates,
                key=lambda x: self.models[x[0]].tier.value
            )
        else:
            # Balance cost and performance for moderate/complex
            # Simple scoring: lower cost is better
            selected = min(candidates, key=lambda x: x[1])

        return selected[0], selected[1]


class IntelligentRouter:
    """Routes requests to appropriate models based on complexity and cost"""

    def __init__(self, model_configs: List[ModelConfig]):
        """
        Initialize router with model configurations

        Args:
            model_configs: List of available model configurations
        """
        self.model_configs = model_configs
        self.models = {m.name: m for m in model_configs}
        self.classifier = ComplexityClassifier()
        self.cost_optimizer = CostOptimizer(model_configs)

        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "requests_per_model": {},
            "requests_per_complexity": {
                RequestComplexity.SIMPLE: 0,
                RequestComplexity.MODERATE: 0,
                RequestComplexity.COMPLEX: 0,
                RequestComplexity.CRITICAL: 0
            },
            "total_estimated_cost": 0.0,
            "fallback_count": 0
        }

        # Initialize per-model metrics
        for config in model_configs:
            self.metrics["requests_per_model"][config.name] = 0

        # Model health tracking
        self.model_health = {m.name: {"healthy": True, "error_rate": 0.0} for m in model_configs}

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route a request to the appropriate model

        Args:
            request: The routing request

        Returns:
            RoutingDecision with selected model and metadata
        """
        # Classify complexity
        complexity, confidence = self.classifier.classify(request)

        # Get available models
        available_models = self._get_available_models(complexity)

        if not available_models:
            # No models available for this complexity, use fallback
            available_models = [m.name for m in self.model_configs if self.model_health[m.name]["healthy"]]
            if not available_models:
                raise Exception("No healthy models available")

        # Select optimal model
        selected_model_name, estimated_cost = self.cost_optimizer.select_optimal_model(
            complexity, request, available_models
        )

        selected_model = self.models[selected_model_name]

        # Estimate tokens and latency
        estimated_tokens = len(request.content.split()) * 2
        estimated_latency = selected_model.avg_latency_ms

        # Check if this is a fallback
        is_fallback = selected_model.max_complexity.value < complexity.value

        # Get alternatives
        alternatives = self._get_alternatives(
            complexity, request, available_models, selected_model_name
        )

        # Create routing decision
        decision = RoutingDecision(
            request_id=request.id,
            model_name=selected_model.name,
            model_path=selected_model.model_path,
            model_tier=selected_model.tier,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            confidence_score=confidence,
            reasoning=self._generate_reasoning(
                complexity, selected_model, estimated_cost, confidence, is_fallback
            ),
            fallback=is_fallback,
            alternatives=alternatives
        )

        # Update metrics
        self._update_metrics(decision)

        logger.info(f"Routed request {request.id}: complexity={complexity.name}, "
                   f"model={selected_model.name}, cost=${estimated_cost:.6f}")

        return decision

    def _get_available_models(self, complexity: RequestComplexity) -> List[str]:
        """Get list of models that can handle the complexity"""
        available = []

        for model in self.model_configs:
            # Check if model can handle complexity
            if model.max_complexity.value >= complexity.value:
                # Check if model is healthy
                if self.model_health[model.name]["healthy"]:
                    available.append(model.name)

        return available

    def _get_alternatives(
        self,
        complexity: RequestComplexity,
        request: RoutingRequest,
        available_models: List[str],
        selected: str
    ) -> List[Dict[str, Any]]:
        """Get alternative routing options"""
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
                "estimated_latency_ms": model.avg_latency_ms
            })

        return alternatives

    def _generate_reasoning(
        self,
        complexity: RequestComplexity,
        model: ModelConfig,
        cost: float,
        confidence: float,
        is_fallback: bool
    ) -> str:
        """Generate reasoning for the routing decision"""
        if is_fallback:
            return (f"Request classified as {complexity.name} with {confidence:.0%} confidence. "
                   f"Using {model.name} as fallback (capability: {model.max_complexity.name}). "
                   f"Estimated cost: ${cost:.6f}")

        return (f"Request classified as {complexity.name} with {confidence:.0%} confidence. "
               f"Selected {model.name} for optimal cost-performance. "
               f"Estimated cost: ${cost:.6f}")

    def _update_metrics(self, decision: RoutingDecision):
        """Update routing metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["requests_per_model"][decision.model_name] += 1
        self.metrics["requests_per_complexity"][decision.complexity] += 1
        self.metrics["total_estimated_cost"] += decision.estimated_cost
        if decision.fallback:
            self.metrics["fallback_count"] += 1

    def mark_model_unhealthy(self, model_name: str):
        """Mark a model as unhealthy"""
        if model_name in self.model_health:
            self.model_health[model_name]["healthy"] = False
            logger.warning(f"Model {model_name} marked as unhealthy")

    def mark_model_healthy(self, model_name: str):
        """Mark a model as healthy"""
        if model_name in self.model_health:
            self.model_health[model_name]["healthy"] = True
            self.model_health[model_name]["error_rate"] = 0.0
            logger.info(f"Model {model_name} marked as healthy")

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics["total_requests"] = 0
        self.metrics["total_estimated_cost"] = 0.0
        self.metrics["fallback_count"] = 0
        for key in self.metrics["requests_per_model"]:
            self.metrics["requests_per_model"][key] = 0
        for key in self.metrics["requests_per_complexity"]:
            self.metrics["requests_per_complexity"][key] = 0


def create_fintech_router() -> IntelligentRouter:
    """
    Create a router with FinTech-optimized model configurations

    Returns:
        Configured IntelligentRouter for FinTech use cases
    """
    model_configs = [
        # Small, fast model for simple queries
        ModelConfig(
            name="opt-125m-fast",
            tier=ModelTier.TINY,
            model_path="facebook/opt-125m",
            max_complexity=RequestComplexity.SIMPLE,
            cost_per_1k_tokens=0.0001,
            avg_latency_ms=50,
            capabilities=["general", "simple_qa"]
        ),
        # Medium model for analysis
        ModelConfig(
            name="opt-1.3b-balanced",
            tier=ModelTier.SMALL,
            model_path="facebook/opt-1.3b",
            max_complexity=RequestComplexity.MODERATE,
            cost_per_1k_tokens=0.001,
            avg_latency_ms=100,
            capabilities=["general", "analysis", "summarization"]
        ),
        # Large model for complex analysis
        ModelConfig(
            name="opt-6.7b-powerful",
            tier=ModelTier.MEDIUM,
            model_path="facebook/opt-6.7b",
            max_complexity=RequestComplexity.COMPLEX,
            cost_per_1k_tokens=0.01,
            avg_latency_ms=300,
            capabilities=["general", "analysis", "reasoning", "code"]
        ),
        # Premium model for critical/production
        ModelConfig(
            name="llama-13b-production",
            tier=ModelTier.LARGE,
            model_path="meta-llama/Llama-2-13b-chat-hf",
            max_complexity=RequestComplexity.CRITICAL,
            cost_per_1k_tokens=0.1,
            avg_latency_ms=500,
            capabilities=["general", "analysis", "reasoning", "financial", "compliance"]
        )
    ]

    return IntelligentRouter(model_configs)


if __name__ == "__main__":
    # Example usage
    router = create_fintech_router()

    test_prompts = [
        ("simple_1", "What is my account balance?"),
        ("moderate_1", "Explain how compound interest works in finance"),
        ("complex_1", "Analyze the Q3 earnings report for Apple Inc with detailed breakdown"),
        ("critical_1", "[PRODUCTION] Generate real-time trading signals for NASDAQ stocks")
    ]

    print("Intelligent Router Example\n" + "="*60)

    for request_id, prompt in test_prompts:
        request = RoutingRequest(
            id=request_id,
            content=prompt,
            user_id="demo_user",
            metadata={"user_tier": "basic"}
        )

        decision = router.route(request)

        print(f"\nRequest: {prompt[:60]}...")
        print(f"  Complexity: {decision.complexity.name}")
        print(f"  Selected Model: {decision.model_name}")
        print(f"  Model Tier: {decision.model_tier.value}")
        print(f"  Estimated Cost: ${decision.estimated_cost:.6f}")
        print(f"  Reasoning: {decision.reasoning}")

    print("\n" + "="*60)
    print("Routing Metrics:")
    metrics = router.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")