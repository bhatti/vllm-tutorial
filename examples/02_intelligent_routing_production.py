#!/usr/bin/env python3
"""
Example 2: Production-Ready Intelligent Routing with Cost Budgets
Demonstrates real-world routing decisions based on complexity, cost, and SLAs
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RequestComplexity(Enum):
    """Request complexity classification"""
    SIMPLE = "simple"           # Simple Q&A, < 100 tokens
    MODERATE = "moderate"       # Analysis, 100-500 tokens
    COMPLEX = "complex"         # Deep analysis, > 500 tokens
    CRITICAL = "critical"       # Mission-critical, needs best model


class ModelTier(Enum):
    """Model tier for cost/performance tradeoff"""
    TINY = "tiny"       # Phi-2 2.7B
    SMALL = "small"     # Mistral-7B
    MEDIUM = "medium"   # Llama-13B
    LARGE = "large"     # Llama-70B


@dataclass
class ModelConfig:
    """Configuration for a vLLM model"""
    name: str
    tier: ModelTier
    model_path: str
    max_complexity: RequestComplexity
    cost_per_1k_tokens: float  # USD
    avg_latency_ms: float
    max_tokens: int = 2048
    gpu_memory_gb: float = 0.0


@dataclass
class CostBudget:
    """Cost budget constraints"""
    max_cost_per_request: float = 0.1      # Max $0.10 per request
    daily_budget: float = 100.0             # Max $100 per day
    current_daily_spend: float = 0.0
    requests_today: int = 0
    high_cost_threshold: float = 0.05      # Flag requests > $0.05


@dataclass
class SLARequirements:
    """Service Level Agreement requirements"""
    max_latency_ms: float = 1000.0         # P95 latency target
    min_availability: float = 0.999        # 99.9% uptime
    max_error_rate: float = 0.01          # 1% error rate


@dataclass
class RoutingDecision:
    """Routing decision with full context"""
    model_name: str
    complexity: RequestComplexity
    estimated_cost: float
    estimated_latency_ms: float
    estimated_tokens: int
    reasoning: str
    fallback_model: Optional[str] = None
    budget_status: str = "ok"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ProductionIntelligentRouter:
    """
    Production-ready intelligent router with:
    - Cost budget enforcement
    - SLA compliance
    - Fallback handling
    - Observability
    """

    def __init__(
        self,
        models: List[ModelConfig],
        cost_budget: Optional[CostBudget] = None,
        sla: Optional[SLARequirements] = None
    ):
        self.models = {m.name: m for m in models}
        self.cost_budget = cost_budget or CostBudget()
        self.sla = sla or SLARequirements()

        # Metrics
        self.total_requests = 0
        self.total_cost = 0.0
        self.complexity_counts = {c: 0 for c in RequestComplexity}
        self.model_usage = {m.name: 0 for m in models}
        self.budget_violations = 0

        logger.info(f"Router initialized with {len(models)} models")
        logger.info(f"Daily budget: ${self.cost_budget.daily_budget}")
        logger.info(f"Max latency: {self.sla.max_latency_ms}ms")

    def classify_complexity(self, prompt: str, metadata: Dict) -> RequestComplexity:
        """
        Classify request complexity based on content and metadata

        Real-world signals:
        - Prompt length
        - Required analysis depth
        - Expected output length
        - Business criticality
        """
        # Get expected output length from metadata
        max_tokens = metadata.get("max_tokens", 256)

        # Check for critical keywords
        critical_keywords = ["urgent", "critical", "production", "executive"]
        is_critical = any(kw in prompt.lower() for kw in critical_keywords)

        if is_critical or metadata.get("priority") == "critical":
            return RequestComplexity.CRITICAL

        # Analyze prompt complexity
        prompt_length = len(prompt.split())

        if prompt_length < 20 and max_tokens < 100:
            return RequestComplexity.SIMPLE
        elif prompt_length < 100 and max_tokens < 500:
            return RequestComplexity.MODERATE
        else:
            return RequestComplexity.COMPLEX

    def estimate_tokens(self, prompt: str, max_tokens: int) -> int:
        """Estimate total tokens (input + output)"""
        # Rough estimation: 1 token ≈ 0.75 words
        input_tokens = int(len(prompt.split()) * 1.33)
        return input_tokens + max_tokens

    def check_cost_budget(self, estimated_cost: float) -> Tuple[bool, str]:
        """
        Check if request fits within budget constraints

        Returns:
            (allowed, reason)
        """
        # Check per-request limit
        if estimated_cost > self.cost_budget.max_cost_per_request:
            return False, f"Cost ${estimated_cost:.4f} exceeds per-request limit"

        # Check daily budget
        if (self.cost_budget.current_daily_spend + estimated_cost) > self.cost_budget.daily_budget:
            return False, f"Would exceed daily budget (${self.cost_budget.current_daily_spend:.2f}/${self.cost_budget.daily_budget:.2f})"

        # Warn on high cost
        if estimated_cost > self.cost_budget.high_cost_threshold:
            logger.warning(f"High-cost request: ${estimated_cost:.4f}")

        return True, "ok"

    def select_model(
        self,
        complexity: RequestComplexity,
        estimated_tokens: int,
        max_latency_ms: Optional[float] = None
    ) -> Optional[ModelConfig]:
        """
        Select optimal model based on complexity, cost, and latency

        Selection criteria:
        1. Can handle complexity level
        2. Meets latency SLA
        3. Minimizes cost
        """
        max_latency = max_latency_ms or self.sla.max_latency_ms

        # Filter models that can handle this complexity
        candidates = [
            m for m in self.models.values()
            if m.max_complexity.value >= complexity.value
        ]

        if not candidates:
            logger.error(f"No model can handle {complexity}")
            return None

        # Filter by latency requirement
        candidates = [
            m for m in candidates
            if m.avg_latency_ms <= max_latency
        ]

        if not candidates:
            logger.warning(f"No model meets {max_latency}ms latency SLA")
            # Relax constraint and take fastest
            candidates = sorted(self.models.values(), key=lambda m: m.avg_latency_ms)
            logger.info(f"Falling back to fastest model: {candidates[0].name}")

        # Select cheapest model that meets requirements
        best_model = min(candidates, key=lambda m: m.cost_per_1k_tokens)

        logger.info(
            f"Selected {best_model.name} for {complexity.value} "
            f"(cost: ${best_model.cost_per_1k_tokens}/1k, "
            f"latency: {best_model.avg_latency_ms}ms)"
        )

        return best_model

    def route_request(
        self,
        prompt: str,
        metadata: Optional[Dict] = None
    ) -> RoutingDecision:
        """
        Make routing decision with full production logic

        Args:
            prompt: User prompt
            metadata: Request metadata (max_tokens, priority, etc.)

        Returns:
            RoutingDecision with selected model and reasoning
        """
        metadata = metadata or {}
        self.total_requests += 1

        # Step 1: Classify complexity
        complexity = self.classify_complexity(prompt, metadata)
        self.complexity_counts[complexity] += 1

        logger.info(f"Request #{self.total_requests}: {complexity.value}")

        # Step 2: Estimate tokens and cost
        max_tokens = metadata.get("max_tokens", 256)
        estimated_tokens = self.estimate_tokens(prompt, max_tokens)

        # Step 3: Select model
        max_latency = metadata.get("max_latency_ms")
        model = self.select_model(complexity, estimated_tokens, max_latency)

        if not model:
            logger.error("No suitable model found")
            # Emergency fallback to smallest model
            model = min(self.models.values(), key=lambda m: m.cost_per_1k_tokens)

        # Step 4: Calculate cost
        estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens

        # Step 5: Check budget
        budget_ok, budget_reason = self.check_cost_budget(estimated_cost)

        if not budget_ok:
            logger.warning(f"Budget constraint: {budget_reason}")
            self.budget_violations += 1

            # Try to downgrade to cheaper model
            cheaper_models = sorted(
                self.models.values(),
                key=lambda m: m.cost_per_1k_tokens
            )
            for fallback_model in cheaper_models:
                fallback_cost = (estimated_tokens / 1000) * fallback_model.cost_per_1k_tokens
                budget_ok, _ = self.check_cost_budget(fallback_cost)
                if budget_ok:
                    logger.info(f"Downgraded to {fallback_model.name} to meet budget")
                    model = fallback_model
                    estimated_cost = fallback_cost
                    break

        # Step 6: Update metrics
        self.model_usage[model.name] += 1
        self.total_cost += estimated_cost
        self.cost_budget.current_daily_spend += estimated_cost
        self.cost_budget.requests_today += 1

        # Step 7: Create decision
        decision = RoutingDecision(
            model_name=model.name,
            complexity=complexity,
            estimated_cost=estimated_cost,
            estimated_latency_ms=model.avg_latency_ms,
            estimated_tokens=estimated_tokens,
            reasoning=f"Selected {model.tier.value} tier for {complexity.value} complexity",
            budget_status=budget_reason
        )

        logger.info(
            f"Routed to {model.name}: "
            f"cost=${estimated_cost:.4f}, "
            f"latency~{model.avg_latency_ms}ms"
        )

        return decision

    def get_metrics(self) -> Dict:
        """Get routing metrics for observability"""
        return {
            "total_requests": self.total_requests,
            "total_cost": self.total_cost,
            "avg_cost_per_request": self.total_cost / max(1, self.total_requests),
            "complexity_distribution": {
                k.value: v for k, v in self.complexity_counts.items()
            },
            "model_usage": self.model_usage,
            "budget_violations": self.budget_violations,
            "daily_spend": self.cost_budget.current_daily_spend,
            "budget_remaining": self.cost_budget.daily_budget - self.cost_budget.current_daily_spend,
            "budget_utilization_pct": (
                self.cost_budget.current_daily_spend / self.cost_budget.daily_budget * 100
            )
        }

    def reset_daily_budget(self):
        """Reset daily budget (call at midnight)"""
        logger.info(
            f"Resetting daily budget. Spent: ${self.cost_budget.current_daily_spend:.2f}, "
            f"Requests: {self.cost_budget.requests_today}"
        )
        self.cost_budget.current_daily_spend = 0.0
        self.cost_budget.requests_today = 0


def main():
    """Demonstrate production routing with cost budgets"""

    print("=" * 80)
    print("Production Intelligent Routing with Cost Budgets")
    print("=" * 80)

    # Define model configurations (realistic for L4 GPU)
    models = [
        ModelConfig(
            name="phi-2",
            tier=ModelTier.TINY,
            model_path="microsoft/phi-2",
            max_complexity=RequestComplexity.SIMPLE,
            cost_per_1k_tokens=0.0001,  # Very cheap
            avg_latency_ms=50,
            gpu_memory_gb=6.0
        ),
        ModelConfig(
            name="mistral-7b",
            tier=ModelTier.SMALL,
            model_path="mistralai/Mistral-7B-Instruct-v0.2",
            max_complexity=RequestComplexity.MODERATE,
            cost_per_1k_tokens=0.001,
            avg_latency_ms=150,
            gpu_memory_gb=16.0
        ),
        ModelConfig(
            name="llama-13b",
            tier=ModelTier.MEDIUM,
            model_path="meta-llama/Llama-2-13b-chat-hf",
            max_complexity=RequestComplexity.COMPLEX,
            cost_per_1k_tokens=0.005,
            avg_latency_ms=400,
            gpu_memory_gb=26.0
        ),
    ]

    # Configure cost budget
    budget = CostBudget(
        max_cost_per_request=0.05,
        daily_budget=50.0,
        high_cost_threshold=0.01
    )

    # Configure SLA
    sla = SLARequirements(
        max_latency_ms=500,
        min_availability=0.999,
        max_error_rate=0.01
    )

    # Initialize router
    router = ProductionIntelligentRouter(models, budget, sla)

    # Example requests with different characteristics
    test_requests = [
        {
            "prompt": "What is the capital of France?",
            "metadata": {"max_tokens": 50},
            "description": "Simple factual query"
        },
        {
            "prompt": "Analyze the quarterly earnings report and provide key insights.",
            "metadata": {"max_tokens": 300},
            "description": "Moderate analysis task"
        },
        {
            "prompt": "URGENT: Perform comprehensive risk analysis on the portfolio including regulatory compliance, market exposure, and hedging strategies.",
            "metadata": {"max_tokens": 1000, "priority": "critical"},
            "description": "Critical complex analysis"
        },
        {
            "prompt": "Summarize this financial document.",
            "metadata": {"max_tokens": 200, "max_latency_ms": 200},
            "description": "Low latency requirement"
        },
    ]

    print("\n" + "="*80)
    print("Processing Test Requests")
    print("="*80)

    # Process requests
    for i, req in enumerate(test_requests, 1):
        print(f"\n📝 Request {i}: {req['description']}")
        print(f"   Prompt: {req['prompt'][:60]}...")

        decision = router.route_request(req["prompt"], req["metadata"])

        print(f"\n✅ Routing Decision:")
        print(f"   Model: {decision.model_name}")
        print(f"   Complexity: {decision.complexity.value}")
        print(f"   Est. Cost: ${decision.estimated_cost:.6f}")
        print(f"   Est. Latency: {decision.estimated_latency_ms}ms")
        print(f"   Est. Tokens: {decision.estimated_tokens}")
        print(f"   Budget Status: {decision.budget_status}")
        print(f"   Reasoning: {decision.reasoning}")

    # Show metrics
    print("\n" + "="*80)
    print("📊 Router Metrics")
    print("="*80)

    metrics = router.get_metrics()
    print(json.dumps(metrics, indent=2))

    # Cost analysis
    print("\n" + "="*80)
    print("💰 Cost Analysis")
    print("="*80)
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Total cost: ${metrics['total_cost']:.6f}")
    print(f"Average cost per request: ${metrics['avg_cost_per_request']:.6f}")
    print(f"Budget utilization: {metrics['budget_utilization_pct']:.1f}%")
    print(f"Budget remaining: ${metrics['budget_remaining']:.2f}")
    print(f"Budget violations: {metrics['budget_violations']}")

    # Projected costs
    print("\n" + "="*80)
    print("📈 Cost Projections")
    print("="*80)

    daily_requests = 10000
    projected_daily = metrics['avg_cost_per_request'] * daily_requests
    projected_monthly = projected_daily * 30

    print(f"At {daily_requests:,} requests/day:")
    print(f"  Daily cost: ${projected_daily:.2f}")
    print(f"  Monthly cost: ${projected_monthly:,.2f}")
    print(f"  Yearly cost: ${projected_monthly * 12:,.2f}")

    # Compare with alternatives
    print("\n" + "="*80)
    print("💵 Cost Comparison")
    print("="*80)

    openai_cost_per_request = metrics['avg_cost_per_request'] * 25  # Assume 25x more expensive
    savings_per_request = openai_cost_per_request - metrics['avg_cost_per_request']
    monthly_savings = savings_per_request * daily_requests * 30

    print(f"OpenAI API (estimated): ${openai_cost_per_request:.6f}/request")
    print(f"vLLM self-hosted: ${metrics['avg_cost_per_request']:.6f}/request")
    print(f"Savings per request: ${savings_per_request:.6f}")
    print(f"Monthly savings: ${monthly_savings:,.2f}")
    print(f"Yearly savings: ${monthly_savings * 12:,.2f}")

    print("\n" + "="*80)
    print("✅ Example Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
