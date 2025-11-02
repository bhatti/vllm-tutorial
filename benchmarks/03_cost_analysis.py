#!/usr/bin/env python3
"""
Benchmark 3: Comprehensive Cost Analysis with Intelligent Routing
- Uses tiktoken for accurate token counting
- Compares OpenAI API vs self-hosted vLLM costs
- Implements complexity-based routing for cost optimization
- Calculates ROI for different deployment scenarios
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  tiktoken not available. Install: pip install tiktoken")


class RequestComplexity(Enum):
    """Request complexity levels"""
    SIMPLE = "simple"          # Single sentence, factual
    MODERATE = "moderate"      # Paragraph, analysis
    COMPLEX = "complex"        # Multi-paragraph, reasoning
    CRITICAL = "critical"      # Extensive, mission-critical


@dataclass
class ModelConfig:
    """Model configuration with pricing"""
    name: str
    tier: str
    vram_gb: float
    cost_per_hour: float  # GPU rental cost
    input_cost_per_1m: float  # Cost per 1M input tokens (OpenAI-equivalent)
    output_cost_per_1m: float  # Cost per 1M output tokens
    max_tokens_per_sec: float  # Throughput
    avg_latency_ms: float


@dataclass
class CostMetrics:
    """Cost analysis metrics"""
    model_name: str
    framework: str
    num_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    inference_cost: float
    infrastructure_cost: float
    total_cost: float
    cost_per_request: float
    cost_per_1k_tokens: float
    avg_latency_ms: float


class ComplexityClassifier:
    """Classify request complexity based on prompt characteristics"""

    def __init__(self):
        self.simple_keywords = ['what', 'who', 'when', 'where', 'define', 'list']
        self.complex_keywords = ['analyze', 'compare', 'evaluate', 'comprehensive', 'detailed']
        self.critical_keywords = ['urgent', 'critical', 'mission-critical', 'emergency']

    def classify(self, prompt: str, metadata: Optional[Dict] = None) -> RequestComplexity:
        """
        Classify request complexity

        Rules:
        - CRITICAL: Contains critical keywords or metadata priority=critical
        - COMPLEX: >500 chars OR contains complex keywords
        - SIMPLE: <100 chars AND contains simple keywords
        - MODERATE: Everything else
        """
        prompt_lower = prompt.lower()

        # Check metadata
        if metadata:
            if metadata.get('priority') == 'critical':
                return RequestComplexity.CRITICAL
            if metadata.get('sla_ms', 1000) < 100:
                return RequestComplexity.SIMPLE  # Need fast response

        # Check critical keywords
        if any(kw in prompt_lower for kw in self.critical_keywords):
            return RequestComplexity.CRITICAL

        # Check length and keywords
        if len(prompt) > 500 or any(kw in prompt_lower for kw in self.complex_keywords):
            return RequestComplexity.COMPLEX

        if len(prompt) < 100 and any(kw in prompt_lower for kw in self.simple_keywords):
            return RequestComplexity.SIMPLE

        return RequestComplexity.MODERATE

    def estimate_tokens(self, prompt: str, encoder_name: str = "cl100k_base") -> Tuple[int, int]:
        """
        Estimate input and output tokens using tiktoken

        Returns: (input_tokens, estimated_output_tokens)
        """
        if not TIKTOKEN_AVAILABLE:
            # Fallback estimation: ~4 chars per token
            input_tokens = len(prompt) // 4
            output_tokens = input_tokens // 2  # Assume output is half of input
            return input_tokens, output_tokens

        try:
            encoder = tiktoken.get_encoding(encoder_name)
            input_tokens = len(encoder.encode(prompt))

            # Estimate output tokens based on complexity
            complexity = self.classify(prompt)
            if complexity == RequestComplexity.SIMPLE:
                output_tokens = min(input_tokens, 50)  # Short answers
            elif complexity == RequestComplexity.MODERATE:
                output_tokens = input_tokens // 2  # Medium answers
            elif complexity == RequestComplexity.COMPLEX:
                output_tokens = input_tokens  # Long answers
            else:  # CRITICAL
                output_tokens = input_tokens * 2  # Very long answers

            return input_tokens, output_tokens

        except Exception as e:
            print(f"⚠️  tiktoken failed: {e}, using fallback")
            input_tokens = len(prompt) // 4
            output_tokens = input_tokens // 2
            return input_tokens, output_tokens


class IntelligentRouter:
    """Route requests to optimal model based on complexity and cost"""

    def __init__(self, models: List[ModelConfig], cost_budget: Optional[float] = None):
        self.models = sorted(models, key=lambda m: m.cost_per_hour)  # Sort by cost
        self.cost_budget = cost_budget  # Daily budget in dollars
        self.total_cost = 0.0
        self.classifier = ComplexityClassifier()

    def route(self, prompt: str, metadata: Optional[Dict] = None) -> Tuple[ModelConfig, str]:
        """
        Route request to optimal model

        Returns: (selected_model, reasoning)
        """
        complexity = self.classifier.classify(prompt, metadata)
        input_tokens, output_tokens = self.classifier.estimate_tokens(prompt)

        # Check budget
        budget_remaining = float('inf')
        if self.cost_budget:
            budget_remaining = self.cost_budget - self.total_cost

        # Routing logic
        if complexity == RequestComplexity.SIMPLE:
            # Use cheapest model
            selected = self.models[0]
            reasoning = "Simple query - using most cost-effective model"

        elif complexity == RequestComplexity.MODERATE:
            # Use mid-tier model if available
            mid_idx = min(1, len(self.models) - 1)
            selected = self.models[mid_idx]
            reasoning = "Moderate complexity - using balanced model"

        elif complexity == RequestComplexity.COMPLEX:
            # Use better model
            high_idx = min(2, len(self.models) - 1)
            selected = self.models[high_idx]
            reasoning = "Complex analysis - using advanced model"

        else:  # CRITICAL
            # Use best model if latency allows
            if metadata and metadata.get('sla_ms', 1000) < 200:
                # Need fast response, use fastest model
                selected = min(self.models, key=lambda m: m.avg_latency_ms)
                reasoning = "Critical + low latency SLA - using fastest model"
            else:
                # Use most capable model
                selected = self.models[-1]
                reasoning = "Critical request - using most capable model"

        # Estimate cost for this request
        estimated_cost = self._estimate_request_cost(selected, input_tokens, output_tokens)

        # Budget check - downgrade if needed
        if estimated_cost > budget_remaining and self.cost_budget:
            # Find cheapest model that fits budget
            for model in self.models:
                cost = self._estimate_request_cost(model, input_tokens, output_tokens)
                if cost <= budget_remaining:
                    selected = model
                    reasoning = f"Budget constraint - downgraded to {model.name}"
                    break

        return selected, reasoning

    def _estimate_request_cost(self, model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a single request"""
        # Token costs
        input_cost = (input_tokens / 1_000_000) * model.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * model.output_cost_per_1m

        # Infrastructure cost (amortized per request)
        # Assume 1000 requests per hour utilization
        infra_cost = model.cost_per_hour / 1000

        return input_cost + output_cost + infra_cost


class CostAnalyzer:
    """Analyze and compare costs across different deployment scenarios"""

    def __init__(self):
        self.results: List[CostMetrics] = []
        self.classifier = ComplexityClassifier()

        # Model configurations based on real benchmarks and market pricing
        self.models = [
            # Self-hosted vLLM models
            ModelConfig(
                name="phi-2",
                tier="tiny",
                vram_gb=6.0,
                cost_per_hour=0.45,  # L4 GPU spot pricing
                input_cost_per_1m=0.10,  # Amortized infrastructure
                output_cost_per_1m=0.10,
                max_tokens_per_sec=934.0,  # From our benchmarks
                avg_latency_ms=99.0,
            ),
            ModelConfig(
                name="mistral-7b",
                tier="small",
                vram_gb=14.0,
                cost_per_hour=0.45,  # L4 can handle this
                input_cost_per_1m=0.15,
                output_cost_per_1m=0.15,
                max_tokens_per_sec=600.0,
                avg_latency_ms=150.0,
            ),
            ModelConfig(
                name="llama-13b",
                tier="medium",
                vram_gb=26.0,
                cost_per_hour=1.10,  # A100 40GB spot pricing
                input_cost_per_1m=0.25,
                output_cost_per_1m=0.25,
                max_tokens_per_sec=400.0,
                avg_latency_ms=200.0,
            ),
        ]

        # OpenAI pricing (for comparison)
        self.openai_pricing = {
            "gpt-3.5-turbo": {
                "input_cost_per_1m": 0.50,
                "output_cost_per_1m": 1.50,
                "avg_latency_ms": 500.0,
            },
            "gpt-4": {
                "input_cost_per_1m": 30.00,
                "output_cost_per_1m": 60.00,
                "avg_latency_ms": 1000.0,
            },
        }

    def analyze_workload(self, requests: List[Dict[str, any]], use_routing: bool = True) -> Dict[str, CostMetrics]:
        """
        Analyze cost for a workload

        requests: List of {"prompt": str, "metadata": Dict}
        use_routing: If True, use intelligent routing; else use single model
        """
        results = {}

        # Analyze with vLLM + routing
        if use_routing:
            router = IntelligentRouter(self.models, cost_budget=None)
            vllm_metrics = self._analyze_vllm_routing(requests, router)
            results["vLLM-Routed"] = vllm_metrics

        # Analyze with single vLLM model (cheapest)
        vllm_single = self._analyze_vllm_single(requests, self.models[0])
        results["vLLM-Single"] = vllm_single

        # Analyze with OpenAI GPT-3.5
        openai_35 = self._analyze_openai(requests, "gpt-3.5-turbo")
        results["OpenAI-GPT3.5"] = openai_35

        # Analyze with OpenAI GPT-4
        openai_4 = self._analyze_openai(requests, "gpt-4")
        results["OpenAI-GPT4"] = openai_4

        return results

    def _analyze_vllm_routing(self, requests: List[Dict], router: IntelligentRouter) -> CostMetrics:
        """Analyze cost with intelligent routing"""
        total_input_tokens = 0
        total_output_tokens = 0
        total_inference_cost = 0.0
        total_infra_cost = 0.0
        total_latency = 0.0

        for req in requests:
            prompt = req["prompt"]
            metadata = req.get("metadata", {})

            # Route request
            model, reasoning = router.route(prompt, metadata)

            # Estimate tokens
            input_tokens, output_tokens = self.classifier.estimate_tokens(prompt)

            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * model.input_cost_per_1m
            output_cost = (output_tokens / 1_000_000) * model.output_cost_per_1m
            infra_cost = model.cost_per_hour / 1000  # Assume 1000 req/hr utilization

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_inference_cost += (input_cost + output_cost)
            total_infra_cost += infra_cost
            total_latency += model.avg_latency_ms

        total_cost = total_inference_cost + total_infra_cost

        return CostMetrics(
            model_name="vLLM-Routed",
            framework="vLLM",
            num_requests=len(requests),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            inference_cost=total_inference_cost,
            infrastructure_cost=total_infra_cost,
            total_cost=total_cost,
            cost_per_request=total_cost / len(requests),
            cost_per_1k_tokens=(total_cost / (total_input_tokens + total_output_tokens)) * 1000,
            avg_latency_ms=total_latency / len(requests),
        )

    def _analyze_vllm_single(self, requests: List[Dict], model: ModelConfig) -> CostMetrics:
        """Analyze cost with single vLLM model"""
        total_input_tokens = 0
        total_output_tokens = 0

        for req in requests:
            input_tokens, output_tokens = self.classifier.estimate_tokens(req["prompt"])
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # Calculate costs
        inference_cost = (
            (total_input_tokens / 1_000_000) * model.input_cost_per_1m +
            (total_output_tokens / 1_000_000) * model.output_cost_per_1m
        )
        infra_cost = (model.cost_per_hour / 1000) * len(requests)
        total_cost = inference_cost + infra_cost

        return CostMetrics(
            model_name=model.name,
            framework="vLLM",
            num_requests=len(requests),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            inference_cost=inference_cost,
            infrastructure_cost=infra_cost,
            total_cost=total_cost,
            cost_per_request=total_cost / len(requests),
            cost_per_1k_tokens=(total_cost / (total_input_tokens + total_output_tokens)) * 1000,
            avg_latency_ms=model.avg_latency_ms,
        )

    def _analyze_openai(self, requests: List[Dict], model_name: str) -> CostMetrics:
        """Analyze cost with OpenAI API"""
        pricing = self.openai_pricing[model_name]
        total_input_tokens = 0
        total_output_tokens = 0

        for req in requests:
            input_tokens, output_tokens = self.classifier.estimate_tokens(req["prompt"])
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # Calculate costs (no infrastructure cost for API)
        inference_cost = (
            (total_input_tokens / 1_000_000) * pricing["input_cost_per_1m"] +
            (total_output_tokens / 1_000_000) * pricing["output_cost_per_1m"]
        )

        return CostMetrics(
            model_name=model_name,
            framework="OpenAI",
            num_requests=len(requests),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            inference_cost=inference_cost,
            infrastructure_cost=0.0,
            total_cost=inference_cost,
            cost_per_request=inference_cost / len(requests),
            cost_per_1k_tokens=(inference_cost / (total_input_tokens + total_output_tokens)) * 1000,
            avg_latency_ms=pricing["avg_latency_ms"],
        )

    def save_results(self, results: Dict[str, CostMetrics], filename: str = "results/cost_analysis.json"):
        """Save analysis results"""
        import os

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Convert to dict
        results_dict = {k: asdict(v) for k, v in results.items()}

        # Save JSON
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save CSV
        csv_filename = filename.replace('.json', '.csv')
        with open(csv_filename, 'w') as f:
            # Header
            if results:
                first_key = list(results.keys())[0]
                headers = list(asdict(results[first_key]).keys())
                f.write(','.join(headers) + '\n')

                # Rows
                for metrics in results.values():
                    row_dict = asdict(metrics)
                    f.write(','.join(str(v) for v in row_dict.values()) + '\n')

        print(f"\n✅ Results saved to {filename}")
        print(f"✅ Results saved to {csv_filename}")

    def print_summary(self, results: Dict[str, CostMetrics]):
        """Print cost analysis summary"""
        print(f"\n{'='*80}")
        print(f"Cost Analysis Summary")
        print(f"{'='*80}")

        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Total requests: {metrics.num_requests}")
            print(f"  Total tokens: {metrics.total_tokens:,} ({metrics.total_input_tokens:,} in, {metrics.total_output_tokens:,} out)")
            print(f"  Total cost: ${metrics.total_cost:.6f}")
            print(f"  Cost per request: ${metrics.cost_per_request:.6f}")
            print(f"  Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.6f}")
            print(f"  Avg latency: {metrics.avg_latency_ms:.1f}ms")

            if metrics.framework == "vLLM":
                print(f"  Inference cost: ${metrics.inference_cost:.6f}")
                print(f"  Infrastructure cost: ${metrics.infrastructure_cost:.6f}")

        # Compare savings
        print(f"\n{'='*80}")
        print(f"Cost Savings Analysis")
        print(f"{'='*80}")

        if "vLLM-Routed" in results and "OpenAI-GPT3.5" in results:
            vllm = results["vLLM-Routed"]
            openai = results["OpenAI-GPT3.5"]

            savings = openai.total_cost - vllm.total_cost
            savings_pct = (savings / openai.total_cost) * 100

            print(f"\nvLLM-Routed vs OpenAI GPT-3.5:")
            print(f"  OpenAI cost: ${openai.total_cost:.6f}")
            print(f"  vLLM cost: ${vllm.total_cost:.6f}")
            print(f"  Savings: ${savings:.6f} ({savings_pct:.1f}%)")

            # Project to scale
            daily_requests = 10000
            scale_factor = daily_requests / vllm.num_requests

            print(f"\nProjected at {daily_requests:,} requests/day:")
            print(f"  OpenAI daily: ${openai.total_cost * scale_factor:.2f}")
            print(f"  vLLM daily: ${vllm.total_cost * scale_factor:.2f}")
            print(f"  Monthly savings: ${savings * scale_factor * 30:.2f}")
            print(f"  Yearly savings: ${savings * scale_factor * 365:.2f}")


def main():
    """Run cost analysis benchmark"""
    print(f"{'='*80}")
    print(f"Cost Analysis Benchmark with Intelligent Routing")
    print(f"{'='*80}")

    if not TIKTOKEN_AVAILABLE:
        print("\n⚠️  Installing tiktoken for accurate token counting...")
        import subprocess
        subprocess.check_call(["pip", "install", "tiktoken"])
        print("✅ tiktoken installed")

    # Create sample workload
    requests = [
        # Simple queries
        {"prompt": "What is the current stock price of AAPL?", "metadata": {"priority": "normal"}},
        {"prompt": "Define market capitalization.", "metadata": {"priority": "normal"}},
        {"prompt": "List the top 5 tech stocks.", "metadata": {"priority": "normal"}},

        # Moderate complexity
        {"prompt": "Analyze the quarterly earnings report for Tesla and summarize key revenue trends, profit margins, and future guidance.", "metadata": {"priority": "normal"}},
        {"prompt": "Compare the financial performance of Amazon and Microsoft over the last 5 years, focusing on revenue growth and market share.", "metadata": {"priority": "normal"}},

        # Complex
        {"prompt": "Provide a comprehensive analysis of the Federal Reserve's monetary policy impact on tech sector valuations, including historical correlations, current market conditions, and forward-looking implications for portfolio allocation.", "metadata": {"priority": "high"}},
        {"prompt": "Evaluate the risk-adjusted returns of a diversified portfolio consisting of 60% equities, 30% bonds, and 10% alternatives, considering various economic scenarios including recession, stagflation, and sustained growth.", "metadata": {"priority": "high"}},

        # Critical
        {"prompt": "URGENT: Analyze this breaking news about bank failures and provide immediate risk assessment for our financial services portfolio with specific recommendations for risk mitigation and capital preservation.", "metadata": {"priority": "critical", "sla_ms": 500}},
    ]

    # Duplicate to create larger workload
    requests = requests * 25  # 200 total requests

    # Run analysis
    analyzer = CostAnalyzer()
    results = analyzer.analyze_workload(requests, use_routing=True)

    # Print and save
    analyzer.print_summary(results)
    analyzer.save_results(results, "results/cost_analysis.json")

    print(f"\n{'='*80}")
    print(f"✅ Cost Analysis Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
