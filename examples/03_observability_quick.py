#!/usr/bin/env python3
"""
Example 3: Quick Observability Demo (No delays)
Fast version for demonstrations
"""

import time
import logging
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    model_name: str
    timestamp: datetime
    latency_ms: float
    tokens_generated: int
    cost: float
    success: bool
    error: str = None


class QuickMetricsCollector:
    """Lightweight metrics collector for demo"""

    def __init__(self):
        self.request_history = deque(maxlen=1000)
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.model_requests = defaultdict(int)

    def record_request(self, metrics: RequestMetrics):
        """Record metrics for a request"""
        self.request_history.append(metrics)
        self.total_requests += 1
        if metrics.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_tokens += metrics.tokens_generated
        self.total_cost += metrics.cost
        self.model_requests[metrics.model_name] += 1

    def get_summary(self) -> Dict:
        """Get metrics summary"""
        latencies = [r.latency_ms for r in self.request_history if r.success]
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "total_requests": self.total_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests) * 100,
            "error_rate": self.failed_requests / max(1, self.total_requests) * 100,
            "total_cost": self.total_cost,
            "avg_cost": self.total_cost / max(1, self.total_requests),
            "total_tokens": self.total_tokens,
            "p50_latency": sorted_latencies[int(n * 0.50)] if n > 0 else 0,
            "p95_latency": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            "p99_latency": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            "requests_per_model": dict(self.model_requests),
        }


def main():
    """Quick observability demo"""

    print("=" * 80)
    print("Quick Observability Demo (Instant)")
    print("=" * 80)

    # Initialize
    metrics = QuickMetricsCollector()

    # Simulate 100 requests instantly
    print("\n📊 Simulating 100 requests...")
    import random

    models = ["phi-2", "mistral-7b", "llama-13b"]

    for i in range(100):
        request = RequestMetrics(
            request_id=f"req-{i}",
            model_name=random.choice(models),
            timestamp=datetime.utcnow(),
            latency_ms=random.gauss(200, 50),
            tokens_generated=random.randint(50, 300),
            cost=random.uniform(0.0001, 0.001),
            success=random.random() > 0.02  # 2% error rate
        )
        metrics.record_request(request)

    print("✅ Simulation complete!")

    # Display results
    summary = metrics.get_summary()

    print("\n" + "="*80)
    print("📊 Metrics Summary")
    print("="*80)
    print(f"\nRequests: {summary['total_requests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Error Rate: {summary['error_rate']:.1f}%")
    print(f"\nLatency Percentiles:")
    print(f"  P50: {summary['p50_latency']:.1f}ms")
    print(f"  P95: {summary['p95_latency']:.1f}ms")
    print(f"  P99: {summary['p99_latency']:.1f}ms")
    print(f"\nCost:")
    print(f"  Total: ${summary['total_cost']:.6f}")
    print(f"  Average: ${summary['avg_cost']:.6f}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"\nPer-Model Distribution:")
    for model, count in summary['requests_per_model'].items():
        print(f"  {model}: {count} requests ({count/summary['total_requests']*100:.1f}%)")

    # Prometheus-style export
    print("\n" + "="*80)
    print("📊 Prometheus Metrics (Sample)")
    print("="*80)
    print(f"# HELP vllm_requests_total Total requests")
    print(f"vllm_requests_total {summary['total_requests']}")
    print(f"\n# HELP vllm_error_rate Error rate")
    print(f"vllm_error_rate {summary['error_rate']/100:.4f}")
    print(f"\n# HELP vllm_latency_p95 P95 latency")
    print(f"vllm_latency_p95 {summary['p95_latency']:.1f}")
    print(f"\n# HELP vllm_cost_total Total cost")
    print(f"vllm_cost_total {summary['total_cost']:.6f}")

    print("\n" + "="*80)
    print("✅ Demo Complete!")
    print("="*80)
    print("\n💡 In production, these metrics would feed into:")
    print("  - Prometheus for collection")
    print("  - Grafana for visualization")
    print("  - PagerDuty for alerts")
    print("  - CloudWatch/Stackdriver for monitoring")


if __name__ == "__main__":
    main()
