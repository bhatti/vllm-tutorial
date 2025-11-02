#!/usr/bin/env python3
"""
Example 3: Production Observability and Monitoring for vLLM
Demonstrates comprehensive monitoring, metrics, and alerting
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    error: Optional[str] = None


@dataclass
class Alert:
    """System alert"""
    severity: str  # info, warning, error, critical
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """
    Collect and aggregate metrics for vLLM deployment
    Prometheus-compatible metrics
    """

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent requests to keep for rolling metrics
        """
        self.window_size = window_size

        # Request history (sliding window)
        self.request_history: deque = deque(maxlen=window_size)

        # Counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Per-model metrics
        self.model_requests = defaultdict(int)
        self.model_latencies = defaultdict(list)
        self.model_costs = defaultdict(list)

        # Latency buckets (for histograms)
        self.latency_buckets = {
            "p50": [],
            "p95": [],
            "p99": []
        }

        # Time-series data (last hour, 1-minute buckets)
        self.time_series_requests = defaultdict(int)
        self.time_series_errors = defaultdict(int)

        # Lock for thread safety
        self.lock = threading.Lock()

        logger.info("MetricsCollector initialized")

    def record_request(self, metrics: RequestMetrics):
        """Record metrics for a request"""
        with self.lock:
            self.request_history.append(metrics)
            self.total_requests += 1

            if metrics.success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            self.total_tokens += metrics.tokens_generated
            self.total_cost += metrics.cost

            # Per-model metrics
            self.model_requests[metrics.model_name] += 1
            self.model_latencies[metrics.model_name].append(metrics.latency_ms)
            self.model_costs[metrics.model_name].append(metrics.cost)

            # Time series (bucket by minute)
            minute_bucket = metrics.timestamp.replace(second=0, microsecond=0)
            self.time_series_requests[minute_bucket] += 1
            if not metrics.success:
                self.time_series_errors[minute_bucket] += 1

    def get_error_rate(self, window_minutes: int = 5) -> float:
        """Calculate error rate over time window"""
        with self.lock:
            if self.total_requests == 0:
                return 0.0

            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            recent_requests = [
                r for r in self.request_history
                if r.timestamp >= cutoff_time
            ]

            if not recent_requests:
                return 0.0

            errors = sum(1 for r in recent_requests if not r.success)
            return errors / len(recent_requests)

    def get_latency_percentiles(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """Calculate latency percentiles"""
        with self.lock:
            if model_name:
                latencies = self.model_latencies.get(model_name, [])
            else:
                latencies = [r.latency_ms for r in self.request_history]

            if not latencies:
                return {"p50": 0, "p95": 0, "p99": 0}

            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)

            return {
                "p50": sorted_latencies[int(n * 0.50)] if n > 0 else 0,
                "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            }

    def get_throughput(self, window_seconds: int = 60) -> float:
        """Calculate requests per second"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)
            recent_requests = sum(
                1 for r in self.request_history
                if r.timestamp >= cutoff_time
            )
            return recent_requests / window_seconds

    def get_tokens_per_second(self, window_seconds: int = 60) -> float:
        """Calculate token generation throughput"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)
            recent_tokens = sum(
                r.tokens_generated for r in self.request_history
                if r.timestamp >= cutoff_time
            )
            return recent_tokens / window_seconds

    def get_cost_metrics(self) -> Dict:
        """Get cost-related metrics"""
        with self.lock:
            avg_cost = self.total_cost / max(1, self.total_requests)

            # Cost per model
            model_costs = {}
            for model, costs in self.model_costs.items():
                if costs:
                    model_costs[model] = {
                        "total": sum(costs),
                        "average": sum(costs) / len(costs),
                        "requests": len(costs)
                    }

            return {
                "total_cost": self.total_cost,
                "average_cost_per_request": avg_cost,
                "total_requests": self.total_requests,
                "cost_per_model": model_costs
            }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        with self.lock:
            metrics = []

            # Request counters
            metrics.append("# HELP vllm_requests_total Total number of requests")
            metrics.append("# TYPE vllm_requests_total counter")
            metrics.append(f"vllm_requests_total {self.total_requests}")

            metrics.append("# HELP vllm_requests_success Successful requests")
            metrics.append("# TYPE vllm_requests_success counter")
            metrics.append(f"vllm_requests_success {self.successful_requests}")

            metrics.append("# HELP vllm_requests_failed Failed requests")
            metrics.append("# TYPE vllm_requests_failed counter")
            metrics.append(f"vllm_requests_failed {self.failed_requests}")

            # Latency percentiles
            percentiles = self.get_latency_percentiles()
            metrics.append("# HELP vllm_latency_ms Request latency in milliseconds")
            metrics.append("# TYPE vllm_latency_ms summary")
            for p, value in percentiles.items():
                metrics.append(f'vllm_latency_ms{{quantile="{p}"}} {value}')

            # Throughput
            throughput = self.get_throughput()
            metrics.append("# HELP vllm_throughput_rps Requests per second")
            metrics.append("# TYPE vllm_throughput_rps gauge")
            metrics.append(f"vllm_throughput_rps {throughput}")

            # Token throughput
            tokens_per_sec = self.get_tokens_per_second()
            metrics.append("# HELP vllm_tokens_per_second Token generation rate")
            metrics.append("# TYPE vllm_tokens_per_second gauge")
            metrics.append(f"vllm_tokens_per_second {tokens_per_sec}")

            # Cost metrics
            metrics.append("# HELP vllm_cost_total Total cost in USD")
            metrics.append("# TYPE vllm_cost_total counter")
            metrics.append(f"vllm_cost_total {self.total_cost}")

            # Error rate
            error_rate = self.get_error_rate()
            metrics.append("# HELP vllm_error_rate Error rate (0-1)")
            metrics.append("# TYPE vllm_error_rate gauge")
            metrics.append(f"vllm_error_rate {error_rate}")

            # Per-model metrics
            for model, count in self.model_requests.items():
                metrics.append(f'vllm_requests_per_model{{model="{model}"}} {count}')

            return "\n".join(metrics)


class AlertManager:
    """
    Manage alerts based on metrics thresholds
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)

        # Alert thresholds
        self.thresholds = {
            "error_rate": 0.05,          # 5% error rate
            "p95_latency_ms": 1000,      # 1 second
            "p99_latency_ms": 2000,      # 2 seconds
            "throughput_rps": 1.0,       # Min 1 req/sec (health check)
            "cost_per_request": 0.10,    # $0.10 per request
        }

        logger.info("AlertManager initialized")

    def check_alerts(self) -> List[Alert]:
        """Check all alert conditions"""
        new_alerts = []

        # Error rate alert
        error_rate = self.metrics.get_error_rate()
        if error_rate > self.thresholds["error_rate"]:
            alert = Alert(
                severity="critical" if error_rate > 0.10 else "warning",
                message=f"High error rate: {error_rate*100:.1f}%",
                metric="error_rate",
                value=error_rate,
                threshold=self.thresholds["error_rate"]
            )
            new_alerts.append(alert)

        # Latency alerts
        percentiles = self.metrics.get_latency_percentiles()

        if percentiles["p95"] > self.thresholds["p95_latency_ms"]:
            alert = Alert(
                severity="warning",
                message=f"High P95 latency: {percentiles['p95']:.0f}ms",
                metric="p95_latency",
                value=percentiles["p95"],
                threshold=self.thresholds["p95_latency_ms"]
            )
            new_alerts.append(alert)

        if percentiles["p99"] > self.thresholds["p99_latency_ms"]:
            alert = Alert(
                severity="critical",
                message=f"High P99 latency: {percentiles['p99']:.0f}ms",
                metric="p99_latency",
                value=percentiles["p99"],
                threshold=self.thresholds["p99_latency_ms"]
            )
            new_alerts.append(alert)

        # Throughput alert (low throughput might indicate issues)
        throughput = self.metrics.get_throughput()
        if throughput < self.thresholds["throughput_rps"] and self.metrics.total_requests > 10:
            alert = Alert(
                severity="warning",
                message=f"Low throughput: {throughput:.2f} req/sec",
                metric="throughput",
                value=throughput,
                threshold=self.thresholds["throughput_rps"]
            )
            new_alerts.append(alert)

        # Cost alert
        cost_metrics = self.metrics.get_cost_metrics()
        avg_cost = cost_metrics["average_cost_per_request"]
        if avg_cost > self.thresholds["cost_per_request"]:
            alert = Alert(
                severity="info",
                message=f"High average cost: ${avg_cost:.4f} per request",
                metric="cost_per_request",
                value=avg_cost,
                threshold=self.thresholds["cost_per_request"]
            )
            new_alerts.append(alert)

        # Log and store alerts
        for alert in new_alerts:
            logger.log(
                logging.CRITICAL if alert.severity == "critical" else
                logging.WARNING if alert.severity == "warning" else
                logging.INFO,
                f"ALERT [{alert.severity.upper()}]: {alert.message}"
            )
            self.alert_history.append(alert)

        self.alerts = new_alerts
        return new_alerts

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return self.alerts

    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        return {
            "active_alerts": len(self.alerts),
            "alerts_by_severity": {
                "critical": sum(1 for a in self.alerts if a.severity == "critical"),
                "warning": sum(1 for a in self.alerts if a.severity == "warning"),
                "info": sum(1 for a in self.alerts if a.severity == "info"),
            },
            "recent_alerts": [
                {
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in list(self.alert_history)[-10:]
            ]
        }


def simulate_production_traffic(
    metrics_collector: MetricsCollector,
    alert_manager: AlertManager,
    duration_seconds: int = 60
):
    """Simulate production traffic for demo"""
    import random

    print(f"\n🚀 Simulating production traffic for {duration_seconds} seconds...")

    models = ["phi-2", "mistral-7b", "llama-13b"]
    start_time = time.time()
    request_count = 0

    while time.time() - start_time < duration_seconds:
        request_count += 1

        # Simulate request
        model = random.choice(models)
        success = random.random() > 0.02  # 2% error rate normally

        # Occasional spike in errors
        if request_count % 100 == 0:
            success = random.random() > 0.15  # 15% error rate during spike

        latency = random.gauss(200, 50) if success else random.gauss(500, 100)
        tokens = random.randint(50, 300) if success else 0

        # Model-specific costs
        cost_per_1k = {
            "phi-2": 0.0001,
            "mistral-7b": 0.001,
            "llama-13b": 0.005
        }
        cost = (tokens / 1000) * cost_per_1k[model]

        # Record metrics
        request_metrics = RequestMetrics(
            request_id=f"req-{request_count}",
            model_name=model,
            timestamp=datetime.utcnow(),
            latency_ms=latency,
            tokens_generated=tokens,
            cost=cost,
            success=success,
            error=None if success else "Simulated error"
        )

        metrics_collector.record_request(request_metrics)

        # Check alerts every 10 requests
        if request_count % 10 == 0:
            alert_manager.check_alerts()

        # Simulate request rate (for demo - remove sleep for instant)
        # time.sleep(0.1)  # Commented out for faster execution

    print(f"✅ Simulated {request_count} requests")


def main():
    """Demonstrate observability and monitoring"""

    print("=" * 80)
    print("Production Observability & Monitoring for vLLM")
    print("=" * 80)

    # Initialize metrics and alerts
    metrics = MetricsCollector(window_size=1000)
    alerts = AlertManager(metrics)

    # Simulate some traffic (instant - no sleep)
    simulate_production_traffic(metrics, alerts, duration_seconds=5)

    # Display metrics
    print("\n" + "="*80)
    print("📊 System Metrics")
    print("="*80)

    print(f"\n📈 Request Metrics:")
    print(f"  Total requests: {metrics.total_requests}")
    print(f"  Successful: {metrics.successful_requests}")
    print(f"  Failed: {metrics.failed_requests}")
    print(f"  Error rate: {metrics.get_error_rate()*100:.2f}%")

    print(f"\n⚡ Performance Metrics:")
    percentiles = metrics.get_latency_percentiles()
    print(f"  P50 latency: {percentiles['p50']:.1f}ms")
    print(f"  P95 latency: {percentiles['p95']:.1f}ms")
    print(f"  P99 latency: {percentiles['p99']:.1f}ms")
    print(f"  Throughput: {metrics.get_throughput():.2f} req/sec")
    print(f"  Token throughput: {metrics.get_tokens_per_second():.1f} tokens/sec")

    print(f"\n💰 Cost Metrics:")
    cost_metrics = metrics.get_cost_metrics()
    print(f"  Total cost: ${cost_metrics['total_cost']:.4f}")
    print(f"  Avg cost/request: ${cost_metrics['average_cost_per_request']:.6f}")

    print(f"\n🎯 Per-Model Metrics:")
    for model, costs in cost_metrics['cost_per_model'].items():
        print(f"  {model}:")
        print(f"    Requests: {costs['requests']}")
        print(f"    Total cost: ${costs['total']:.4f}")
        print(f"    Avg cost: ${costs['average']:.6f}")

    # Show alerts
    print("\n" + "="*80)
    print("🚨 Alerts")
    print("="*80)

    alert_summary = alerts.get_alert_summary()
    print(f"\nActive alerts: {alert_summary['active_alerts']}")
    print(f"  Critical: {alert_summary['alerts_by_severity']['critical']}")
    print(f"  Warning: {alert_summary['alerts_by_severity']['warning']}")
    print(f"  Info: {alert_summary['alerts_by_severity']['info']}")

    if alert_summary['recent_alerts']:
        print("\nRecent alerts:")
        for alert in alert_summary['recent_alerts'][-5:]:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")

    # Export Prometheus metrics
    print("\n" + "="*80)
    print("📊 Prometheus Metrics Export")
    print("="*80)
    print(metrics.get_prometheus_metrics())

    # Dashboard summary
    print("\n" + "="*80)
    print("📈 Dashboard Summary (for Grafana)")
    print("="*80)

    dashboard_data = {
        "overview": {
            "total_requests": metrics.total_requests,
            "success_rate": metrics.successful_requests / max(1, metrics.total_requests) * 100,
            "avg_latency_ms": percentiles["p50"],
            "throughput_rps": metrics.get_throughput(),
        },
        "performance": percentiles,
        "costs": cost_metrics,
        "alerts": alert_summary
    }

    print(json.dumps(dashboard_data, indent=2))

    print("\n" + "="*80)
    print("✅ Observability Example Complete!")
    print("="*80)
    print("\n💡 In production, these metrics would be:")
    print("  - Scraped by Prometheus every 15s")
    print("  - Visualized in Grafana dashboards")
    print("  - Trigger PagerDuty/Slack alerts")
    print("  - Stored in time-series database")
    print("  - Used for capacity planning")


if __name__ == "__main__":
    main()
