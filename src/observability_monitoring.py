"""
Production Observability & Monitoring for vLLM with Langfuse and Phoenix
Complete implementation for tracking, debugging, and optimizing LLM applications

This module provides:
- Real-time performance monitoring
- Cost tracking and analysis
- Quality evaluation
- Trace analysis
- A/B testing support
- Anomaly detection
- User feedback integration
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import json
import uuid
from collections import deque, defaultdict
import logging
from contextlib import contextmanager

# Observability platforms
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.model import CreateTrace, CreateGeneration, CreateSpan, CreateEvent

# Phoenix for detailed tracing
import phoenix as px
from phoenix.trace import SpanContext, using_project
from phoenix.trace.langchain import LangChainInstrumentor
from phoenix.trace.llama_index import LlamaIndexInstrumentor
import opentelemetry.trace as trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation import set_span_in_context

# Metrics and monitoring
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, CollectorRegistry, push_to_gateway
import psutil
import GPUtil

# For dashboards
import plotly.graph_objects as go
import plotly.express as px_plots
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    QUALITY = "quality"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    CACHE_HIT = "cache_hit"
    MODEL_PERFORMANCE = "model_performance"


@dataclass
class TraceMetadata:
    """Metadata for each trace"""
    trace_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    feedback_score: Optional[float] = None


class ObservabilityManager:
    """
    Comprehensive observability manager for vLLM applications
    Integrates Langfuse, Phoenix, and custom metrics
    """

    def __init__(
        self,
        langfuse_config: Optional[Dict] = None,
        phoenix_config: Optional[Dict] = None,
        prometheus_config: Optional[Dict] = None,
        enable_cost_tracking: bool = True,
        enable_quality_monitoring: bool = True
    ):
        # Initialize Langfuse
        self.langfuse = None
        if langfuse_config:
            self.langfuse = Langfuse(
                public_key=langfuse_config.get("public_key"),
                secret_key=langfuse_config.get("secret_key"),
                host=langfuse_config.get("host", "https://cloud.langfuse.com"),
                flush_at=langfuse_config.get("flush_at", 10),
                flush_interval=langfuse_config.get("flush_interval", 0.5)
            )
            logger.info("Langfuse initialized")

        # Initialize Phoenix
        self.phoenix_session = None
        if phoenix_config:
            px.launch_app(port=phoenix_config.get("port", 6006))
            self.phoenix_session = px.active_session()
            logger.info(f"Phoenix launched at port {phoenix_config.get('port', 6006)}")

        # Initialize OpenTelemetry tracer
        self.tracer = trace.get_tracer(__name__)

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._init_prometheus_metrics()

        # Cost tracking
        self.enable_cost_tracking = enable_cost_tracking
        self.cost_tracker = CostTracker() if enable_cost_tracking else None

        # Quality monitoring
        self.enable_quality_monitoring = enable_quality_monitoring
        self.quality_monitor = QualityMonitor() if enable_quality_monitoring else None

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # Trace storage (for analysis)
        self.traces = deque(maxlen=10000)
        self.active_traces = {}

        # Alert thresholds
        self.alert_thresholds = {
            "latency_p95_ms": 5000,
            "error_rate": 0.05,
            "cost_per_request": 0.10,
            "tokens_per_second": 10  # Minimum expected
        }

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Counters
        self.request_counter = Counter(
            'vllm_requests_total',
            'Total number of requests',
            ['model', 'status', 'user_tier'],
            registry=self.registry
        )

        self.token_counter = Counter(
            'vllm_tokens_total',
            'Total tokens processed',
            ['model', 'token_type'],
            registry=self.registry
        )

        # Histograms
        self.latency_histogram = Histogram(
            'vllm_request_latency_seconds',
            'Request latency in seconds',
            ['model', 'operation'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )

        self.cost_histogram = Histogram(
            'vllm_request_cost_dollars',
            'Request cost in dollars',
            ['model', 'user_tier'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
            registry=self.registry
        )

        # Gauges
        self.active_requests_gauge = Gauge(
            'vllm_active_requests',
            'Number of active requests',
            ['model'],
            registry=self.registry
        )

        self.gpu_utilization_gauge = Gauge(
            'vllm_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )

        self.cache_hit_rate_gauge = Gauge(
            'vllm_cache_hit_rate',
            'Cache hit rate',
            registry=self.registry
        )

        # Summary for quality scores
        self.quality_summary = Summary(
            'vllm_response_quality',
            'Response quality scores',
            ['model'],
            registry=self.registry
        )

    @contextmanager
    def trace_request(
        self,
        operation: str,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Context manager for tracing a request

        Usage:
            with obs_manager.trace_request("generate", "llama-2-7b") as trace:
                # Your vLLM generation code
                response = llm.generate(prompt)
                trace.log_output(response)
        """
        trace_id = str(uuid.uuid4())
        start_time = time.time()

        # Create Langfuse trace
        langfuse_trace = None
        if self.langfuse:
            langfuse_trace = self.langfuse.trace(
                id=trace_id,
                name=operation,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )

        # Create Phoenix/OpenTelemetry span
        span = self.tracer.start_span(operation)
        span.set_attribute("model", model)
        span.set_attribute("user_id", user_id or "anonymous")
        span.set_attribute("trace_id", trace_id)

        # Create trace metadata
        trace_metadata = TraceMetadata(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            model=model,
            metadata=metadata or {}
        )

        self.active_traces[trace_id] = trace_metadata

        # Increment active requests
        self.active_requests_gauge.labels(model=model).inc()

        try:
            # Create trace context object
            trace_context = TraceContext(
                trace_id=trace_id,
                langfuse_trace=langfuse_trace,
                span=span,
                metadata=trace_metadata,
                start_time=start_time,
                obs_manager=self
            )

            yield trace_context

            # Success - calculate final metrics
            trace_context.finalize()

        except Exception as e:
            # Error handling
            trace_metadata.error = str(e)

            if span:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

            if langfuse_trace:
                langfuse_trace.update(
                    output={"error": str(e)},
                    level="ERROR"
                )

            self.request_counter.labels(
                model=model,
                status="error",
                user_tier=metadata.get("user_tier", "unknown") if metadata else "unknown"
            ).inc()

            raise

        finally:
            # Cleanup
            self.active_requests_gauge.labels(model=model).dec()

            if span:
                span.end()

            # Store trace for analysis
            self.traces.append(trace_metadata)

            # Flush Langfuse if needed
            if self.langfuse:
                self.langfuse.flush()

    def log_generation(
        self,
        trace_id: str,
        prompt: str,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        latency_ms: float,
        metadata: Optional[Dict] = None
    ):
        """Log a generation event"""
        if trace_id not in self.active_traces:
            logger.warning(f"Trace {trace_id} not found")
            return

        trace_metadata = self.active_traces[trace_id]
        trace_metadata.prompt_tokens = prompt_tokens
        trace_metadata.completion_tokens = completion_tokens
        trace_metadata.total_tokens = prompt_tokens + completion_tokens
        trace_metadata.latency_ms = latency_ms

        # Calculate cost
        if self.cost_tracker:
            cost = self.cost_tracker.calculate_cost(
                model, prompt_tokens, completion_tokens
            )
            trace_metadata.cost = cost

        # Update Langfuse
        if self.langfuse:
            self.langfuse.generation(
                trace_id=trace_id,
                name="generation",
                model=model,
                input=prompt,
                output=completion,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                metadata=metadata
            )

        # Update metrics
        self.token_counter.labels(model=model, token_type="prompt").inc(prompt_tokens)
        self.token_counter.labels(model=model, token_type="completion").inc(completion_tokens)
        self.latency_histogram.labels(model=model, operation="generation").observe(latency_ms / 1000)

        if trace_metadata.cost:
            self.cost_histogram.labels(
                model=model,
                user_tier=trace_metadata.metadata.get("user_tier", "unknown")
            ).observe(trace_metadata.cost)

    def log_feedback(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log user feedback for a trace"""
        if self.langfuse and trace_id in self.active_traces:
            self.langfuse.score(
                trace_id=trace_id,
                name="user_feedback",
                value=score,
                comment=comment,
                metadata=metadata
            )

            # Update trace metadata
            self.active_traces[trace_id].feedback_score = score

            # Update quality metrics
            if self.quality_monitor:
                self.quality_monitor.add_feedback(
                    self.active_traces[trace_id].model,
                    score
                )

    def get_analytics(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive analytics"""
        df = pd.DataFrame([asdict(t) for t in self.traces])

        if df.empty:
            return {"message": "No data available"}

        if time_range:
            cutoff = datetime.now() - time_range
            df = df[df['timestamp'] > cutoff]

        analytics = {
            "summary": {
                "total_requests": len(df),
                "unique_users": df['user_id'].nunique(),
                "total_tokens": df['total_tokens'].sum(),
                "total_cost": df['cost'].sum(),
                "avg_latency_ms": df['latency_ms'].mean(),
                "p95_latency_ms": df['latency_ms'].quantile(0.95),
                "error_rate": (df['error'].notna()).mean()
            },
            "by_model": df.groupby('model').agg({
                'trace_id': 'count',
                'latency_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
                'cost': 'sum',
                'total_tokens': 'sum',
                'error': lambda x: (x.notna()).mean()
            }).to_dict(),
            "by_user": df.groupby('user_id').agg({
                'trace_id': 'count',
                'cost': 'sum',
                'total_tokens': 'sum'
            }).nlargest(10, 'cost').to_dict(),
            "hourly_trend": df.set_index('timestamp').resample('H').agg({
                'trace_id': 'count',
                'latency_ms': 'mean',
                'cost': 'sum'
            }).to_dict()
        }

        # Add quality metrics if available
        if self.quality_monitor:
            analytics['quality'] = self.quality_monitor.get_metrics()

        # Add performance insights
        analytics['performance'] = self.performance_tracker.get_insights()

        return analytics

    def create_dashboard(self) -> go.Figure:
        """Create an interactive dashboard"""
        df = pd.DataFrame([asdict(t) for t in self.traces])

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper")
            return fig

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Requests Over Time',
                'Latency Distribution',
                'Cost by Model',
                'Token Usage',
                'Error Rate Trend',
                'Quality Scores'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "box"}]
            ]
        )

        # Requests over time
        hourly = df.set_index('timestamp').resample('H').size()
        fig.add_trace(
            go.Scatter(x=hourly.index, y=hourly.values, name='Requests/Hour'),
            row=1, col=1
        )

        # Latency distribution
        fig.add_trace(
            go.Histogram(x=df['latency_ms'], name='Latency (ms)', nbinsx=50),
            row=1, col=2
        )

        # Cost by model
        cost_by_model = df.groupby('model')['cost'].sum()
        fig.add_trace(
            go.Bar(x=cost_by_model.index, y=cost_by_model.values, name='Cost'),
            row=2, col=1
        )

        # Token usage over time
        token_hourly = df.set_index('timestamp').resample('H')['total_tokens'].sum()
        fig.add_trace(
            go.Scatter(x=token_hourly.index, y=token_hourly.values, name='Tokens/Hour'),
            row=2, col=2
        )

        # Error rate trend
        error_hourly = df.set_index('timestamp').resample('H')['error'].apply(
            lambda x: (x.notna()).mean() * 100
        )
        fig.add_trace(
            go.Scatter(x=error_hourly.index, y=error_hourly.values, name='Error Rate %'),
            row=3, col=1
        )

        # Quality scores distribution
        if 'feedback_score' in df.columns:
            fig.add_trace(
                go.Box(y=df['feedback_score'].dropna(), name='Quality Score'),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="vLLM Observability Dashboard",
            title_font_size=20
        )

        return fig

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        df = pd.DataFrame([asdict(t) for t in self.traces])

        if df.empty:
            return alerts

        # Recent data (last hour)
        recent = df[df['timestamp'] > datetime.now() - timedelta(hours=1)]

        if not recent.empty:
            # Check latency
            p95_latency = recent['latency_ms'].quantile(0.95)
            if p95_latency > self.alert_thresholds['latency_p95_ms']:
                alerts.append({
                    "type": "HIGH_LATENCY",
                    "severity": "warning",
                    "message": f"P95 latency ({p95_latency:.0f}ms) exceeds threshold ({self.alert_thresholds['latency_p95_ms']}ms)",
                    "value": p95_latency
                })

            # Check error rate
            error_rate = (recent['error'].notna()).mean()
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    "type": "HIGH_ERROR_RATE",
                    "severity": "critical",
                    "message": f"Error rate ({error_rate:.1%}) exceeds threshold ({self.alert_thresholds['error_rate']:.1%})",
                    "value": error_rate
                })

            # Check cost per request
            avg_cost = recent['cost'].mean()
            if avg_cost > self.alert_thresholds['cost_per_request']:
                alerts.append({
                    "type": "HIGH_COST",
                    "severity": "warning",
                    "message": f"Average cost per request (${avg_cost:.3f}) exceeds threshold (${self.alert_thresholds['cost_per_request']})",
                    "value": avg_cost
                })

        return alerts


class TraceContext:
    """Context object for active traces"""

    def __init__(
        self,
        trace_id: str,
        langfuse_trace,
        span,
        metadata: TraceMetadata,
        start_time: float,
        obs_manager: ObservabilityManager
    ):
        self.trace_id = trace_id
        self.langfuse_trace = langfuse_trace
        self.span = span
        self.metadata = metadata
        self.start_time = start_time
        self.obs_manager = obs_manager
        self.events = []

    def log_event(self, name: str, data: Dict[str, Any]):
        """Log an event within the trace"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "data": data
        })

        if self.langfuse_trace:
            self.langfuse_trace.event(
                name=name,
                metadata=data
            )

        if self.span:
            self.span.add_event(name, attributes=data)

    def log_output(
        self,
        output: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[Dict] = None
    ):
        """Log the output of the generation"""
        latency_ms = (time.time() - self.start_time) * 1000

        self.obs_manager.log_generation(
            trace_id=self.trace_id,
            prompt="",  # Should be passed from the actual prompt
            completion=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.metadata.model,
            latency_ms=latency_ms,
            metadata=metadata
        )

    def finalize(self):
        """Finalize the trace with success"""
        latency_ms = (time.time() - self.start_time) * 1000
        self.metadata.latency_ms = latency_ms

        if self.span:
            self.span.set_status(Status(StatusCode.OK))

        if self.langfuse_trace:
            self.langfuse_trace.update(
                output={"status": "success", "events": self.events}
            )

        # Log successful request
        self.obs_manager.request_counter.labels(
            model=self.metadata.model,
            status="success",
            user_tier=self.metadata.metadata.get("user_tier", "unknown")
        ).inc()


class CostTracker:
    """Track and optimize costs across models"""

    def __init__(self):
        # Cost per 1K tokens for different models
        self.model_costs = {
            # vLLM self-hosted (including infrastructure)
            "microsoft/phi-2": 0.0001,
            "mistralai/Mistral-7B-Instruct-v0.2": 0.0002,
            "meta-llama/Llama-2-7b-chat-hf": 0.0002,
            "meta-llama/Llama-2-13b-chat-hf": 0.0005,
            "meta-llama/Llama-2-70b-chat-hf": 0.002,

            # External APIs
            "gpt-3.5-turbo": 0.0015,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "claude-3-sonnet": 0.003,
            "claude-3-opus": 0.015
        }

        self.daily_costs = defaultdict(float)
        self.user_costs = defaultdict(float)

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost for a request"""
        cost_per_1k = self.model_costs.get(model, 0.001)

        # Some models have different costs for input/output
        if model.startswith("gpt-4"):
            prompt_cost = prompt_tokens / 1000 * cost_per_1k
            completion_cost = completion_tokens / 1000 * cost_per_1k * 2  # Output is 2x
            total_cost = prompt_cost + completion_cost
        else:
            total_tokens = prompt_tokens + completion_tokens
            total_cost = total_tokens / 1000 * cost_per_1k

        # Track daily costs
        today = datetime.now().date()
        self.daily_costs[today] += total_cost

        return total_cost

    def get_cost_report(self, days: int = 30) -> Dict[str, Any]:
        """Get cost report for the specified period"""
        cutoff = datetime.now().date() - timedelta(days=days)

        recent_costs = {
            date: cost
            for date, cost in self.daily_costs.items()
            if date >= cutoff
        }

        return {
            "total_cost": sum(recent_costs.values()),
            "average_daily_cost": np.mean(list(recent_costs.values())) if recent_costs else 0,
            "peak_day": max(recent_costs.items(), key=lambda x: x[1]) if recent_costs else None,
            "daily_breakdown": recent_costs,
            "projected_monthly": sum(recent_costs.values()) / len(recent_costs) * 30 if recent_costs else 0
        }


class QualityMonitor:
    """Monitor and evaluate response quality"""

    def __init__(self):
        self.feedback_scores = defaultdict(list)
        self.quality_metrics = defaultdict(list)

    def add_feedback(self, model: str, score: float):
        """Add user feedback score"""
        self.feedback_scores[model].append({
            "score": score,
            "timestamp": datetime.now()
        })

    def evaluate_response(
        self,
        prompt: str,
        response: str,
        model: str,
        expected_format: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Evaluate response quality"""
        metrics = {}

        # Length appropriateness
        prompt_length = len(prompt.split())
        response_length = len(response.split())
        length_ratio = response_length / max(prompt_length, 1)

        if length_ratio < 0.5:
            metrics["length_score"] = 0.5  # Too short
        elif length_ratio > 10:
            metrics["length_score"] = 0.7  # Maybe too verbose
        else:
            metrics["length_score"] = 1.0  # Appropriate length

        # Check for common quality issues
        issues = 0
        if response.count("I don't know") > 2:
            issues += 1
        if response.count("...") > 5:
            issues += 1
        if len(response.strip()) < 10:
            issues += 2

        metrics["quality_score"] = max(0, 1 - (issues * 0.25))

        # Format compliance (if expected format provided)
        if expected_format:
            format_score = self._check_format_compliance(response, expected_format)
            metrics["format_score"] = format_score

        # Overall score
        metrics["overall_score"] = np.mean(list(metrics.values()))

        self.quality_metrics[model].append({
            "metrics": metrics,
            "timestamp": datetime.now()
        })

        return metrics

    def _check_format_compliance(
        self,
        response: str,
        expected_format: Dict
    ) -> float:
        """Check if response matches expected format"""
        score = 1.0

        if "required_keys" in expected_format:
            for key in expected_format["required_keys"]:
                if key not in response:
                    score -= 0.2

        if "max_length" in expected_format:
            if len(response) > expected_format["max_length"]:
                score -= 0.1

        return max(0, score)

    def get_metrics(self) -> Dict[str, Any]:
        """Get quality metrics summary"""
        summary = {}

        for model, scores in self.feedback_scores.items():
            recent_scores = [s["score"] for s in scores[-100:]]  # Last 100
            if recent_scores:
                summary[model] = {
                    "avg_feedback": np.mean(recent_scores),
                    "feedback_std": np.std(recent_scores),
                    "feedback_count": len(recent_scores)
                }

        for model, metrics in self.quality_metrics.items():
            recent_metrics = metrics[-100:]  # Last 100
            if recent_metrics:
                if model not in summary:
                    summary[model] = {}

                all_scores = [m["metrics"]["overall_score"] for m in recent_metrics]
                summary[model]["avg_quality"] = np.mean(all_scores)
                summary[model]["quality_std"] = np.std(all_scores)

        return summary


class PerformanceTracker:
    """Track system performance metrics"""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.start_time = time.time()

    def collect_metrics(self):
        """Collect current system metrics"""
        metrics = {
            "timestamp": datetime.now(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict()
        }

        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics["gpu"] = [{
                    "id": gpu.id,
                    "name": gpu.name,
                    "utilization": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                } for gpu in gpus]
        except:
            pass

        self.metrics_history.append(metrics)

        return metrics

    def get_insights(self) -> Dict[str, Any]:
        """Get performance insights"""
        if not self.metrics_history:
            return {}

        df = pd.DataFrame(list(self.metrics_history))

        insights = {
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "avg_cpu_percent": df["cpu_percent"].mean(),
            "peak_cpu_percent": df["cpu_percent"].max(),
            "avg_memory_percent": df["memory_percent"].mean(),
            "peak_memory_percent": df["memory_percent"].max()
        }

        # GPU insights
        if "gpu" in df.columns and not df["gpu"].empty:
            gpu_data = []
            for gpus in df["gpu"]:
                if gpus:
                    gpu_data.extend(gpus)

            if gpu_data:
                gpu_df = pd.DataFrame(gpu_data)
                insights["avg_gpu_utilization"] = gpu_df["utilization"].mean()
                insights["peak_gpu_utilization"] = gpu_df["utilization"].max()
                insights["avg_gpu_memory_gb"] = gpu_df["memory_used"].mean() / 1024
                insights["peak_gpu_temp"] = gpu_df["temperature"].max()

        return insights


# Example usage
def demonstrate_observability():
    """Demonstrate the observability system"""

    # Initialize observability manager
    obs_manager = ObservabilityManager(
        langfuse_config={
            "public_key": "your-public-key",
            "secret_key": "your-secret-key",
            "host": "https://cloud.langfuse.com"
        },
        phoenix_config={
            "port": 6006
        }
    )

    # Simulate some requests
    import random

    models = ["phi-2", "mistral-7b", "llama-2-13b"]
    users = ["user_123", "user_456", "user_789"]

    for i in range(10):
        model = random.choice(models)
        user = random.choice(users)

        # Trace a request
        with obs_manager.trace_request(
            operation="generate",
            model=model,
            user_id=user,
            metadata={"user_tier": "premium" if i % 3 == 0 else "basic"}
        ) as trace:
            # Simulate processing
            time.sleep(random.uniform(0.1, 0.5))

            # Log events
            trace.log_event("preprocessing", {"step": "tokenization"})
            trace.log_event("inference", {"batch_size": 1})

            # Log output
            trace.log_output(
                output=f"Sample response {i}",
                prompt_tokens=random.randint(50, 200),
                completion_tokens=random.randint(100, 500)
            )

        # Simulate some feedback
        if i % 2 == 0:
            obs_manager.log_feedback(
                trace_id=trace.trace_id,
                score=random.uniform(0.6, 1.0),
                comment="Good response"
            )

    # Get analytics
    analytics = obs_manager.get_analytics()
    print("Analytics:", json.dumps(analytics, indent=2, default=str))

    # Check alerts
    alerts = obs_manager.check_alerts()
    if alerts:
        print("Alerts:", alerts)

    # Create dashboard
    fig = obs_manager.create_dashboard()
    # fig.show()  # Uncomment to display dashboard


if __name__ == "__main__":
    print("vLLM Observability & Monitoring System")
    print("=" * 60)
    print("Features:")
    print("- Langfuse integration for tracing")
    print("- Phoenix integration for detailed analysis")
    print("- Prometheus metrics")
    print("- Cost tracking")
    print("- Quality monitoring")
    print("- Performance insights")
    print("- Alert system")
    print("")

    # Run demonstration
    demonstrate_observability()