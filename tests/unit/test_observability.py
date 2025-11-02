"""
Unit tests for Observability and Monitoring with >90% coverage requirement
Following TDD principles: Write test first, then implementation
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import time

# Import modules to test
from src.observability_monitoring import (
    ObservabilityStack,
    LangfuseTracer,
    PhoenixMonitor,
    MetricType,
    TraceEvent,
    SpanContext,
    PerformanceMonitor,
    AlertManager,
    DashboardGenerator,
)


class TestLangfuseTracer:
    """Test suite for Langfuse tracing integration"""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Create mock Langfuse client"""
        mock = MagicMock()
        mock.trace.return_value = MagicMock(id="trace_123")
        mock.generation.return_value = MagicMock(id="gen_456")
        mock.score.return_value = MagicMock(id="score_789")
        mock.flush.return_value = None
        return mock

    @pytest.fixture
    def tracer(self, mock_langfuse_client):
        """Create tracer with mocked client"""
        with patch("src.observability_monitoring.Langfuse") as mock_class:
            mock_class.return_value = mock_langfuse_client
            tracer = LangfuseTracer(
                public_key="test_public",
                secret_key="test_secret",
                host="https://test.langfuse.com",
            )
            return tracer

    def test_trace_creation(self, tracer, mock_langfuse_client):
        """Test creating a new trace"""
        # Arrange
        trace_data = {
            "name": "test_operation",
            "user_id": "user_123",
            "session_id": "session_456",
            "metadata": {"model": "llama-13b", "temperature": 0.7},
        }

        # Act
        trace_id = tracer.create_trace(**trace_data)

        # Assert
        assert trace_id == "trace_123"
        mock_langfuse_client.trace.assert_called_once_with(
            name="test_operation",
            user_id="user_123",
            session_id="session_456",
            metadata={"model": "llama-13b", "temperature": 0.7},
        )

    def test_generation_tracking(self, tracer, mock_langfuse_client):
        """Test tracking LLM generation events"""
        # Arrange
        generation_data = {
            "trace_id": "trace_123",
            "name": "llm_generation",
            "model": "llama-13b",
            "input": "Analyze this financial report",
            "output": "The report shows strong growth...",
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
            "latency_ms": 250,
        }

        # Act
        gen_id = tracer.track_generation(**generation_data)

        # Assert
        assert gen_id == "gen_456"
        mock_langfuse_client.generation.assert_called_once()
        call_args = mock_langfuse_client.generation.call_args[1]
        assert call_args["model"] == "llama-13b"
        assert call_args["usage"]["total_tokens"] == 40

    def test_score_recording(self, tracer, mock_langfuse_client):
        """Test recording quality scores"""
        # Arrange
        score_data = {
            "trace_id": "trace_123",
            "name": "response_quality",
            "value": 0.92,
            "comment": "High quality financial analysis",
        }

        # Act
        score_id = tracer.record_score(**score_data)

        # Assert
        assert score_id == "score_789"
        mock_langfuse_client.score.assert_called_once_with(
            trace_id="trace_123",
            name="response_quality",
            value=0.92,
            comment="High quality financial analysis",
        )

    def test_batch_flush(self, tracer, mock_langfuse_client):
        """Test flushing pending events"""
        # Arrange - Create multiple events
        for i in range(5):
            tracer.create_trace(name=f"trace_{i}", user_id=f"user_{i}")

        # Act
        tracer.flush()

        # Assert
        mock_langfuse_client.flush.assert_called()
        assert mock_langfuse_client.trace.call_count == 5

    def test_error_handling(self, tracer, mock_langfuse_client):
        """Test graceful error handling"""
        # Arrange - Make client raise exception
        mock_langfuse_client.trace.side_effect = Exception("Connection failed")

        # Act & Assert - Should not raise, but log error
        trace_id = tracer.create_trace(name="failing_trace")
        assert trace_id is None  # Should return None on failure


class TestPhoenixMonitor:
    """Test suite for Phoenix (Arize) monitoring"""

    @pytest.fixture
    def mock_phoenix_client(self):
        """Create mock Phoenix client"""
        mock = MagicMock()
        mock.log_inference.return_value = {"inference_id": "inf_123"}
        mock.log_evaluation.return_value = {"evaluation_id": "eval_456"}
        return mock

    @pytest.fixture
    def monitor(self, mock_phoenix_client):
        """Create monitor with mocked client"""
        with patch("src.observability_monitoring.px") as mock_px:
            mock_px.Client.return_value = mock_phoenix_client
            monitor = PhoenixMonitor(
                api_key="test_key",
                space_id="test_space",
            )
            return monitor

    def test_inference_logging(self, monitor, mock_phoenix_client):
        """Test logging model inferences"""
        # Arrange
        inference_data = {
            "model_name": "llama-13b",
            "model_version": "v1.2.0",
            "prediction": {"sentiment": "positive", "score": 0.85},
            "features": {
                "text_length": 500,
                "complexity": "moderate",
            },
            "tags": {"environment": "production", "user_tier": "premium"},
        }

        # Act
        result = monitor.log_inference(**inference_data)

        # Assert
        assert result["inference_id"] == "inf_123"
        mock_phoenix_client.log_inference.assert_called_once()
        call_args = mock_phoenix_client.log_inference.call_args[1]
        assert call_args["model_name"] == "llama-13b"
        assert call_args["features"]["complexity"] == "moderate"

    def test_evaluation_metrics(self, monitor, mock_phoenix_client):
        """Test logging evaluation metrics"""
        # Arrange
        eval_data = {
            "model_name": "llama-13b",
            "metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.95,
                "f1_score": 0.92,
            },
            "dataset": "financial_test_set",
            "timestamp": datetime.now(),
        }

        # Act
        result = monitor.log_evaluation(**eval_data)

        # Assert
        assert result["evaluation_id"] == "eval_456"
        mock_phoenix_client.log_evaluation.assert_called_once()
        call_args = mock_phoenix_client.log_evaluation.call_args[1]
        assert call_args["metrics"]["accuracy"] == 0.92

    def test_drift_detection(self, monitor):
        """Test data drift detection"""
        # Arrange
        reference_data = {
            "text_length": [100, 200, 150, 180, 220],
            "sentiment_scores": [0.7, 0.8, 0.75, 0.82, 0.78],
        }

        current_data = {
            "text_length": [300, 400, 350, 380, 420],  # Significant drift
            "sentiment_scores": [0.3, 0.2, 0.25, 0.28, 0.22],  # Significant drift
        }

        # Act
        drift_detected = monitor.detect_drift(reference_data, current_data)

        # Assert
        assert drift_detected == True
        assert monitor.get_drift_metrics()["text_length"]["drift_score"] > 0.5
        assert monitor.get_drift_metrics()["sentiment_scores"]["drift_score"] > 0.5

    def test_model_performance_tracking(self, monitor):
        """Test tracking model performance over time"""
        # Arrange & Act
        for day in range(7):
            timestamp = datetime.now() - timedelta(days=day)
            monitor.track_performance(
                model_name="llama-13b",
                accuracy=0.90 - (day * 0.01),  # Degrading performance
                latency_ms=100 + (day * 10),  # Increasing latency
                timestamp=timestamp,
            )

        # Assert
        performance_trend = monitor.get_performance_trend("llama-13b")
        assert performance_trend["accuracy"]["trend"] == "declining"
        assert performance_trend["latency"]["trend"] == "increasing"
        assert performance_trend["accuracy"]["change_rate"] < 0


class TestPerformanceMonitor:
    """Test suite for performance monitoring"""

    @pytest.fixture
    def perf_monitor(self):
        """Create performance monitor instance"""
        return PerformanceMonitor()

    def test_latency_histogram(self, perf_monitor):
        """Test latency histogram collection"""
        # Arrange & Act
        latencies = [100, 150, 200, 120, 180, 90, 250, 110, 130, 170]
        for latency in latencies:
            perf_monitor.record_latency("model_inference", latency)

        # Assert
        stats = perf_monitor.get_latency_stats("model_inference")
        assert stats["count"] == 10
        assert 100 <= stats["mean"] <= 200
        assert stats["p50"] == 140  # Median
        assert stats["p95"] <= 250
        assert stats["p99"] <= 250

    def test_throughput_measurement(self, perf_monitor):
        """Test throughput calculation"""
        # Arrange & Act
        start_time = time.time()
        for i in range(100):
            perf_monitor.record_request("inference_endpoint")
            time.sleep(0.01)  # Simulate processing

        # Assert
        throughput = perf_monitor.get_throughput("inference_endpoint")
        assert 80 <= throughput <= 120  # ~100 requests per second

    def test_resource_utilization(self, perf_monitor):
        """Test resource utilization tracking"""
        # Arrange & Act
        perf_monitor.record_resource_usage(
            cpu_percent=75.5,
            memory_percent=82.3,
            gpu_percent=91.2,
            disk_io_mb=150,
        )

        # Assert
        resources = perf_monitor.get_resource_stats()
        assert resources["cpu"]["current"] == 75.5
        assert resources["memory"]["current"] == 82.3
        assert resources["gpu"]["current"] == 91.2
        assert resources["disk_io"]["current"] == 150

    def test_error_rate_calculation(self, perf_monitor):
        """Test error rate tracking"""
        # Arrange & Act
        for i in range(1000):
            perf_monitor.record_request("api_endpoint")
            if i % 50 == 0:  # 2% error rate
                perf_monitor.record_error("api_endpoint", "timeout")

        # Assert
        error_rate = perf_monitor.get_error_rate("api_endpoint")
        assert 0.018 <= error_rate <= 0.022  # ~2%
        error_breakdown = perf_monitor.get_error_breakdown("api_endpoint")
        assert "timeout" in error_breakdown

    def test_sla_compliance(self, perf_monitor):
        """Test SLA compliance monitoring"""
        # Arrange
        sla_config = {
            "latency_p95_ms": 200,
            "error_rate_percent": 1.0,
            "availability_percent": 99.9,
        }

        # Act - Record metrics
        for i in range(100):
            latency = 150 if i < 95 else 250  # 5% exceed SLA
            perf_monitor.record_latency("service", latency)
            if i < 99:  # 1% errors
                perf_monitor.record_request("service")
            else:
                perf_monitor.record_error("service", "failure")

        # Assert
        compliance = perf_monitor.check_sla_compliance("service", sla_config)
        assert compliance["latency_p95"]["compliant"] == False  # Exceeds 200ms
        assert compliance["error_rate"]["compliant"] == True  # 1% is within SLA


class TestAlertManager:
    """Test suite for alert management"""

    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance"""
        return AlertManager(
            alert_channels=["email", "slack"],
            thresholds={
                "error_rate": 0.05,  # 5%
                "latency_p95": 500,  # 500ms
                "cpu_percent": 90,
                "memory_percent": 85,
            },
        )

    def test_alert_triggering(self, alert_manager):
        """Test alert triggering based on thresholds"""
        # Arrange
        metrics = {
            "error_rate": 0.08,  # Exceeds 5% threshold
            "latency_p95": 450,  # Below 500ms threshold
            "cpu_percent": 95,  # Exceeds 90% threshold
            "memory_percent": 80,  # Below 85% threshold
        }

        # Act
        alerts = alert_manager.check_thresholds(metrics)

        # Assert
        assert len(alerts) == 2  # error_rate and cpu_percent
        assert any(a["metric"] == "error_rate" for a in alerts)
        assert any(a["metric"] == "cpu_percent" for a in alerts)

    def test_alert_deduplication(self, alert_manager):
        """Test that duplicate alerts are suppressed"""
        # Arrange
        metrics = {"error_rate": 0.10}  # High error rate

        # Act - Trigger same alert multiple times
        alerts1 = alert_manager.check_thresholds(metrics)
        alerts2 = alert_manager.check_thresholds(metrics)
        alerts3 = alert_manager.check_thresholds(metrics)

        # Assert
        assert len(alerts1) == 1  # First alert sent
        assert len(alerts2) == 0  # Duplicate suppressed
        assert len(alerts3) == 0  # Still suppressed

    def test_alert_recovery(self, alert_manager):
        """Test alert recovery notifications"""
        # Arrange
        high_metrics = {"error_rate": 0.10}
        normal_metrics = {"error_rate": 0.02}

        # Act
        alert_manager.check_thresholds(high_metrics)  # Trigger alert
        recovery_alerts = alert_manager.check_thresholds(normal_metrics)  # Recovery

        # Assert
        assert any(a["type"] == "recovery" for a in recovery_alerts)
        assert any("recovered" in a["message"].lower() for a in recovery_alerts)

    def test_alert_escalation(self, alert_manager):
        """Test alert escalation for persistent issues"""
        # Arrange
        high_metrics = {"error_rate": 0.15}

        # Act - Persistent high error rate
        for minute in range(10):
            alert_manager.check_thresholds(high_metrics)
            alert_manager.advance_time(60)  # Advance 1 minute

        # Assert
        escalations = alert_manager.get_escalations()
        assert "error_rate" in escalations
        assert escalations["error_rate"]["level"] >= 2  # Should escalate

    def test_custom_alert_rules(self, alert_manager):
        """Test custom alert rule evaluation"""
        # Arrange
        custom_rule = {
            "name": "high_cost_alert",
            "condition": lambda m: m.get("cost_per_hour", 0) > 100,
            "message": "Cost exceeds $100/hour",
            "severity": "critical",
        }
        alert_manager.add_custom_rule(custom_rule)

        # Act
        metrics = {"cost_per_hour": 150}
        alerts = alert_manager.check_custom_rules(metrics)

        # Assert
        assert len(alerts) == 1
        assert alerts[0]["name"] == "high_cost_alert"
        assert alerts[0]["severity"] == "critical"


class TestDashboardGenerator:
    """Test suite for dashboard generation"""

    @pytest.fixture
    def dashboard_gen(self):
        """Create dashboard generator instance"""
        return DashboardGenerator()

    def test_metric_visualization(self, dashboard_gen):
        """Test generating metric visualizations"""
        # Arrange
        metrics_data = {
            "timestamps": [datetime.now() - timedelta(hours=i) for i in range(24)],
            "latency": [100 + (i * 5) for i in range(24)],
            "throughput": [1000 - (i * 20) for i in range(24)],
            "error_rate": [0.01 + (i * 0.001) for i in range(24)],
        }

        # Act
        dashboard = dashboard_gen.create_dashboard(metrics_data)

        # Assert
        assert "latency_chart" in dashboard
        assert "throughput_chart" in dashboard
        assert "error_rate_chart" in dashboard
        assert dashboard["summary"]["avg_latency"] > 0
        assert dashboard["summary"]["total_requests"] > 0

    def test_comparison_charts(self, dashboard_gen):
        """Test generating comparison charts between models"""
        # Arrange
        model_comparisons = {
            "llama-13b": {
                "latency_p95": 250,
                "throughput": 800,
                "cost_per_1k": 0.002,
            },
            "mistral-7b": {
                "latency_p95": 150,
                "throughput": 1200,
                "cost_per_1k": 0.001,
            },
            "phi-2": {
                "latency_p95": 80,
                "throughput": 2000,
                "cost_per_1k": 0.0005,
            },
        }

        # Act
        comparison = dashboard_gen.create_model_comparison(model_comparisons)

        # Assert
        assert "latency_comparison" in comparison
        assert "throughput_comparison" in comparison
        assert "cost_comparison" in comparison
        assert comparison["best_latency"] == "phi-2"
        assert comparison["best_throughput"] == "phi-2"

    def test_export_formats(self, dashboard_gen):
        """Test exporting dashboards in different formats"""
        # Arrange
        dashboard_data = {
            "title": "vLLM Performance Dashboard",
            "metrics": {"latency": 150, "throughput": 1000},
        }

        # Act
        html_export = dashboard_gen.export_html(dashboard_data)
        json_export = dashboard_gen.export_json(dashboard_data)
        csv_export = dashboard_gen.export_csv(dashboard_data)

        # Assert
        assert "<html>" in html_export
        assert "vLLM Performance Dashboard" in html_export
        assert json.loads(json_export)["title"] == "vLLM Performance Dashboard"
        assert "latency,throughput" in csv_export


class TestObservabilityStack:
    """Test suite for complete observability stack integration"""

    @pytest.fixture
    def obs_stack(self, mock_langfuse, mock_redis):
        """Create observability stack with mocked dependencies"""
        with patch("src.observability_monitoring.Langfuse") as mock_lf:
            with patch("src.observability_monitoring.px.Client") as mock_px:
                mock_lf.return_value = mock_langfuse
                mock_px.return_value = MagicMock()

                stack = ObservabilityStack(
                    langfuse_config={
                        "public_key": "test",
                        "secret_key": "test",
                    },
                    phoenix_config={
                        "api_key": "test",
                        "space_id": "test",
                    },
                    redis_client=mock_redis,
                )
                return stack

    def test_end_to_end_tracing(self, obs_stack):
        """Test complete request tracing through the stack"""
        # Arrange
        request = {
            "id": "req_123",
            "user_id": "user_456",
            "model": "llama-13b",
            "prompt": "Analyze financial report",
        }

        # Act
        with obs_stack.trace_request(request) as trace:
            # Simulate model inference
            trace.record_event("model_start", {"model": "llama-13b"})
            time.sleep(0.1)  # Simulate processing
            trace.record_event("model_complete", {"tokens": 150})
            trace.set_output("Financial analysis complete")

        # Assert
        trace_data = obs_stack.get_trace(request["id"])
        assert trace_data["user_id"] == "user_456"
        assert trace_data["duration_ms"] >= 100
        assert len(trace_data["events"]) == 2
        assert trace_data["output"] == "Financial analysis complete"

    def test_metric_aggregation(self, obs_stack):
        """Test aggregating metrics across multiple sources"""
        # Arrange & Act
        for i in range(100):
            obs_stack.record_inference(
                model="llama-13b",
                latency_ms=100 + (i % 50),
                tokens=50 + (i % 30),
                success=i % 10 != 0,  # 10% failures
            )

        # Assert
        metrics = obs_stack.get_aggregated_metrics("llama-13b")
        assert metrics["total_inferences"] == 100
        assert metrics["success_rate"] == 0.9
        assert metrics["avg_latency_ms"] > 100
        assert metrics["total_tokens"] > 5000

    def test_distributed_tracing(self, obs_stack):
        """Test distributed tracing across services"""
        # Arrange
        parent_trace = obs_stack.create_trace("parent_operation")

        # Act - Simulate distributed calls
        with obs_stack.create_span(parent_trace, "service_a") as span_a:
            time.sleep(0.05)
            with obs_stack.create_span(span_a, "service_b") as span_b:
                time.sleep(0.05)
                span_b.set_tag("cache_hit", True)

            with obs_stack.create_span(span_a, "service_c") as span_c:
                time.sleep(0.05)
                span_c.set_tag("model", "mistral-7b")

        # Assert
        trace_tree = obs_stack.get_trace_tree(parent_trace.id)
        assert len(trace_tree["children"]) == 1  # service_a
        assert len(trace_tree["children"][0]["children"]) == 2  # service_b and c
        assert trace_tree["total_duration_ms"] >= 150

    def test_anomaly_detection(self, obs_stack):
        """Test anomaly detection in metrics"""
        # Arrange - Establish baseline
        for i in range(100):
            obs_stack.record_inference(
                model="llama-13b",
                latency_ms=100 + (i % 10),  # Normal: 100-110ms
                tokens=50,
                success=True,
            )

        # Act - Introduce anomalies
        for i in range(10):
            obs_stack.record_inference(
                model="llama-13b",
                latency_ms=500 + (i * 10),  # Anomaly: 500-590ms
                tokens=50,
                success=True,
            )

        # Assert
        anomalies = obs_stack.detect_anomalies("llama-13b")
        assert len(anomalies) > 0
        assert any(a["metric"] == "latency" for a in anomalies)
        assert any(a["severity"] == "high" for a in anomalies)


# Performance benchmarks
@pytest.mark.benchmark
class TestObservabilityPerformance:
    """Performance benchmarks for observability"""

    def test_tracing_overhead(self, benchmark):
        """Benchmark tracing overhead"""

        def traced_operation():
            with ObservabilityStack().trace_request({"id": "test"}) as trace:
                # Simulate some work
                result = sum(i * i for i in range(1000))
                trace.set_output(result)
                return result

        # Benchmark
        result = benchmark(traced_operation)

        # Assert low overhead
        assert benchmark.stats["mean"] < 0.01  # <10ms overhead
        assert result > 0

    def test_metric_collection_throughput(self, benchmark):
        """Benchmark metric collection throughput"""

        stack = ObservabilityStack()

        def collect_metrics():
            for i in range(100):
                stack.record_inference(
                    model="test",
                    latency_ms=100,
                    tokens=50,
                    success=True,
                )

        # Benchmark
        benchmark(collect_metrics)

        # Assert high throughput
        assert benchmark.stats["mean"] < 0.1  # <100ms for 100 metrics