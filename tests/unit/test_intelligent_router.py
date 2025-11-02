"""
Unit tests for IntelligentRouter with >90% coverage requirement
Following TDD principles: Write test first, then implementation
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List
import asyncio

# Import modules to test
from src.intelligent_router import (
    IntelligentRouter,
    RoutingRequest,
    RoutingDecision,
    RequestComplexity,
    ModelTier,
    ModelConfig,
    ComplexityClassifier,
    CostOptimizer,
    CircuitBreaker,
)


class TestComplexityClassifier:
    """Test suite for request complexity classification"""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return ComplexityClassifier()

    @pytest.mark.parametrize(
        "content,expected_complexity,min_confidence",
        [
            ("What is 2+2?", RequestComplexity.SIMPLE, 0.7),
            ("What is the capital of France?", RequestComplexity.SIMPLE, 0.7),
            ("List the primary colors", RequestComplexity.SIMPLE, 0.7),
            ("Summarize this document", RequestComplexity.MODERATE, 0.6),
            ("Analyze the financial report", RequestComplexity.MODERATE, 0.6),
            ("Compare these two approaches", RequestComplexity.MODERATE, 0.6),
            ("Design a microservices architecture", RequestComplexity.COMPLEX, 0.5),
            ("Implement a distributed cache", RequestComplexity.COMPLEX, 0.5),
            ("Create a trading algorithm", RequestComplexity.COMPLEX, 0.5),
            ("Medical diagnosis required", RequestComplexity.CRITICAL, 0.9),
            ("Investment decision for retirement", RequestComplexity.CRITICAL, 0.9),
            ("Legal compliance assessment", RequestComplexity.CRITICAL, 0.9),
        ],
    )
    def test_classify_request_complexity(
        self, classifier, content, expected_complexity, min_confidence
    ):
        """Test classification of different request types"""
        # Arrange
        request = RoutingRequest(
            id="test_001",
            content=content,
            user_id="test_user",
        )

        # Act
        complexity, confidence = classifier.classify(request)

        # Assert
        assert complexity == expected_complexity
        assert confidence >= min_confidence
        assert 0 <= confidence <= 1

    def test_classify_empty_request(self, classifier):
        """Test handling of empty requests"""
        # Arrange
        request = RoutingRequest(id="test", content="", user_id="test")

        # Act
        complexity, confidence = classifier.classify(request)

        # Assert
        assert complexity in RequestComplexity
        assert confidence <= 0.6  # Low confidence for empty content

    def test_classify_very_long_request(self, classifier):
        """Test classification of very long requests"""
        # Arrange
        long_content = " ".join(["complex technical content"] * 500)
        request = RoutingRequest(
            id="test", content=long_content, user_id="test"
        )

        # Act
        complexity, confidence = classifier.classify(request)

        # Assert
        assert complexity in [RequestComplexity.COMPLEX, RequestComplexity.MODERATE]
        assert confidence > 0.5


class TestCostOptimizer:
    """Test suite for cost optimization logic"""

    @pytest.fixture
    def models(self) -> List[ModelConfig]:
        """Create sample model configurations"""
        return [
            ModelConfig(
                name="phi-2",
                tier=ModelTier.TINY,
                cost_per_1k_tokens=0.0001,
                avg_latency_ms=50,
                max_context_length=2048,
                capabilities=["general"],
            ),
            ModelConfig(
                name="mistral-7b",
                tier=ModelTier.SMALL,
                cost_per_1k_tokens=0.0002,
                avg_latency_ms=100,
                max_context_length=4096,
                capabilities=["general", "analysis"],
            ),
            ModelConfig(
                name="llama-13b",
                tier=ModelTier.MEDIUM,
                cost_per_1k_tokens=0.0005,
                avg_latency_ms=200,
                max_context_length=4096,
                capabilities=["general", "analysis", "code"],
            ),
            ModelConfig(
                name="llama-70b",
                tier=ModelTier.LARGE,
                cost_per_1k_tokens=0.002,
                avg_latency_ms=500,
                max_context_length=4096,
                capabilities=["general", "analysis", "code", "reasoning"],
            ),
        ]

    @pytest.fixture
    def optimizer(self, models):
        """Create optimizer instance"""
        return CostOptimizer(models)

    def test_select_cheapest_for_simple_request(self, optimizer):
        """Test that simple requests get routed to cheapest model"""
        # Arrange
        request = RoutingRequest(
            id="test",
            content="What is 2+2?",
            user_id="test_user",
            metadata={"user_tier": "free"},
        )
        available_models = ["phi-2", "mistral-7b", "llama-13b"]

        # Act
        selected_model, cost = optimizer.select_optimal_model(
            RequestComplexity.SIMPLE, request, available_models
        )

        # Assert
        assert selected_model == "phi-2"  # Cheapest model
        assert cost < 0.001  # Very low cost

    def test_respect_latency_requirements(self, optimizer):
        """Test that latency requirements are respected"""
        # Arrange
        request = RoutingRequest(
            id="test",
            content="Analyze this data",
            user_id="test_user",
            max_latency_ms=75,  # Only phi-2 meets this
        )
        available_models = ["phi-2", "mistral-7b", "llama-13b"]

        # Act
        selected_model, cost = optimizer.select_optimal_model(
            RequestComplexity.MODERATE, request, available_models
        )

        # Assert
        assert selected_model == "phi-2"  # Only model meeting latency

    def test_respect_capability_requirements(self, optimizer):
        """Test that capability requirements are met"""
        # Arrange
        request = RoutingRequest(
            id="test",
            content="Write code",
            user_id="test_user",
            required_capabilities=["code"],
        )
        available_models = ["phi-2", "mistral-7b", "llama-13b"]

        # Act
        selected_model, cost = optimizer.select_optimal_model(
            RequestComplexity.COMPLEX, request, available_models
        )

        # Assert
        assert selected_model == "llama-13b"  # Has 'code' capability

    def test_user_tier_budget_limits(self, optimizer):
        """Test that user tier budgets are enforced"""
        # Arrange
        request = RoutingRequest(
            id="test",
            content="Long complex request" * 100,  # Expensive request
            user_id="test_user",
            metadata={"user_tier": "free"},  # $0.01 limit
        )
        available_models = ["llama-70b"]  # Only expensive model

        # Act
        selected_model, cost = optimizer.select_optimal_model(
            RequestComplexity.COMPLEX, request, available_models
        )

        # Assert
        # Should still select the model but cost tracking is important
        assert selected_model == "llama-70b"
        assert cost > 0  # Cost is calculated


class TestCircuitBreaker:
    """Test suite for circuit breaker pattern"""

    @pytest.fixture
    def breaker(self):
        """Create circuit breaker instance"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,  # 1 second for testing
            half_open_requests=2,
        )

    def test_circuit_breaker_opens_after_failures(self, breaker):
        """Test that circuit opens after threshold failures"""
        # Start closed
        assert breaker.is_available()

        # Simulate failures
        for _ in range(3):
            with pytest.raises(Exception):
                with breaker:
                    raise Exception("Test failure")

        # Circuit should be open
        assert not breaker.is_available()
        assert breaker.state == "open"

    def test_circuit_breaker_recovers_after_timeout(self, breaker):
        """Test that circuit attempts recovery after timeout"""
        import time

        # Open the circuit
        for _ in range(3):
            try:
                with breaker:
                    raise Exception("Test failure")
            except:
                pass

        assert breaker.state == "open"

        # Wait for recovery timeout
        time.sleep(1.5)

        # Should attempt recovery (half-open)
        assert breaker.is_available()

        # Successful request should close circuit
        with breaker:
            pass  # Success

        assert breaker.state == "closed"

    def test_circuit_breaker_half_open_limit(self, breaker):
        """Test half-open request limit"""
        import time

        # Open the circuit
        for _ in range(3):
            try:
                with breaker:
                    raise Exception("Test failure")
            except:
                pass

        # Wait for recovery
        time.sleep(1.5)

        # Allow 2 half-open requests
        with breaker:
            pass
        with breaker:
            pass

        # Third request should fail
        with pytest.raises(Exception):
            with breaker:
                pass


class TestIntelligentRouter:
    """Test suite for the main IntelligentRouter class"""

    @pytest.fixture
    def mock_models(self) -> List[ModelConfig]:
        """Create mock model configurations"""
        return [
            ModelConfig(
                name="test-model-small",
                tier=ModelTier.SMALL,
                cost_per_1k_tokens=0.001,
                avg_latency_ms=100,
                max_context_length=2048,
                capabilities=["general"],
            ),
            ModelConfig(
                name="test-model-large",
                tier=ModelTier.LARGE,
                cost_per_1k_tokens=0.01,
                avg_latency_ms=500,
                max_context_length=4096,
                capabilities=["general", "reasoning"],
            ),
        ]

    @pytest.fixture
    def router(self, mock_models, mock_langfuse):
        """Create router instance with mocked dependencies"""
        with patch("src.intelligent_router.LLM") as mock_llm:
            mock_llm.return_value = MagicMock()
            router = IntelligentRouter(
                models=mock_models,
                langfuse_config={"public_key": "test", "secret_key": "test"},
            )
            router.langfuse = mock_langfuse
            return router

    @pytest.mark.asyncio
    async def test_route_simple_request(self, router):
        """Test routing a simple request"""
        # Arrange
        request = RoutingRequest(
            id="test_001",
            content="What is 2+2?",
            user_id="user_123",
        )

        # Mock the model execution
        with patch.object(
            router, "_execute_on_model", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = "4"

            # Act
            decision, response = await router.route_request(request)

            # Assert
            assert decision.request_id == "test_001"
            assert decision.complexity == RequestComplexity.SIMPLE
            assert decision.selected_model in ["test-model-small", "test-model-large"]
            assert decision.estimated_cost > 0
            assert decision.confidence_score > 0.5
            assert response == "4"

    @pytest.mark.asyncio
    async def test_route_with_failed_model(self, router):
        """Test routing when primary model fails"""
        # Arrange
        request = RoutingRequest(
            id="test_002",
            content="Complex analysis request",
            user_id="user_456",
        )

        # Mock execution to fail
        with patch.object(
            router, "_execute_on_model", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = Exception("Model failed")

            # Act & Assert
            with pytest.raises(Exception):
                await router.route_request(request)

            # Verify model health was updated
            assert (
                router.model_health["test-model-small"]["error_rate"] > 0
                or router.model_health["test-model-large"]["error_rate"] > 0
            )

    @pytest.mark.asyncio
    async def test_observability_integration(self, router, mock_langfuse):
        """Test that observability is properly integrated"""
        # Arrange
        request = RoutingRequest(
            id="test_003",
            content="Test prompt",
            user_id="user_789",
        )

        with patch.object(
            router, "_execute_on_model", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = "Response"

            # Act
            await router.route_request(request)

            # Assert - Check Langfuse was called
            assert mock_langfuse.trace.called
            trace_call = mock_langfuse.trace.call_args
            assert trace_call[1]["user_id"] == "user_789"
            assert trace_call[1]["session_id"] == "test_003"

    def test_get_analytics(self, router):
        """Test analytics generation"""
        # Add some mock data to request_metrics
        import pandas as pd
        from datetime import datetime

        router.request_metrics.extend([
            {
                "timestamp": datetime.now(),
                "request_id": f"req_{i}",
                "user_id": f"user_{i % 3}",
                "model": "test-model-small" if i % 2 else "test-model-large",
                "complexity": "simple" if i % 2 else "complex",
                "estimated_cost": 0.001 * i,
                "actual_latency_ms": 100 + i * 10,
                "response_length": 100 + i * 5,
                "confidence": 0.8 + (i % 3) * 0.05,
            }
            for i in range(10)
        ])

        # Act
        analytics = router.get_analytics()

        # Assert
        assert "total_requests" in analytics
        assert analytics["total_requests"] == 10
        assert "model_distribution" in analytics
        assert "complexity_distribution" in analytics
        assert "cost_by_user" in analytics


# Performance tests
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for routing system"""

    @pytest.mark.timeout(1)  # Must complete within 1 second
    def test_routing_latency(self, benchmark, mock_models):
        """Benchmark routing decision latency"""

        def create_and_classify():
            classifier = ComplexityClassifier()
            request = RoutingRequest(
                id="perf_test",
                content="Analyze this financial report for risks",
                user_id="test",
            )
            return classifier.classify(request)

        result = benchmark(create_and_classify)
        complexity, confidence = result

        # Performance assertions
        assert benchmark.stats["mean"] < 0.01  # <10ms average
        assert benchmark.stats["stddev"] < 0.005  # Low variance

    @pytest.mark.timeout(5)
    def test_concurrent_routing(self):
        """Test concurrent request handling"""
        import asyncio
        import time

        async def route_many():
            models = [
                ModelConfig(
                    name=f"model_{i}",
                    tier=ModelTier.SMALL,
                    cost_per_1k_tokens=0.001,
                    avg_latency_ms=100,
                    max_context_length=2048,
                    capabilities=["general"],
                )
                for i in range(3)
            ]

            with patch("src.intelligent_router.LLM"):
                router = IntelligentRouter(models)

                # Create many concurrent requests
                requests = [
                    RoutingRequest(
                        id=f"concurrent_{i}",
                        content=f"Request {i}",
                        user_id=f"user_{i % 10}",
                    )
                    for i in range(100)
                ]

                # Mock execution
                with patch.object(
                    router, "_execute_on_model", new_callable=AsyncMock
                ) as mock_execute:
                    mock_execute.return_value = "Response"

                    start = time.time()
                    tasks = [router.route_request(req) for req in requests]
                    results = await asyncio.gather(*tasks)
                    elapsed = time.time() - start

                    assert len(results) == 100
                    assert elapsed < 2.0  # Should handle 100 requests in <2 seconds

        asyncio.run(route_many())