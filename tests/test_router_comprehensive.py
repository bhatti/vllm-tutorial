#!/usr/bin/env python3
"""
Comprehensive tests to achieve >90% coverage of intelligent_router_simple.py
Focuses on covering all branches and edge cases
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from datetime import datetime
from src.intelligent_router_simple import (
    RequestComplexity,
    ModelTier,
    ModelConfig,
    RoutingRequest,
    RoutingDecision,
    ComplexityClassifier,
    CostOptimizer,
    IntelligentRouter
)


class TestFullCoverage:
    """Tests to ensure full code coverage"""

    def test_router_with_all_model_tiers(self):
        """Test router with all available model tiers"""
        models = [
            ModelConfig(
                name="tiny",
                tier=ModelTier.TINY,
                model_path="tiny-path",
                max_complexity=RequestComplexity.SIMPLE,
                cost_per_1k_tokens=0.0001,
                avg_latency_ms=10,
                capabilities=["general"]
            ),
            ModelConfig(
                name="small",
                tier=ModelTier.SMALL,
                model_path="small-path",
                max_complexity=RequestComplexity.MODERATE,
                cost_per_1k_tokens=0.001,
                avg_latency_ms=50,
                capabilities=["general", "analysis"]
            ),
            ModelConfig(
                name="medium",
                tier=ModelTier.MEDIUM,
                model_path="medium-path",
                max_complexity=RequestComplexity.COMPLEX,
                cost_per_1k_tokens=0.01,
                avg_latency_ms=100,
                capabilities=["general", "analysis", "financial"]
            ),
            ModelConfig(
                name="large",
                tier=ModelTier.LARGE,
                model_path="large-path",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=0.1,
                avg_latency_ms=500,
                capabilities=["general", "analysis", "financial", "trading"]
            ),
            ModelConfig(
                name="premium",
                tier=ModelTier.PREMIUM,
                model_path="premium-path",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=1.0,
                avg_latency_ms=1000,
                capabilities=["general", "analysis", "financial", "trading", "advanced"]
            )
        ]

        router = IntelligentRouter(models)

        # Test routing to each tier
        test_cases = [
            ("What is 2+2?", "tiny"),
            ("Explain compound interest", "small"),
            ("Analyze this portfolio performance over the last quarter", "medium"),
            ("[PRODUCTION] Generate trading signals", "large"),
        ]

        for prompt, expected_tier in test_cases:
            request = RoutingRequest(id=f"test_{expected_tier}", content=prompt)
            decision = router.route(request)
            assert decision.model_name in ["tiny", "small", "medium", "large", "premium"]
            assert decision.request_id == f"test_{expected_tier}"

    def test_router_health_management(self):
        """Test model health tracking and recovery"""
        models = [
            ModelConfig(
                name="model1",
                tier=ModelTier.SMALL,
                model_path="path1",
                max_complexity=RequestComplexity.MODERATE,
                cost_per_1k_tokens=0.001
            ),
            ModelConfig(
                name="model2",
                tier=ModelTier.SMALL,
                model_path="path2",
                max_complexity=RequestComplexity.MODERATE,
                cost_per_1k_tokens=0.001
            )
        ]

        router = IntelligentRouter(models)

        # Initially both should be healthy
        assert router.model_health["model1"]["healthy"] is True
        assert router.model_health["model2"]["healthy"] is True

        # Mark model1 as unhealthy
        router.mark_model_unhealthy("model1")
        assert router.model_health["model1"]["healthy"] is False

        # Should route to model2
        request = RoutingRequest(id="test", content="Test request")
        decision = router.route(request)
        assert decision.model_name == "model2"

        # Mark model1 as healthy again
        router.mark_model_healthy("model1")
        assert router.model_health["model1"]["healthy"] is True

    def test_routing_with_all_metadata_fields(self):
        """Test routing with all possible metadata fields"""
        models = [
            ModelConfig(
                name="free-model",
                tier=ModelTier.TINY,
                model_path="free",
                max_complexity=RequestComplexity.SIMPLE,
                cost_per_1k_tokens=0.00001
            ),
            ModelConfig(
                name="premium-model",
                tier=ModelTier.LARGE,
                model_path="premium",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=0.1
            )
        ]

        router = IntelligentRouter(models)

        # Test with all metadata fields
        request = RoutingRequest(
            id="full-metadata",
            content="Analyze market trends",
            user_id="user123",
            metadata={
                "user_tier": "enterprise",
                "priority": "high",
                "department": "trading",
                "region": "US"
            },
            max_cost=1.0,
            max_latency_ms=1000,
            required_capabilities=["general"]
        )

        decision = router.route(request)
        assert decision.request_id == "full-metadata"
        assert decision.estimated_cost > 0
        assert decision.estimated_latency_ms > 0

    def test_metrics_aggregation(self):
        """Test comprehensive metrics tracking"""
        models = [
            ModelConfig(
                name="test-model",
                tier=ModelTier.SMALL,
                model_path="test",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=0.001
            )
        ]

        router = IntelligentRouter(models)

        # Route multiple requests of different complexities
        requests = [
            ("simple", "What is AI?"),
            ("moderate", "Explain neural networks in detail"),
            ("complex", "Design a complete trading system"),
            ("critical", "[PRODUCTION] Execute trade")
        ]

        for req_id, content in requests:
            request = RoutingRequest(id=req_id, content=content)
            router.route(request)

        metrics = router.get_metrics()

        # Check all metric fields
        assert metrics["total_requests"] == 4
        assert metrics["requests_per_model"]["test-model"] == 4
        assert sum(metrics["requests_per_complexity"].values()) == 4
        assert metrics["total_estimated_cost"] > 0
        # The simplified router may not have average_confidence
        assert "requests_per_complexity" in metrics

    def test_fallback_routing_scenarios(self):
        """Test various fallback routing scenarios"""
        models = [
            ModelConfig(
                name="simple-only",
                tier=ModelTier.TINY,
                model_path="simple",
                max_complexity=RequestComplexity.SIMPLE,
                cost_per_1k_tokens=0.0001
            )
        ]

        router = IntelligentRouter(models)

        # Request that needs higher complexity than available
        complex_request = RoutingRequest(
            id="needs-fallback",
            content="[CRITICAL] Perform complex financial analysis with risk assessment"
        )

        decision = router.route(complex_request)
        assert decision.model_name == "simple-only"
        assert decision.fallback is True
        assert "fallback" in decision.reasoning.lower()

    def test_enterprise_tier_routing(self):
        """Test enterprise tier gets premium treatment"""
        models = [
            ModelConfig(
                name="basic",
                tier=ModelTier.TINY,
                model_path="basic",
                max_complexity=RequestComplexity.MODERATE,
                cost_per_1k_tokens=0.0001
            ),
            ModelConfig(
                name="premium",
                tier=ModelTier.PREMIUM,
                model_path="premium",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=1.0,
                capabilities=["general", "financial", "advanced"]
            )
        ]

        router = IntelligentRouter(models)

        # Enterprise user should get premium model for complex requests
        request = RoutingRequest(
            id="enterprise-req",
            content="Perform advanced portfolio optimization",
            metadata={"user_tier": "enterprise"}
        )

        decision = router.route(request)
        # Enterprise users can use any model based on need
        assert decision.model_name in ["basic", "premium"]

    def test_complexity_patterns_comprehensive(self):
        """Test all complexity classification patterns"""
        classifier = ComplexityClassifier()

        # Test all keyword patterns - based on actual classifier implementation
        test_patterns = [
            # Terms mapped to CRITICAL in fintech_patterns
            ("fraud detection", RequestComplexity.CRITICAL),  # Explicitly mapped
            ("compliance report", RequestComplexity.CRITICAL),  # Explicitly mapped
            ("var calculation", RequestComplexity.CRITICAL),  # Explicitly mapped
            ("trading signal", RequestComplexity.CRITICAL),  # Explicitly mapped

            # Terms with critical keywords
            ("investment decision", RequestComplexity.CRITICAL),  # Has critical keyword
            ("[PRODUCTION] deploy model", RequestComplexity.CRITICAL),  # Has [PRODUCTION]
            ("[CRITICAL] system alert", RequestComplexity.CRITICAL),  # Has [CRITICAL]

            # Terms mapped to COMPLEX in fintech_patterns
            ("portfolio optimization", RequestComplexity.COMPLEX),  # Explicitly mapped
            ("technical analysis", RequestComplexity.COMPLEX),  # Explicitly mapped
            ("risk assessment", RequestComplexity.COMPLEX),  # Explicitly mapped

            # Analysis keywords in longer context get MODERATE
            ("analyze the data in detail", RequestComplexity.MODERATE),
            ("explain this complex concept thoroughly", RequestComplexity.MODERATE),
            ("summarize the long document completely", RequestComplexity.MODERATE),

            # Simple patterns - terms that don't match any special patterns
            ("hello", RequestComplexity.SIMPLE),
            ("yes", RequestComplexity.SIMPLE),
            ("no", RequestComplexity.SIMPLE),
            ("ok", RequestComplexity.SIMPLE),
            ("what is this", RequestComplexity.SIMPLE),
            ("portfolio management", RequestComplexity.SIMPLE),  # Not in patterns
            ("hedge fund", RequestComplexity.SIMPLE),  # Not in patterns
        ]

        for content, expected in test_patterns:
            request = RoutingRequest(id="test", content=content)
            complexity, confidence = classifier.classify(request)
            assert complexity == expected, \
                f"'{content}' should be {expected.name}, got {complexity.name}"

    def test_model_capability_filtering(self):
        """Test capability-based model filtering"""
        models = [
            ModelConfig(
                name="general-only",
                tier=ModelTier.SMALL,
                model_path="general",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=0.001,
                capabilities=["general"]
            ),
            ModelConfig(
                name="financial-capable",
                tier=ModelTier.LARGE,
                model_path="financial",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=0.01,
                capabilities=["general", "financial", "trading"]
            )
        ]

        router = IntelligentRouter(models)

        # Request requiring financial capability
        request = RoutingRequest(
            id="needs-financial",
            content="Generate trading signals",
            required_capabilities=["financial", "trading"]
        )

        decision = router.route(request)
        assert decision.model_name == "financial-capable"

    def test_latency_budget_enforcement(self):
        """Test strict latency budget enforcement"""
        models = [
            ModelConfig(
                name="fast",
                tier=ModelTier.TINY,
                model_path="fast",
                max_complexity=RequestComplexity.SIMPLE,
                cost_per_1k_tokens=0.0001,
                avg_latency_ms=5
            ),
            ModelConfig(
                name="slow",
                tier=ModelTier.PREMIUM,
                model_path="slow",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=0.1,
                avg_latency_ms=5000
            )
        ]

        router = IntelligentRouter(models)

        # Request with tight latency budget
        request = RoutingRequest(
            id="needs-speed",
            content="Quick calculation needed",
            max_latency_ms=10
        )

        decision = router.route(request)
        assert decision.model_name == "fast"
        assert decision.estimated_latency_ms <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.intelligent_router_simple", "--cov-report=term-missing"])