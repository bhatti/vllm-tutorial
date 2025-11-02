#!/usr/bin/env python3
"""
Test suite for FastAPI production server
Following TDD principles - tests first, then implementation
Real-world Enterprise examples included
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json


class TestHealthEndpoints:
    """Test health and status endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.api_server import app
        return TestClient(app)

    def test_health_check(self, client):
        """Test basic health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_readiness_check(self, client):
        """Test readiness check for Kubernetes"""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "models_loaded" in data
        assert "router_initialized" in data

    def test_liveness_check(self, client):
        """Test liveness check for Kubernetes"""
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True


class TestModelEndpoints:
    """Test model management endpoints"""

    @pytest.fixture
    def client(self):
        from src.api_server import app
        return TestClient(app)

    def test_list_models(self, client):
        """Test listing available models"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

        # Should have at least one model
        assert len(data["models"]) > 0

        # Check model structure
        model = data["models"][0]
        assert "name" in model
        assert "tier" in model
        assert "max_complexity" in model
        assert "capabilities" in model
        assert "status" in model

    def test_get_model_details(self, client):
        """Test getting specific model details"""
        response = client.get("/models/test-model")
        # Model might not exist yet
        assert response.status_code in [200, 404]


class TestInferenceEndpoint:
    """Test main inference endpoint with real Enterprise examples"""

    @pytest.fixture
    def client(self):
        from src.api_server import app
        return TestClient(app)

    def test_simple_inference_request(self, client):
        """Test simple inference request"""
        request_data = {
            "prompt": "What is the current interest rate?",
            "user_id": "test_user_001",
            "session_id": "session_001"
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "request_id" in data
        assert "response" in data
        assert "model_used" in data
        assert "complexity" in data
        assert "latency_ms" in data
        assert "tokens_used" in data

    def test_fintech_earnings_analysis(self, client):
        """Test earnings report analysis request"""
        request_data = {
            "prompt": "Analyze Q3 2024 earnings: Revenue $5.2B (+23% YoY), EPS $2.34 (beat by $0.12). What are the key takeaways?",
            "user_id": "analyst_001",
            "metadata": {
                "user_tier": "premium",
                "use_case": "earnings_analysis",
                "company": "TECH_CORP"
            }
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Should route to at least MODERATE complexity
        assert data["complexity"] in ["MODERATE", "COMPLEX", "CRITICAL"]
        assert data["model_used"] is not None

    def test_risk_assessment_request(self, client):
        """Test risk assessment for portfolio"""
        request_data = {
            "prompt": "var calculation for portfolio: 60% equities, 30% bonds, 10% commodities. 95% confidence level.",
            "user_id": "risk_manager_001",
            "metadata": {
                "user_tier": "enterprise",
                "use_case": "risk_assessment",
                "priority": "high"
            },
            "max_tokens": 500
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # VAR calculation should be CRITICAL when pattern matched
        assert data["complexity"] == "CRITICAL"

    def test_trading_signal_generation(self, client):
        """Test trading signal generation"""
        request_data = {
            "prompt": "[PRODUCTION] Generate trading signals for AAPL based on current market conditions",
            "user_id": "trader_001",
            "metadata": {
                "user_tier": "enterprise",
                "use_case": "trading",
                "asset": "AAPL"
            }
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Production trading should be CRITICAL
        assert data["complexity"] == "CRITICAL"
        assert "warnings" not in data or data["warnings"] == []

    def test_customer_support_query(self, client):
        """Test simple customer support query"""
        request_data = {
            "prompt": "How do I reset my password?",
            "user_id": "customer_001",
            "metadata": {
                "user_tier": "free",
                "use_case": "support"
            }
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Simple query should use SIMPLE complexity
        assert data["complexity"] == "SIMPLE"
        # Should use cheapest model
        assert "cost_estimate" in data
        assert data["cost_estimate"] < 0.001

    def test_streaming_request(self, client):
        """Test streaming response"""
        request_data = {
            "prompt": "Explain the 2008 financial crisis",
            "stream": True,
            "user_id": "student_001"
        }

        # Streaming endpoint might be different
        response = client.post("/v1/generate", json=request_data)
        assert response.status_code in [200, 501]  # 501 if not implemented yet

    def test_rate_limiting(self, client):
        """Test rate limiting for free tier"""
        request_data = {
            "prompt": "Test query",
            "user_id": "free_user_001",
            "metadata": {"user_tier": "free"}
        }

        # Make multiple requests
        for i in range(5):
            response = client.post("/v1/generate", json=request_data)
            # After certain requests, might get rate limited
            assert response.status_code in [200, 429]

    def test_input_validation(self, client):
        """Test input validation"""
        # Missing required field
        request_data = {
            "user_id": "test_001"
            # Missing prompt
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 422  # Validation error

        # Invalid max_tokens
        request_data = {
            "prompt": "Test",
            "user_id": "test_001",
            "max_tokens": -1
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 422

    def test_budget_constraint(self, client):
        """Test budget constraint handling"""
        request_data = {
            "prompt": "What is 2+2?",  # Simple prompt to ensure tiny-model
            "user_id": "budget_user_001",
            "metadata": {"user_tier": "free"},
            "max_cost": 0.00001  # Very low budget forcing tiny-model
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # With extremely low budget, should use tiny-model
        assert data["model_used"] == "tiny-model"
        assert data["cost_estimate"] <= 0.0001


class TestMetricsEndpoint:
    """Test metrics and monitoring endpoints"""

    @pytest.fixture
    def client(self):
        from src.api_server import app
        return TestClient(app)

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200

        # Check for standard Prometheus format
        content = response.text
        assert "# HELP" in content or "requests_total" in content

    def test_stats_endpoint(self, client):
        """Test statistics endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_requests" in data
        assert "requests_per_model" in data
        assert "average_latency_ms" in data
        assert "requests_per_complexity" in data


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture
    def client(self):
        from src.api_server import app
        return TestClient(app)

    def test_model_unavailable(self, client):
        """Test handling when request exceeds all model capabilities"""
        request_data = {
            "prompt": "Test query",
            "user_id": "test_001",
            "max_cost": 0.00001,  # Impossible budget
            "timeout_ms": 1  # Impossible timeout
        }

        response = client.post("/v1/generate", json=request_data)
        # Should still succeed with best effort
        assert response.status_code == 200
        data = response.json()
        # Should use cheapest/fastest model
        assert data["model_used"] == "tiny-model"

    def test_timeout_handling(self, client):
        """Test request timeout handling"""
        request_data = {
            "prompt": "Test query",
            "user_id": "test_001",
            "timeout_ms": 1  # 1ms timeout - should fail
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code in [200, 408, 504]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
