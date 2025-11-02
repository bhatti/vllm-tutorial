"""
Unit tests for API endpoints with >90% coverage requirement
Following TDD principles: Write test first, then implementation
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List
import json
from datetime import datetime
import asyncio

# Import the FastAPI app and models
from src.api.main import app, InferenceRequest, InferenceResponse, BatchRequest
from src.api.routes import health, inference, analytics, admin
from src.api.middleware import RateLimitMiddleware, AuthMiddleware, LoggingMiddleware
from src.api.models import (
    ModelInfo,
    HealthStatus,
    UserTier,
    RequestMetadata,
    ErrorResponse,
)


class TestHealthEndpoints:
    """Test suite for health check endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_basic_health_check(self, client):
        """Test basic health endpoint"""
        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "uptime" in data

    def test_detailed_health_check(self, client):
        """Test detailed health with component status"""
        # Act
        response = client.get("/health/detailed")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "database" in data["components"]
        assert "cache" in data["components"]
        assert "models" in data["components"]
        assert all(c["status"] in ["up", "down"] for c in data["components"].values())

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe endpoint"""
        # Act
        response = client.get("/health/live")

        # Assert
        assert response.status_code == 200
        assert response.json()["alive"] == True

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe endpoint"""
        # Act
        response = client.get("/health/ready")

        # Assert
        assert response.status_code in [200, 503]
        data = response.json()
        assert "ready" in data
        if not data["ready"]:
            assert "reason" in data


class TestInferenceEndpoints:
    """Test suite for model inference endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client with mocked models"""
        with patch("src.api.main.load_models") as mock_load:
            mock_load.return_value = {
                "llama-13b": MagicMock(),
                "mistral-7b": MagicMock(),
            }
            return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Create authenticated headers"""
        return {"Authorization": "Bearer test_token_123"}

    def test_single_inference(self, client, auth_headers):
        """Test single model inference request"""
        # Arrange
        request_data = {
            "prompt": "What is machine learning?",
            "model": "llama-13b",
            "max_tokens": 100,
            "temperature": 0.7,
        }

        with patch("src.api.routes.inference.process_request") as mock_process:
            mock_process.return_value = {
                "text": "Machine learning is...",
                "tokens_used": 75,
                "latency_ms": 120,
            }

            # Act
            response = client.post(
                "/v1/inference",
                json=request_data,
                headers=auth_headers,
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert "tokens_used" in data
            assert "model" in data
            assert data["model"] == "llama-13b"

    def test_streaming_inference(self, client, auth_headers):
        """Test streaming inference response"""
        # Arrange
        request_data = {
            "prompt": "Write a story",
            "model": "mistral-7b",
            "stream": True,
            "max_tokens": 200,
        }

        def mock_stream():
            for i in range(5):
                yield f"data: {json.dumps({'text': f'Part {i}', 'done': i == 4})}\n\n"

        with patch("src.api.routes.inference.stream_response") as mock_stream_fn:
            mock_stream_fn.return_value = mock_stream()

            # Act
            response = client.post(
                "/v1/inference/stream",
                json=request_data,
                headers=auth_headers,
            )

            # Assert
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    def test_batch_inference(self, client, auth_headers):
        """Test batch inference for multiple prompts"""
        # Arrange
        request_data = {
            "prompts": [
                "Question 1?",
                "Question 2?",
                "Question 3?",
            ],
            "model": "llama-13b",
            "max_tokens": 50,
        }

        with patch("src.api.routes.inference.process_batch") as mock_batch:
            mock_batch.return_value = [
                {"text": f"Answer {i}", "tokens_used": 30}
                for i in range(3)
            ]

            # Act
            response = client.post(
                "/v1/inference/batch",
                json=request_data,
                headers=auth_headers,
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "responses" in data
            assert len(data["responses"]) == 3
            assert all("text" in r for r in data["responses"])

    def test_model_routing(self, client, auth_headers):
        """Test intelligent model routing"""
        # Arrange
        request_data = {
            "prompt": "Complex financial analysis required",
            "auto_route": True,  # Enable intelligent routing
            "max_latency_ms": 200,
            "max_cost": 0.01,
        }

        with patch("src.api.routes.inference.route_request") as mock_route:
            mock_route.return_value = {
                "selected_model": "llama-13b",
                "reason": "complexity_match",
                "estimated_cost": 0.005,
                "text": "Analysis results...",
            }

            # Act
            response = client.post(
                "/v1/inference",
                json=request_data,
                headers=auth_headers,
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["model"] == "llama-13b"
            assert "routing_reason" in data["metadata"]

    def test_inference_validation(self, client, auth_headers):
        """Test request validation"""
        # Test invalid model
        response = client.post(
            "/v1/inference",
            json={"prompt": "test", "model": "invalid-model"},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "Invalid model" in response.json()["detail"]

        # Test empty prompt
        response = client.post(
            "/v1/inference",
            json={"prompt": "", "model": "llama-13b"},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

        # Test excessive max_tokens
        response = client.post(
            "/v1/inference",
            json={"prompt": "test", "model": "llama-13b", "max_tokens": 10000},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "max_tokens" in response.json()["detail"].lower()


class TestAnalyticsEndpoints:
    """Test suite for analytics and metrics endpoints"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def admin_headers(self):
        """Create admin authenticated headers"""
        return {"Authorization": "Bearer admin_token_123", "X-Admin-Key": "secret"}

    def test_get_metrics(self, client, admin_headers):
        """Test retrieving system metrics"""
        # Act
        response = client.get(
            "/v1/analytics/metrics",
            headers=admin_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "average_latency_ms" in data
        assert "error_rate" in data
        assert "models" in data

    def test_get_model_stats(self, client, admin_headers):
        """Test retrieving per-model statistics"""
        # Act
        response = client.get(
            "/v1/analytics/models/llama-13b/stats",
            headers=admin_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "total_inferences" in data
        assert "average_latency_ms" in data
        assert "token_usage" in data
        assert "cost_summary" in data

    def test_get_user_analytics(self, client, admin_headers):
        """Test retrieving user-specific analytics"""
        # Act
        response = client.get(
            "/v1/analytics/users/user_123/usage",
            headers=admin_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "total_requests" in data
        assert "total_tokens" in data
        assert "total_cost" in data
        assert "models_used" in data

    def test_time_series_metrics(self, client, admin_headers):
        """Test retrieving time series metrics"""
        # Act
        response = client.get(
            "/v1/analytics/timeseries",
            params={
                "metric": "latency",
                "interval": "1h",
                "duration": "24h",
            },
            headers=admin_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "timestamps" in data
        assert "values" in data
        assert len(data["timestamps"]) == len(data["values"])

    def test_export_analytics(self, client, admin_headers):
        """Test exporting analytics data"""
        # Act
        response = client.get(
            "/v1/analytics/export",
            params={"format": "csv", "start_date": "2024-01-01"},
            headers=admin_headers,
        )

        # Assert
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert len(response.text) > 0


class TestAdminEndpoints:
    """Test suite for admin management endpoints"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def admin_headers(self):
        return {"Authorization": "Bearer admin_token", "X-Admin-Key": "admin_secret"}

    def test_list_models(self, client, admin_headers):
        """Test listing available models"""
        # Act
        response = client.get("/v1/admin/models", headers=admin_headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        for model in data["models"]:
            assert "name" in model
            assert "status" in model
            assert "tier" in model

    def test_update_model_config(self, client, admin_headers):
        """Test updating model configuration"""
        # Arrange
        config_update = {
            "max_batch_size": 16,
            "timeout_seconds": 30,
            "temperature_default": 0.8,
        }

        # Act
        response = client.patch(
            "/v1/admin/models/llama-13b/config",
            json=config_update,
            headers=admin_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "llama-13b"
        assert data["config"]["max_batch_size"] == 16

    def test_model_enable_disable(self, client, admin_headers):
        """Test enabling/disabling models"""
        # Act - Disable model
        response = client.post(
            "/v1/admin/models/mistral-7b/disable",
            headers=admin_headers,
        )
        assert response.status_code == 200

        # Act - Re-enable model
        response = client.post(
            "/v1/admin/models/mistral-7b/enable",
            headers=admin_headers,
        )
        assert response.status_code == 200

    def test_cache_management(self, client, admin_headers):
        """Test cache management endpoints"""
        # Get cache stats
        response = client.get("/v1/admin/cache/stats", headers=admin_headers)
        assert response.status_code == 200
        assert "size" in response.json()
        assert "hit_rate" in response.json()

        # Clear cache
        response = client.post("/v1/admin/cache/clear", headers=admin_headers)
        assert response.status_code == 200
        assert response.json()["cleared"] == True

    def test_rate_limit_management(self, client, admin_headers):
        """Test rate limit configuration"""
        # Arrange
        new_limits = {
            "basic": {"requests_per_minute": 10, "tokens_per_hour": 10000},
            "premium": {"requests_per_minute": 100, "tokens_per_hour": 100000},
        }

        # Act
        response = client.put(
            "/v1/admin/rate-limits",
            json=new_limits,
            headers=admin_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["basic"]["requests_per_minute"] == 10


class TestMiddleware:
    """Test suite for API middleware"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_rate_limiting(self, client):
        """Test rate limiting middleware"""
        # Make multiple requests quickly
        responses = []
        for i in range(15):
            response = client.get("/health")
            responses.append(response)

        # Should hit rate limit
        assert any(r.status_code == 429 for r in responses[-5:])

        # Check rate limit headers
        limited_response = next(r for r in responses if r.status_code == 429)
        assert "X-RateLimit-Limit" in limited_response.headers
        assert "X-RateLimit-Remaining" in limited_response.headers
        assert "X-RateLimit-Reset" in limited_response.headers

    def test_authentication(self, client):
        """Test authentication middleware"""
        # No auth header
        response = client.post("/v1/inference", json={"prompt": "test"})
        assert response.status_code == 401

        # Invalid token
        response = client.post(
            "/v1/inference",
            json={"prompt": "test"},
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert response.status_code == 401

        # Valid token
        with patch("src.api.middleware.verify_token") as mock_verify:
            mock_verify.return_value = {"user_id": "user_123", "tier": "basic"}
            response = client.post(
                "/v1/inference",
                json={"prompt": "test", "model": "llama-13b"},
                headers={"Authorization": "Bearer valid_token"},
            )
            assert response.status_code != 401

    def test_cors_headers(self, client):
        """Test CORS middleware configuration"""
        # Preflight request
        response = client.options(
            "/v1/inference",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Assert CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers

    def test_request_logging(self, client):
        """Test request logging middleware"""
        with patch("src.api.middleware.logger") as mock_logger:
            # Make request
            response = client.get("/health")

            # Verify logging
            assert mock_logger.info.called
            log_call = mock_logger.info.call_args[0][0]
            assert "GET" in log_call
            assert "/health" in log_call
            assert "200" in log_call

    def test_error_handling(self, client):
        """Test error handling middleware"""
        # Force an error
        with patch("src.api.routes.health.get_health") as mock_health:
            mock_health.side_effect = Exception("Test error")

            response = client.get("/health")

            # Should return 500 with error details
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "request_id" in data
            assert "timestamp" in data


class TestWebSocket:
    """Test suite for WebSocket endpoints"""

    def test_websocket_inference(self):
        """Test WebSocket streaming inference"""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/v1/ws/inference") as websocket:
            # Send inference request
            websocket.send_json({
                "prompt": "Tell me a story",
                "model": "llama-13b",
                "max_tokens": 100,
            })

            # Receive streaming response
            chunks = []
            while True:
                data = websocket.receive_json()
                chunks.append(data)
                if data.get("done", False):
                    break

            # Assert
            assert len(chunks) > 0
            assert all("text" in chunk for chunk in chunks)
            assert chunks[-1]["done"] == True

    def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        client = TestClient(app)

        with client.websocket_connect("/v1/ws/inference") as websocket:
            # Send invalid request
            websocket.send_json({
                "prompt": "",  # Empty prompt
                "model": "invalid-model",
            })

            # Should receive error
            error = websocket.receive_json()
            assert "error" in error
            assert error["error"]["code"] == "INVALID_REQUEST"


# Integration tests
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for complete API flow"""

    @pytest.fixture
    def client(self):
        """Create client with full app setup"""
        return TestClient(app)

    def test_full_inference_flow(self, client):
        """Test complete inference flow from auth to response"""
        # 1. Authenticate
        auth_response = client.post(
            "/v1/auth/login",
            json={"username": "test_user", "password": "test_pass"},
        )
        assert auth_response.status_code == 200
        token = auth_response.json()["access_token"]

        # 2. Make inference request
        headers = {"Authorization": f"Bearer {token}"}
        inference_response = client.post(
            "/v1/inference",
            json={
                "prompt": "What is AI?",
                "model": "llama-13b",
                "max_tokens": 50,
            },
            headers=headers,
        )
        assert inference_response.status_code == 200

        # 3. Check analytics
        analytics_response = client.get(
            "/v1/analytics/users/test_user/usage",
            headers=headers,
        )
        assert analytics_response.status_code == 200
        assert analytics_response.json()["total_requests"] > 0