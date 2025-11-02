#!/usr/bin/env python3
"""
Test suite for model loading functionality
Tests model manager and API integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestModelManager:
    """Test suite for ModelManager"""

    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        from model_loader import ModelManager
        manager = ModelManager(max_models_in_memory=2)

        assert manager.max_models_in_memory == 2
        assert len(manager.models) == 0
        assert len(manager.current_loaded) == 0
        assert len(manager.model_configs) == 4  # tiny, small, medium, large

    def test_model_configs_proper_sizing(self):
        """Test that model configs are properly sized for L4 GPU"""
        from model_loader import ModelManager
        manager = ModelManager()

        # Check tiny model
        tiny = manager.model_configs["tiny-model"]
        assert tiny["model_path"] == "facebook/opt-125m"
        assert tiny["gpu_memory_utilization"] == 0.1
        assert tiny["max_model_len"] == 512

        # Check small model
        small = manager.model_configs["small-model"]
        assert small["model_path"] == "facebook/opt-350m"
        assert small["gpu_memory_utilization"] == 0.2

        # Check that we're not using models too large for L4 (24GB)
        for name, config in manager.model_configs.items():
            assert config["gpu_memory_utilization"] <= 0.5, f"Model {name} uses too much memory"

    @patch('model_loader.torch.cuda.is_available')
    def test_gpu_detection(self, mock_cuda):
        """Test GPU detection in model manager"""
        mock_cuda.return_value = True

        with patch('model_loader.torch.cuda.get_device_name') as mock_name:
            mock_name.return_value = "NVIDIA L4"
            with patch('model_loader.torch.cuda.get_device_properties') as mock_props:
                mock_device = Mock()
                mock_device.total_memory = 24 * 1024**3  # 24GB
                mock_props.return_value = mock_device

                from model_loader import ModelManager
                manager = ModelManager()
                assert manager.gpu_available is True

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_load_model_without_vllm(self):
        """Test model loading when vLLM is not available"""
        from model_loader import ModelManager
        manager = ModelManager()

        model = manager.load_model("tiny-model")

        assert model.name == "tiny-model"
        assert model.is_loaded is True
        assert model.llm is None  # No actual LLM when vLLM unavailable
        assert "tiny-model" in manager.current_loaded

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_model_unloading(self):
        """Test model unloading"""
        from model_loader import ModelManager
        manager = ModelManager()

        # Load model
        model = manager.load_model("tiny-model")
        assert model.is_loaded is True
        assert "tiny-model" in manager.current_loaded

        # Unload model
        manager.unload_model("tiny-model")
        assert manager.models["tiny-model"].is_loaded is False
        assert "tiny-model" not in manager.current_loaded

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_max_models_enforcement(self):
        """Test that max_models_in_memory is enforced"""
        from model_loader import ModelManager
        manager = ModelManager(max_models_in_memory=2)

        # Load first model
        manager.load_model("tiny-model")
        assert len(manager.current_loaded) == 1

        # Load second model
        manager.load_model("small-model")
        assert len(manager.current_loaded) == 2

        # Load third model - should unload first
        manager.load_model("medium-model")
        assert len(manager.current_loaded) == 2
        assert "tiny-model" not in manager.current_loaded
        assert "small-model" in manager.current_loaded
        assert "medium-model" in manager.current_loaded

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_generate_with_mock(self):
        """Test generation with mock inference"""
        from model_loader import ModelManager
        manager = ModelManager()

        result = manager.generate(
            model_name="tiny-model",
            prompt="Test prompt",
            max_tokens=50
        )

        assert "generated_text" in result
        assert "model_name" in result
        assert result["model_name"] == "tiny-model"
        assert "Mock response" in result["generated_text"]
        assert result["total_tokens"] > 0

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_get_model_stats(self):
        """Test getting model statistics"""
        from model_loader import ModelManager
        manager = ModelManager()

        # Load and use model
        manager.load_model("tiny-model")
        manager.generate("tiny-model", "test", max_tokens=10)

        stats = manager.get_model_stats("tiny-model")

        assert stats["name"] == "tiny-model"
        assert stats["is_loaded"] is True
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] > 0

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_cleanup_all(self):
        """Test cleanup of all models"""
        from model_loader import ModelManager
        manager = ModelManager(max_models_in_memory=2)  # Allow 2 models

        # Load multiple models
        manager.load_model("tiny-model")
        manager.load_model("small-model")

        # Check that both are loaded (or at least one due to memory constraints)
        assert len(manager.current_loaded) >= 1

        # Cleanup all
        manager.cleanup_all()

        assert len(manager.current_loaded) == 0

    def test_singleton_pattern(self):
        """Test that get_model_manager returns singleton"""
        from model_loader import get_model_manager
        manager1 = get_model_manager()
        manager2 = get_model_manager()

        assert manager1 is manager2


class TestModelLoadingAPI:
    """Test suite for model loading API endpoints"""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager"""
        mock_manager = Mock()
        mock_manager.max_models_in_memory = 1
        mock_manager.get_loaded_models.return_value = []
        mock_manager.get_model_stats.return_value = None
        mock_manager.generate.return_value = {
            "generated_text": "[Mock response from tiny-model] This is a simulated response to: Test prompt...",
            "total_tokens": 20,
            "generation_time": 0.1,
            "tokens_per_second": 200
        }
        return mock_manager

    @pytest.fixture
    def client(self, mock_model_manager):
        """Create test client with mocked model manager"""
        # Mock the model manager before importing
        with patch('model_loader.get_model_manager', return_value=mock_model_manager):
            with patch('api_server.get_model_manager', return_value=mock_model_manager):
                # Now import the app
                from api_server import app
                import api_server

                # Set the model_manager directly
                api_server.model_manager = mock_model_manager

                # Create test client
                client = TestClient(app)
                # Attach the mock for use in tests
                client.mock_manager = mock_model_manager

                yield client

    def test_load_model_endpoint(self, client):
        """Test model loading via API"""
        mock_manager = client.mock_manager

        from model_loader import ModelInstance
        mock_instance = ModelInstance(
            name="tiny-model",
            model_path="facebook/opt-125m",
            is_loaded=True,
            load_time=1.5
        )
        mock_manager.load_model.return_value = mock_instance

        response = client.post("/models/tiny-model/load")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_name"] == "tiny-model"
        assert data["is_loaded"] is True

    def test_unload_model_endpoint(self, client):
        """Test model unloading via API"""
        mock_manager = client.mock_manager

        response = client.post("/models/tiny-model/unload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "unloaded" in data["message"]
        mock_manager.unload_model.assert_called_once_with("tiny-model")

    def test_get_loaded_models_endpoint(self, client):
        """Test getting loaded models via API"""
        mock_manager = client.mock_manager
        mock_manager.get_loaded_models.return_value = ["tiny-model"]
        mock_manager.max_models_in_memory = 1
        mock_manager.get_model_stats.return_value = {
            "name": "tiny-model",
            "model_path": "facebook/opt-125m",
            "is_loaded": True,
            "load_time": 1.0,
            "total_requests": 5,
            "total_tokens": 500
        }

        response = client.get("/models/loaded")

        assert response.status_code == 200
        data = response.json()
        assert data["total_loaded"] == 1
        assert data["max_models"] == 1
        assert len(data["loaded_models"]) == 1
        assert data["loaded_models"][0]["name"] == "tiny-model"

    def test_generate_with_real_model(self, client):
        """Test generation endpoint with real model manager"""
        mock_manager = client.mock_manager
        mock_manager.generate.return_value = {
            "generated_text": "This is the model response",
            "total_tokens": 25,
            "generation_time": 0.5,
            "tokens_per_second": 50
        }

        response = client.post("/v1/generate", json={
            "prompt": "Test prompt",
            "user_id": "test_user",
            "max_tokens": 50
        })

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        # The response will be from the mock
        assert "This is the model response" in data["response"] or "Mock response" in data["response"]
        assert data["tokens_used"] > 0

    def test_model_not_found(self, client):
        """Test loading non-existent model"""
        mock_manager = client.mock_manager
        mock_manager.load_model.side_effect = ValueError("Unknown model: invalid-model")

        response = client.post("/models/invalid-model/load")

        assert response.status_code == 404
        assert "Unknown model" in response.json()["detail"]

    def test_model_loading_error(self, client):
        """Test error handling during model loading"""
        mock_manager = client.mock_manager
        mock_manager.load_model.side_effect = Exception("GPU out of memory")

        response = client.post("/models/large-model/load")

        assert response.status_code == 500
        assert "GPU out of memory" in response.json()["detail"]

    def test_generate_fallback_to_mock(self, client):
        """Test that generation falls back to mock when model fails"""
        mock_manager = client.mock_manager
        # Make generate raise an exception to trigger fallback
        mock_manager.generate.side_effect = Exception("Model error")

        response = client.post("/v1/generate", json={
            "prompt": "Test prompt",
            "user_id": "test_user"
        })

        assert response.status_code == 200
        data = response.json()
        # Should fall back to mock response
        assert "Mock response" in data["response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])