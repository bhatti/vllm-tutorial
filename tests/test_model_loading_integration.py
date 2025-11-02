#!/usr/bin/env python3
"""
Integration tests for model loading - testing business logic directly
This avoids all FastAPI/HTTP issues and tests the actual functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestModelManagerBusinessLogic:
    """Test ModelManager business logic directly without FastAPI"""

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
        manager = ModelManager(max_models_in_memory=2)

        # Load multiple models
        manager.load_model("tiny-model")
        manager.load_model("small-model")

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

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_get_loaded_models_logic(self):
        """Test the get_loaded_models business logic"""
        from model_loader import ModelManager
        # Create manager with room for 2 models
        manager = ModelManager(max_models_in_memory=2)

        # Initially empty
        loaded = manager.get_loaded_models()
        assert loaded == []

        # Load a model
        manager.load_model("tiny-model")
        loaded = manager.get_loaded_models()
        assert loaded == ["tiny-model"]

        # Load another
        manager.load_model("small-model")
        loaded = manager.get_loaded_models()
        assert set(loaded) == {"tiny-model", "small-model"}

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_generate_endpoint_logic(self):
        """Test the generation logic that the endpoint would use"""
        from model_loader import ModelManager
        manager = ModelManager()

        # Test successful generation
        result = manager.generate(
            model_name="tiny-model",
            prompt="Test prompt for generation",
            max_tokens=50,
            temperature=0.7
        )

        assert "generated_text" in result
        assert "total_tokens" in result
        assert "generation_time" in result
        assert result["model_name"] == "tiny-model"
        assert len(result["generated_text"]) > 0

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_load_model_endpoint_logic(self):
        """Test the load model logic that the endpoint would use"""
        from model_loader import ModelManager, ModelInstance
        manager = ModelManager()

        # Load a model
        model = manager.load_model("tiny-model")

        assert isinstance(model, ModelInstance)
        assert model.name == "tiny-model"
        assert model.is_loaded is True
        assert model.load_time > 0
        assert "tiny-model" in manager.current_loaded

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_unload_model_endpoint_logic(self):
        """Test the unload model logic that the endpoint would use"""
        from model_loader import ModelManager
        manager = ModelManager()

        # Load then unload
        manager.load_model("tiny-model")
        assert "tiny-model" in manager.current_loaded

        manager.unload_model("tiny-model")
        assert "tiny-model" not in manager.current_loaded
        assert manager.models["tiny-model"].is_loaded is False

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_model_not_found_logic(self):
        """Test error handling for unknown model"""
        from model_loader import ModelManager
        manager = ModelManager()

        with pytest.raises(ValueError, match="Unknown model"):
            manager.load_model("invalid-model")

    @patch('model_loader.VLLM_AVAILABLE', False)
    def test_generate_fallback_logic(self):
        """Test that generation has fallback when model errors occur"""
        from model_loader import ModelManager
        manager = ModelManager()

        # Even if something goes wrong, generate should return a result
        # This tests the mock fallback path
        result = manager.generate(
            model_name="tiny-model",
            prompt="Test",
            max_tokens=10
        )

        # Should get a mock response
        assert "generated_text" in result
        assert result["total_tokens"] > 0


class TestIntelligentRoutingLogic:
    """Test intelligent routing logic directly"""

    def test_routing_decision_simple_request(self):
        """Test routing decision for simple requests"""
        from intelligent_router_simple import IntelligentRouter, RoutingRequest, ModelConfig, ModelTier, RequestComplexity

        configs = [
            ModelConfig(
                name="tiny-model",
                tier=ModelTier.TINY,
                model_path="facebook/opt-125m",
                max_complexity=RequestComplexity.SIMPLE,
                cost_per_1k_tokens=0.0001,
                avg_latency_ms=50,
                capabilities=["general"]
            )
        ]

        router = IntelligentRouter(configs)
        request = RoutingRequest(
            id="test-1",
            content="What is 2+2?",
            user_id="user1"
        )

        decision = router.route(request)
        assert decision.model_name == "tiny-model"
        assert decision.complexity == RequestComplexity.SIMPLE

    def test_routing_with_cost_constraint(self):
        """Test routing respects cost constraints"""
        from intelligent_router_simple import IntelligentRouter, RoutingRequest, ModelConfig, ModelTier, RequestComplexity

        configs = [
            ModelConfig(
                name="tiny-model",
                tier=ModelTier.TINY,
                model_path="facebook/opt-125m",
                max_complexity=RequestComplexity.SIMPLE,
                cost_per_1k_tokens=0.0001,
                avg_latency_ms=50,
                capabilities=["general"]
            ),
            ModelConfig(
                name="large-model",
                tier=ModelTier.LARGE,
                model_path="meta-llama/Llama-2-13b-hf",
                max_complexity=RequestComplexity.CRITICAL,
                cost_per_1k_tokens=0.1,
                avg_latency_ms=1000,
                capabilities=["general", "analysis"]
            )
        ]

        router = IntelligentRouter(configs)
        request = RoutingRequest(
            id="test-2",
            content="Complex financial analysis question",
            user_id="user1",
            max_cost=0.001  # Very low budget
        )

        decision = router.route(request)
        # Should route to cheaper model despite complexity
        assert decision.model_name == "tiny-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
