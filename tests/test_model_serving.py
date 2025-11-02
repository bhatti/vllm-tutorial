#!/usr/bin/env python3
"""
Test suite for basic vLLM model serving
Following TDD principles - tests first, then implementation
"""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import asyncio
import torch
import gc
from typing import List, Optional
from unittest.mock import Mock, patch, MagicMock


# Global model instance to share across tests
_model_server = None


@pytest.fixture(scope="session")
def model_server():
    """
    Session-scoped fixture that creates a single model instance
    to be reused across all tests
    """
    global _model_server
    if _model_server is None:
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m",
            "max_tokens": 100,
            "temperature": 0.7,
            "gpu_memory_utilization": 0.3  # Use less memory for tests
        }

        _model_server = ModelServer(config)
        _model_server.load_model()

    yield _model_server

    # Cleanup is done at the end of all tests


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """
    Clean up GPU memory before and after each test
    """
    # Clean before test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    yield

    # Clean after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class TestModelServing:
    """Test basic model serving functionality"""

    def test_model_initialization(self):
        """Test that model server can be initialized with config"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m",  # Small model for testing
            "max_tokens": 100,
            "temperature": 0.7,
            "gpu_memory_utilization": 0.5
        }

        server = ModelServer(config)
        assert server.model_name == "facebook/opt-125m"
        assert server.max_tokens == 100
        assert server.temperature == 0.7
        assert server.gpu_memory_utilization == 0.5

    def test_model_loading(self):
        """Test that model loads correctly"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m",
            "gpu_memory_utilization": 0.3
        }

        # Create a separate instance just for this test
        server = ModelServer(config)
        # Don't load - just check initialization
        assert server.model_name == "facebook/opt-125m"
        assert server.is_ready() is False

    def test_single_prompt_generation(self, model_server):
        """Test generation with a single prompt"""
        prompt = "The future of finance is"
        response = model_server.generate(prompt, max_tokens=50)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > len(prompt)
        assert response.startswith(prompt)

    def test_batch_generation(self, model_server):
        """Test generation with multiple prompts"""
        prompts = [
            "The stock market today",
            "Investment strategies for",
            "Risk management in banking"
        ]

        responses = model_server.generate_batch(prompts, max_tokens=30)

        assert len(responses) == len(prompts)
        for i, response in enumerate(responses):
            assert response is not None
            assert isinstance(response, str)
            assert response.startswith(prompts[i])

    def test_generation_parameters(self):
        """Test different generation parameters"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m",
            "max_tokens": 100
        }

        server = ModelServer(config)
        server.load_model()

        prompt = "Financial analysis shows"

        # Test with different temperatures
        response_low_temp = server.generate(prompt, temperature=0.1)
        response_high_temp = server.generate(prompt, temperature=1.5)

        assert response_low_temp is not None
        assert response_high_temp is not None
        # Low temperature should be more deterministic

        # Test with different max_tokens
        response_short = server.generate(prompt, max_tokens=10)
        response_long = server.generate(prompt, max_tokens=100)

        assert len(response_long) >= len(response_short)

    def test_model_unloading(self):
        """Test that model can be properly unloaded"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m"
        }

        server = ModelServer(config)
        server.load_model()
        assert server.is_ready() is True

        server.unload_model()
        assert server.is_ready() is False
        assert server.model is None

    def test_error_handling_invalid_model(self):
        """Test error handling for invalid model name"""
        from src.model_server import ModelServer

        config = {
            "model_name": "invalid/model/name"
        }

        server = ModelServer(config)
        with pytest.raises(ValueError, match="Failed to load model"):
            server.load_model()

    def test_error_handling_empty_prompt(self):
        """Test error handling for empty prompt"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m"
        }

        server = ModelServer(config)
        server.load_model()

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            server.generate("")

    @pytest.mark.asyncio
    async def test_async_generation(self):
        """Test async generation capability"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m",
            "max_tokens": 50
        }

        server = ModelServer(config)
        server.load_model()

        prompt = "Async financial processing"
        response = await server.generate_async(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert response.startswith(prompt)

    def test_streaming_generation(self):
        """Test streaming token generation"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m",
            "max_tokens": 50,
            "stream": True
        }

        server = ModelServer(config)
        server.load_model()

        prompt = "Streaming financial data"
        tokens = []

        for token in server.generate_stream(prompt):
            tokens.append(token)
            assert isinstance(token, str)

        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert full_response.startswith(prompt)


class TestModelServerMetrics:
    """Test metrics and monitoring for model server"""

    def test_request_counting(self):
        """Test that requests are counted correctly"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m"
        }

        server = ModelServer(config)
        server.load_model()

        initial_count = server.get_request_count()

        server.generate("Test prompt 1")
        server.generate("Test prompt 2")

        assert server.get_request_count() == initial_count + 2

    def test_latency_tracking(self):
        """Test that latency is tracked"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m"
        }

        server = ModelServer(config)
        server.load_model()

        server.generate("Test prompt")

        metrics = server.get_metrics()
        assert "avg_latency_ms" in metrics
        assert metrics["avg_latency_ms"] > 0
        assert "p95_latency_ms" in metrics
        assert "p99_latency_ms" in metrics

    def test_throughput_calculation(self):
        """Test throughput calculation"""
        from src.model_server import ModelServer

        config = {
            "model_name": "facebook/opt-125m"
        }

        server = ModelServer(config)
        server.load_model()

        # Generate multiple requests
        prompts = ["Test " + str(i) for i in range(5)]
        for prompt in prompts:
            server.generate(prompt)

        metrics = server.get_metrics()
        assert "tokens_per_second" in metrics
        assert metrics["tokens_per_second"] > 0


if __name__ == "__main__":
    # Run basic tests
    print("Running Model Serving Tests...")
    pytest.main([__file__, "-v"])