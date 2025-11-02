#!/usr/bin/env python3
"""
Simplified test suite for vLLM model serving
Uses a single model instance to avoid GPU memory issues
"""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import time
import torch
import gc
from src.model_server import ModelServer


# Global model server instance
_model_server = None


def get_model_server():
    """Get or create a shared model server instance"""
    global _model_server
    if _model_server is None:
        # Clean GPU memory before creating
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        config = {
            "model_name": "facebook/opt-125m",
            "max_tokens": 50,
            "temperature": 0.7,
            "gpu_memory_utilization": 0.3,  # Use less memory for testing
        }
        _model_server = ModelServer(config)
        _model_server.load_model()
    return _model_server


class TestBasicServing:
    """Test basic model serving functionality"""

    def test_initialization(self):
        """Test model server initialization"""
        config = {
            "model_name": "test-model",
            "max_tokens": 100,
        }
        server = ModelServer(config)
        assert server.model_name == "test-model"
        assert server.max_tokens == 100
        assert server.model is None  # Not loaded yet

    def test_single_generation(self):
        """Test single prompt generation"""
        server = get_model_server()

        prompt = "Hello, the weather is"
        response = server.generate(prompt, max_tokens=20)

        assert response is not None
        assert isinstance(response, str)
        assert response.startswith(prompt)
        assert len(response) > len(prompt)
        print(f"\nGenerated: {response}")

    def test_batch_generation(self):
        """Test batch generation"""
        server = get_model_server()

        prompts = [
            "The stock market",
            "Machine learning is",
        ]
        responses = server.generate_batch(prompts, max_tokens=20)

        assert len(responses) == len(prompts)
        for i, response in enumerate(responses):
            assert response.startswith(prompts[i])
            print(f"\nBatch {i}: {response[:100]}")

    def test_empty_prompt_handling(self):
        """Test error handling for empty prompt"""
        server = get_model_server()

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            server.generate("")

    def test_metrics_tracking(self):
        """Test that metrics are tracked"""
        server = get_model_server()

        # Reset metrics
        server.reset_metrics()

        # Generate some text
        server.generate("Test prompt", max_tokens=10)

        metrics = server.get_metrics()
        assert metrics["request_count"] >= 1
        assert metrics["total_tokens"] > 0
        assert metrics["avg_latency_ms"] > 0
        print(f"\nMetrics: {metrics}")


class TestAdvancedFeatures:
    """Test advanced features"""

    def test_temperature_variation(self):
        """Test generation with different temperatures"""
        server = get_model_server()

        prompt = "The number is"

        # Low temperature (more deterministic)
        response_low = server.generate(prompt, max_tokens=10, temperature=0.1)

        # High temperature (more random)
        response_high = server.generate(prompt, max_tokens=10, temperature=1.5)

        assert response_low is not None
        assert response_high is not None
        print(f"\nLow temp: {response_low}")
        print(f"High temp: {response_high}")

    def test_max_tokens_limit(self):
        """Test that max_tokens is respected"""
        server = get_model_server()

        prompt = "Count from one to ten:"

        response_short = server.generate(prompt, max_tokens=5)
        response_long = server.generate(prompt, max_tokens=20)

        # Long response should have more tokens
        # Note: Can't guarantee exact token count due to tokenization
        assert len(response_long) >= len(response_short)
        print(f"\nShort: {response_short}")
        print(f"Long: {response_long}")


def test_cleanup():
    """Test cleanup at the end"""
    global _model_server
    if _model_server is not None:
        _model_server.unload_model()
        _model_server = None
        print("\nModel server cleaned up")


if __name__ == "__main__":
    # Run tests with minimal output
    pytest.main([__file__, "-v", "-s"])