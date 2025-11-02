#!/usr/bin/env python3
"""
Basic vLLM Model Server Implementation
Provides core functionality for serving LLMs with vLLM
"""

import time
import asyncio
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass, field
import logging

from vllm import LLM, SamplingParams
from vllm import AsyncLLMEngine, AsyncEngineArgs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model server"""
    model_name: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    stream: bool = False


class ModelServer:
    """
    vLLM-based model server for high-performance inference
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model server with configuration

        Args:
            config: Dictionary containing model configuration
        """
        self.model_name = config.get("model_name", "facebook/opt-125m")
        self.max_tokens = config.get("max_tokens", 100)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 50)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        self.max_model_len = config.get("max_model_len", None)
        self.stream = config.get("stream", False)

        self.model: Optional[LLM] = None
        self.async_engine: Optional[AsyncLLMEngine] = None

        # Metrics tracking
        self.request_count = 0
        self.total_tokens_generated = 0
        self.latencies: List[float] = []
        self.start_time = time.time()

    def load_model(self):
        """
        Load the model into memory
        """
        try:
            # Clean up any existing GPU memory first
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            logger.info(f"Loading model: {self.model_name}")

            # vLLM configuration with enforced eager mode for testing
            engine_args = {
                "model": self.model_name,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": True,
                "enforce_eager": True,  # Important for testing to avoid memory issues
            }

            if self.max_model_len:
                engine_args["max_model_len"] = self.max_model_len

            # Initialize vLLM engine
            self.model = LLM(**engine_args)

            logger.info(f"Model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ValueError(f"Failed to load model: {e}")

    def unload_model(self):
        """
        Unload the model from memory
        """
        if self.model:
            del self.model
            self.model = None

            # Clean up GPU memory
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logger.info("Model unloaded and GPU memory cleared")

    def is_ready(self) -> bool:
        """
        Check if the model is loaded and ready

        Returns:
            True if model is ready, False otherwise
        """
        return self.model is not None

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Generate text from a single prompt

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Generated text including the prompt
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        start_time = time.time()

        # Use provided parameters or defaults
        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p,
            top_k=top_k or self.top_k
        )

        # Generate with vLLM
        outputs = self.model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Track metrics
        latency = (time.time() - start_time) * 1000  # Convert to ms
        self.latencies.append(latency)
        self.request_count += 1
        self.total_tokens_generated += len(outputs[0].outputs[0].token_ids)

        # Return prompt + generated text
        return prompt + generated_text

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Generate text from multiple prompts in batch

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        if not all(prompts):
            raise ValueError("All prompts must be non-empty")

        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        start_time = time.time()

        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )

        # Batch generation with vLLM
        outputs = self.model.generate(prompts, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            results.append(prompts[i] + generated_text)
            self.total_tokens_generated += len(output.outputs[0].token_ids)

        # Track metrics
        latency = (time.time() - start_time) * 1000
        self.latencies.append(latency)
        self.request_count += len(prompts)

        return results

    async def generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Async generation for better concurrency

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # For now, wrap synchronous generation
        # In production, would use AsyncLLMEngine
        return self.generate(prompt, max_tokens, temperature)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        Stream tokens as they are generated

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated tokens
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # For demonstration, we'll simulate streaming
        # In production, would use vLLM's streaming API
        full_response = self.generate(prompt, max_tokens, temperature)

        # Remove the prompt from response
        generated_only = full_response[len(prompt):]

        # Simulate streaming by yielding prompt first, then tokens
        yield prompt

        # Yield tokens one at a time (simplified)
        words = generated_only.split()
        for word in words:
            yield " " + word

    def get_request_count(self) -> int:
        """
        Get total number of requests processed

        Returns:
            Request count
        """
        return self.request_count

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get server metrics

        Returns:
            Dictionary of metrics
        """
        if not self.latencies:
            return {
                "request_count": self.request_count,
                "total_tokens": self.total_tokens_generated,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "tokens_per_second": 0
            }

        # Calculate latency percentiles
        sorted_latencies = sorted(self.latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        tokens_per_second = self.total_tokens_generated / elapsed_time if elapsed_time > 0 else 0

        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens_generated,
            "avg_latency_ms": sum(self.latencies) / len(self.latencies),
            "p95_latency_ms": sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1],
            "p99_latency_ms": sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else sorted_latencies[-1],
            "tokens_per_second": tokens_per_second
        }

    def reset_metrics(self):
        """
        Reset all metrics
        """
        self.request_count = 0
        self.total_tokens_generated = 0
        self.latencies = []
        self.start_time = time.time()


def create_server(config: Dict[str, Any]) -> ModelServer:
    """
    Factory function to create a model server

    Args:
        config: Server configuration

    Returns:
        Configured ModelServer instance
    """
    return ModelServer(config)


if __name__ == "__main__":
    # Example usage
    config = {
        "model_name": "facebook/opt-125m",  # Small model for testing
        "max_tokens": 100,
        "temperature": 0.7,
        "gpu_memory_utilization": 0.5
    }

    server = create_server(config)
    server.load_model()

    # Test single generation
    prompt = "The future of artificial intelligence is"
    response = server.generate(prompt)
    print(f"Generated: {response}")

    # Test batch generation
    prompts = [
        "Machine learning can help",
        "The stock market today",
        "Investment strategies include"
    ]
    responses = server.generate_batch(prompts)
    for r in responses:
        print(f"Batch generated: {r[:100]}...")

    # Show metrics
    metrics = server.get_metrics()
    print(f"\nMetrics: {metrics}")

    server.unload_model()