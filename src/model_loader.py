#!/usr/bin/env python3
"""
vLLM Model Loader for production inference
Handles real model loading and inference with GPU support
"""

import os
import gc
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import torch

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, using mock inference")

logger = logging.getLogger(__name__)


@dataclass
class ModelInstance:
    """Represents a loaded model instance"""
    name: str
    model_path: str
    llm: Optional[Any] = None  # vLLM LLM instance
    is_loaded: bool = False
    load_time: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0


class ModelManager:
    """Manages vLLM model instances with intelligent loading/unloading"""

    def __init__(self, max_models_in_memory: int = 1):
        """
        Initialize model manager

        Args:
            max_models_in_memory: Maximum number of models to keep loaded
        """
        self.models: Dict[str, ModelInstance] = {}
        self.max_models_in_memory = max_models_in_memory
        self.current_loaded: List[str] = []

        # Default model configurations (optimized for L4 GPU - 24GB VRAM)
        self.model_configs = {
            "tiny-model": {
                "model_path": "facebook/opt-125m",  # ~250MB
                "gpu_memory_utilization": 0.1,
                "max_model_len": 512,
            },
            "small-model": {
                "model_path": "facebook/opt-350m",  # ~700MB, changed from 1.3b
                "gpu_memory_utilization": 0.2,
                "max_model_len": 1024,
            },
            "medium-model": {
                "model_path": "facebook/opt-1.3b",  # ~2.6GB, changed from 6.7b
                "gpu_memory_utilization": 0.3,
                "max_model_len": 2048,
            },
            "large-model": {
                "model_path": "facebook/opt-2.7b",  # ~5.4GB, fits on L4
                "gpu_memory_utilization": 0.5,
                "max_model_len": 2048,
            }
        }

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.2f}GB memory")
        else:
            logger.warning("No GPU detected, using CPU (will be slow)")

    def cleanup_gpu_memory(self):
        """Clean up GPU memory before loading new model"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleaned up")

    def load_model(self, model_name: str, force_reload: bool = False) -> ModelInstance:
        """
        Load a model into memory

        Args:
            model_name: Name of the model to load
            force_reload: Force reload even if already loaded

        Returns:
            ModelInstance object
        """
        # Check if model already loaded
        if model_name in self.models and self.models[model_name].is_loaded and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return self.models[model_name]

        # Get model config
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = self.model_configs[model_name]

        # Check if we need to unload other models
        if len(self.current_loaded) >= self.max_models_in_memory:
            # Unload least recently used model
            if self.current_loaded:
                old_model = self.current_loaded[0]
                self.unload_model(old_model)

        # Clean up GPU memory
        self.cleanup_gpu_memory()

        # Create model instance
        model_instance = ModelInstance(
            name=model_name,
            model_path=config["model_path"],
            is_loaded=False
        )

        start_time = time.time()

        try:
            if VLLM_AVAILABLE:
                logger.info(f"Loading {model_name} from {config['model_path']}...")

                # vLLM configuration
                llm = LLM(
                    model=config["model_path"],
                    gpu_memory_utilization=config.get("gpu_memory_utilization", 0.3),
                    max_model_len=config.get("max_model_len", 1024),
                    trust_remote_code=True,
                    dtype="float16" if self.gpu_available else "float32",
                    enforce_eager=True,  # Disable CUDA graphs for stability
                    disable_log_stats=True
                )

                model_instance.llm = llm
                model_instance.is_loaded = True
                model_instance.load_time = time.time() - start_time

                logger.info(f"Model {model_name} loaded in {model_instance.load_time:.2f}s")
            else:
                logger.warning(f"vLLM not available, using mock for {model_name}")
                model_instance.is_loaded = True
                model_instance.load_time = 0.1

            # Update tracking
            self.models[model_name] = model_instance
            self.current_loaded.append(model_name)

            return model_instance

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            # Try cleanup
            self.cleanup_gpu_memory()
            raise

    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.models and self.models[model_name].is_loaded:
            logger.info(f"Unloading model {model_name}")

            # Delete the LLM instance
            if self.models[model_name].llm is not None:
                del self.models[model_name].llm
                self.models[model_name].llm = None

            self.models[model_name].is_loaded = False

            # Remove from loaded list
            if model_name in self.current_loaded:
                self.current_loaded.remove(model_name)

            # Clean up GPU memory
            self.cleanup_gpu_memory()

            logger.info(f"Model {model_name} unloaded")

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using specified model

        Args:
            model_name: Name of model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Whether to stream response

        Returns:
            Dictionary with generated text and metadata
        """
        # Load model if not already loaded
        model = self.load_model(model_name)

        if not model.is_loaded:
            raise RuntimeError(f"Model {model_name} failed to load")

        start_time = time.time()

        if VLLM_AVAILABLE and model.llm is not None:
            # Real vLLM inference
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            # Generate
            outputs = model.llm.generate([prompt], sampling_params)

            # Extract response
            generated_text = outputs[0].outputs[0].text
            total_tokens = len(outputs[0].outputs[0].token_ids)

        else:
            # Mock generation for testing
            generated_text = f"[Mock response from {model_name}] Response to: {prompt[:50]}..."
            total_tokens = len(prompt.split()) + 20

        # Calculate metrics
        generation_time = time.time() - start_time
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

        # Update model stats
        model.total_requests += 1
        model.total_tokens += total_tokens

        return {
            "generated_text": generated_text,
            "model_name": model_name,
            "total_tokens": total_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return self.current_loaded.copy()

    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific model"""
        if model_name not in self.models:
            return None

        model = self.models[model_name]
        return {
            "name": model.name,
            "model_path": model.model_path,
            "is_loaded": model.is_loaded,
            "load_time": model.load_time,
            "total_requests": model.total_requests,
            "total_tokens": model.total_tokens
        }

    def cleanup_all(self):
        """Unload all models and cleanup"""
        logger.info("Cleaning up all models...")
        for model_name in list(self.current_loaded):
            self.unload_model(model_name)
        self.cleanup_gpu_memory()


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create model manager singleton"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(max_models_in_memory=1)
    return _model_manager


if __name__ == "__main__":
    # Test the model loader
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Testing Model Loader...")
    print("-" * 50)

    manager = get_model_manager()

    # Test with tiny model
    print("\n1. Testing with tiny model (facebook/opt-125m)...")
    try:
        result = manager.generate(
            model_name="tiny-model",
            prompt="What is machine learning?",
            max_tokens=50,
            temperature=0.7
        )
        print(f"✅ Generated: {result['generated_text'][:100]}...")
        print(f"   Tokens: {result['total_tokens']}")
        print(f"   Time: {result['generation_time']:.2f}s")
        print(f"   Speed: {result['tokens_per_second']:.2f} tokens/s")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Cleanup
    manager.cleanup_all()
    print("\n✅ Test complete!")