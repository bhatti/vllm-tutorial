#!/usr/bin/env python3
"""
Basic test script for vLLM model serving
Run this to verify the model server works on GCP
"""

import sys
import torch
import gc
from src.model_server import ModelServer


def cleanup_gpu():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_basic_serving():
    """Test basic model serving functionality"""
    print("=" * 60)
    print("Testing vLLM Model Server on GCP")
    print("=" * 60)

    # Clean up before starting
    cleanup_gpu()

    # Configuration
    config = {
        "model_name": "facebook/opt-125m",  # Small model for testing
        "max_tokens": 50,
        "temperature": 0.7,
        "gpu_memory_utilization": 0.3,  # Conservative memory usage
    }

    print(f"\n1. Initializing model server with config:")
    print(f"   Model: {config['model_name']}")
    print(f"   GPU Memory: {config['gpu_memory_utilization']}")

    # Create server
    server = ModelServer(config)

    # Load model
    print("\n2. Loading model...")
    try:
        server.load_model()
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False

    # Test single generation
    print("\n3. Testing single prompt generation...")
    prompt = "The future of artificial intelligence is"
    try:
        response = server.generate(prompt, max_tokens=30)
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response}")
        print("   ✅ Single generation successful!")
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False

    # Test batch generation
    print("\n4. Testing batch generation...")
    prompts = [
        "Machine learning helps",
        "The stock market today",
    ]
    try:
        responses = server.generate_batch(prompts, max_tokens=20)
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"   Batch {i+1}:")
            print(f"     Prompt: {prompt}")
            print(f"     Response: {response[:100]}...")
        print("   ✅ Batch generation successful!")
    except Exception as e:
        print(f"   ❌ Batch generation failed: {e}")
        return False

    # Test metrics
    print("\n5. Checking metrics...")
    metrics = server.get_metrics()
    print(f"   Requests: {metrics['request_count']}")
    print(f"   Total tokens: {metrics['total_tokens']}")
    print(f"   Avg latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"   Tokens/sec: {metrics['tokens_per_second']:.2f}")
    print("   ✅ Metrics working!")

    # Clean up
    print("\n6. Cleaning up...")
    server.unload_model()
    cleanup_gpu()
    print("   ✅ Cleanup complete!")

    print("\n" + "=" * 60)
    print("✅ All tests passed! vLLM is working correctly on GCP.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_basic_serving()
    sys.exit(0 if success else 1)