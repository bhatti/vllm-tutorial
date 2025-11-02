#!/usr/bin/env python3
"""
Basic vLLM test script to verify installation
Run this after setting up vLLM to ensure everything works correctly
"""
import torch
from vllm import LLM, SamplingParams
import time
import sys
import os

def test_vllm():
    print("=" * 60)
    print("vLLM Installation Test")
    print("=" * 60)

    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("ERROR: No GPU detected! vLLM requires NVIDIA GPU.")
        return False

    # Test with Phi-2 (smallest model)
    print("\n" + "=" * 60)
    print("Testing inference with Phi-2...")
    print("=" * 60)

    # Check if model exists
    model_path = "./models/phi-2"

    # If model doesn't exist locally, try to use HuggingFace model ID
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Attempting to use HuggingFace model ID directly...")
        model_path = "microsoft/phi-2"

    try:
        print(f"\nInitializing vLLM with model: {model_path}")
        print("This may take a minute on first run...")

        llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            trust_remote_code=True  # Required for Phi-2
        )

        print("✓ vLLM initialized successfully!")

        # Test prompts
        prompts = [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers."
        ]

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=100
        )

        print("\nRunning inference on test prompts...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)

        for i, output in enumerate(outputs):
            print(f"\nPrompt {i+1}: {output.prompt[:50]}...")
            print(f"Response: {output.outputs[0].text[:200]}...")
            print(f"Tokens generated: {len(output.outputs[0].token_ids)}")

        # Performance metrics
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_time = end_time - start_time

        print(f"\n" + "=" * 60)
        print("PERFORMANCE METRICS:")
        print("=" * 60)
        print(f"✓ Test completed successfully!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Throughput: {total_tokens / total_time:.1f} tokens/sec")
        print(f"Average latency per prompt: {total_time / len(prompts) * 1000:.1f} ms")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure vLLM is installed: pip install vllm")
        print("2. Check if GPU drivers are properly installed: nvidia-smi")
        print("3. Ensure you have enough GPU memory available")
        print("4. Try downloading the model first:")
        print("   huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2")
        return False

def test_multiple_models():
    """Test multiple models if available"""
    print("\n" + "=" * 60)
    print("Testing Multiple Models")
    print("=" * 60)

    models_to_test = [
        ("./models/phi-2", "microsoft/phi-2", "Phi-2 (2.7B)"),
        ("./models/mistral-7b", "mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B"),
    ]

    for local_path, hf_id, name in models_to_test:
        print(f"\nTesting {name}...")

        # Check which path to use
        if os.path.exists(local_path):
            model_path = local_path
            print(f"Using local model at {local_path}")
        else:
            model_path = hf_id
            print(f"Using HuggingFace model: {hf_id}")
            print("Note: First run will download the model (may take several minutes)")

        try:
            # Different memory allocation for different model sizes
            if "7b" in name.lower() or "7B" in name:
                gpu_util = 0.9
                max_len = 4096
            else:
                gpu_util = 0.5
                max_len = 2048

            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_util,
                max_model_len=max_len,
                trust_remote_code=True
            )

            # Quick test
            prompt = "Write a short poem about AI."
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=50
            )

            start_time = time.time()
            output = llm.generate([prompt], sampling_params)[0]
            inference_time = time.time() - start_time

            print(f"✓ {name} working!")
            print(f"  Response: {output.outputs[0].text[:100]}...")
            print(f"  Inference time: {inference_time:.2f}s")
            print(f"  Tokens/sec: {len(output.outputs[0].token_ids) / inference_time:.1f}")

            # Clean up to free memory for next model
            del llm
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ {name} failed: {str(e)[:100]}...")

def check_environment():
    """Check the environment setup"""
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)

    # Python version
    print(f"Python version: {sys.version}")

    # Check important packages
    packages = {
        "torch": None,
        "vllm": None,
        "transformers": None,
        "fastapi": None,
        "uvicorn": None
    }

    for package in packages:
        try:
            module = __import__(package)
            if hasattr(module, "__version__"):
                packages[package] = module.__version__
            else:
                packages[package] = "installed"
        except ImportError:
            packages[package] = "not installed"

    print("\nPackage Status:")
    for package, status in packages.items():
        symbol = "✓" if status != "not installed" else "✗"
        print(f"{symbol} {package}: {status}")

    # Check model directory
    print("\nModel Directory:")
    if os.path.exists("./models"):
        models = os.listdir("./models")
        if models:
            print("Models found:")
            for model in models:
                model_path = os.path.join("./models", model)
                size_gb = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                ) / (1024**3)
                print(f"  - {model} ({size_gb:.1f} GB)")
        else:
            print("  No models downloaded yet")
    else:
        print("  Models directory not found")

    # GPU memory status
    if torch.cuda.is_available():
        print("\nGPU Memory Status:")
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Available: {total - reserved:.2f} GB")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║                    vLLM Test Suite                        ║
║                                                            ║
║  This script will test your vLLM installation and         ║
║  verify that everything is working correctly.             ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Run environment check first
    check_environment()

    # Run main test
    success = test_vllm()

    if success:
        # Try testing multiple models if the basic test passed
        test_multiple_models()

        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start the API server: python vllm_server.py")
        print("2. Run benchmarks: python benchmark.py")
        print("3. Test with your enterprise examples")
        print("\nYour vLLM setup is ready for production use! 🚀")
    else:
        print("\n" + "=" * 60)
        print("✗ Some tests failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)
