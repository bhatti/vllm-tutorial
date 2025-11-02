#!/usr/bin/env python3
"""
Example 1: Basic vLLM Setup and Usage
Demonstrates the fundamentals of vLLM for the blog series
"""

import time
from typing import List, Dict
import torch

# Check if vLLM is available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("⚠️  vLLM not installed. Install with: pip install vllm")
    VLLM_AVAILABLE = False


class BasicVLLMExample:
    """Basic vLLM usage example for blog"""

    def __init__(self, model_name: str = "microsoft/phi-2"):
        """
        Initialize vLLM with a small model

        Args:
            model_name: HuggingFace model ID (default: Phi-2 2.7B)
        """
        self.model_name = model_name
        self.llm = None

        if VLLM_AVAILABLE:
            print(f"🚀 Loading {model_name} with vLLM...")
            start = time.time()

            # Initialize vLLM with optimized settings
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=0.9,  # Use 90% of GPU memory
                max_model_len=2048,           # Context length
                trust_remote_code=True        # For models like Phi-2
            )

            load_time = time.time() - start
            print(f"✅ Model loaded in {load_time:.2f} seconds")
        else:
            print("❌ vLLM not available, using mock mode")

    def simple_generation(self, prompt: str) -> str:
        """
        Generate text from a single prompt

        Args:
            prompt: Input text

        Returns:
            Generated text
        """
        if not VLLM_AVAILABLE or not self.llm:
            return f"[Mock response to: {prompt[:50]}...]"

        print(f"\n📝 Prompt: {prompt}")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256
        )

        # Generate
        start = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        generation_time = time.time() - start

        response = outputs[0].outputs[0].text
        tokens = len(outputs[0].outputs[0].token_ids)

        print(f"⚡ Generated {tokens} tokens in {generation_time:.2f}s")
        print(f"📊 Throughput: {tokens/generation_time:.1f} tokens/sec")
        print(f"💬 Response: {response}\n")

        return response

    def batch_generation(self, prompts: List[str]) -> List[str]:
        """
        Generate text from multiple prompts (demonstrates batching)

        Args:
            prompts: List of input texts

        Returns:
            List of generated texts
        """
        if not VLLM_AVAILABLE or not self.llm:
            return [f"[Mock response {i}]" for i in range(len(prompts))]

        print(f"\n📦 Batch processing {len(prompts)} prompts...")

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100
        )

        # Batch generate - vLLM automatically optimizes this
        start = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        batch_time = time.time() - start

        # Calculate statistics
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / batch_time

        print(f"✅ Batch complete!")
        print(f"⏱️  Total time: {batch_time:.2f}s")
        print(f"📊 Total tokens: {total_tokens}")
        print(f"⚡ Throughput: {throughput:.1f} tokens/sec")
        print(f"📈 Per-prompt avg: {batch_time/len(prompts):.3f}s")

        return [output.outputs[0].text for output in outputs]

    def streaming_generation(self, prompt: str):
        """
        Demonstrate streaming output (for interactive applications)

        Note: This shows the concept - full streaming requires async API
        """
        print(f"\n🌊 Streaming generation (simulated)...")
        print(f"📝 Prompt: {prompt}\n")

        # For streaming, you'd use the AsyncLLMEngine
        # This is a simplified example
        response = self.simple_generation(prompt)

        # Simulate streaming output
        print("💬 Streaming output:")
        for i, char in enumerate(response):
            print(char, end='', flush=True)
            if i % 10 == 0:
                time.sleep(0.01)  # Simulate streaming delay
        print("\n")

    def compare_with_transformers(self, prompt: str):
        """
        Compare vLLM with HuggingFace Transformers (for blog benchmarks)
        """
        print("\n🔬 Comparing vLLM vs HuggingFace Transformers...")

        if not VLLM_AVAILABLE:
            print("❌ vLLM not available for comparison")
            return

        # vLLM generation
        print("\n1️⃣ Testing vLLM...")
        vllm_start = time.time()
        vllm_output = self.llm.generate(
            [prompt],
            SamplingParams(max_tokens=100, temperature=0.7)
        )
        vllm_time = time.time() - vllm_start
        vllm_tokens = len(vllm_output[0].outputs[0].token_ids)

        print(f"✅ vLLM: {vllm_tokens} tokens in {vllm_time:.2f}s")
        print(f"   Throughput: {vllm_tokens/vllm_time:.1f} tokens/sec")

        # HuggingFace comparison (would need transformers installed)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("\n2️⃣ Testing HuggingFace Transformers...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            hf_start = time.time()
            inputs = hf_tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = hf_model.generate(**inputs, max_new_tokens=100)
            hf_time = time.time() - hf_start
            hf_tokens = outputs.shape[1]

            print(f"✅ HuggingFace: {hf_tokens} tokens in {hf_time:.2f}s")
            print(f"   Throughput: {hf_tokens/hf_time:.1f} tokens/sec")

            # Comparison
            speedup = (hf_tokens/hf_time) / (vllm_tokens/vllm_time)
            print(f"\n🚀 vLLM Speedup: {1/speedup:.1f}x faster!")

            # Cleanup
            del hf_model
            torch.cuda.empty_cache()

        except ImportError:
            print("⚠️  transformers not installed, skipping comparison")
        except Exception as e:
            print(f"⚠️  Comparison failed: {e}")


def main():
    """Run basic vLLM examples"""

    print("=" * 60)
    print("vLLM Basic Examples - Blog Series Demo")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("⚠️  No GPU detected, vLLM requires CUDA GPU\n")
        return

    # Initialize vLLM
    example = BasicVLLMExample(model_name="microsoft/phi-2")

    if not VLLM_AVAILABLE:
        print("\n❌ Please install vLLM: pip install vllm")
        return

    # Example 1: Simple generation
    print("\n" + "="*60)
    print("Example 1: Simple Text Generation")
    print("="*60)
    example.simple_generation("Explain what vLLM is in simple terms.")

    # Example 2: Batch generation
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    prompts = [
        "What is PagedAttention?",
        "Why is vLLM faster than transformers?",
        "How does continuous batching work?",
        "What are the benefits of vLLM for production?",
        "Explain tensor parallelism.",
    ]
    example.batch_generation(prompts)

    # Example 3: Streaming (simulated)
    print("\n" + "="*60)
    print("Example 3: Streaming Generation")
    print("="*60)
    example.streaming_generation("Write a short poem about AI inference.")

    # Example 4: Performance comparison
    print("\n" + "="*60)
    print("Example 4: Performance Comparison")
    print("="*60)
    example.compare_with_transformers("Explain machine learning.")

    print("\n" + "="*60)
    print("✅ All examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
