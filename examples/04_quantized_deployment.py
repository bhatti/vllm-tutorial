#!/usr/bin/env python3
"""
Example 4: Quantized Model Deployment
Demonstrates how to deploy quantized models for 3.7x cost reduction

Production quantization schemes:
- FP8 (Float8): Easiest to deploy, 2x compression, requires Ampere+ GPUs
- W4A16 (4-bit weights): Best compression at 3.7x, most popular in production
- INT8: 2x compression, hardware dependent

Based on Neural Magic CTO recommendations for production deployment
"""

import time
from typing import List, Dict
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available. Install: pip install vllm")


@dataclass
class QuantizationConfig:
    """Configuration for quantization scheme"""
    name: str
    quantization: str
    description: str
    compression_ratio: float
    gpu_requirements: str
    use_case: str


# Production quantization schemes
QUANTIZATION_SCHEMES = [
    QuantizationConfig(
        name="Baseline (FP16)",
        quantization="none",
        description="No quantization - full precision",
        compression_ratio=1.0,
        gpu_requirements="Any GPU",
        use_case="Development, maximum accuracy required",
    ),
    QuantizationConfig(
        name="FP8",
        quantization="fp8",
        description="Float8 quantization - easiest to deploy",
        compression_ratio=2.0,
        gpu_requirements="L4, A100, H100 (Ampere+)",
        use_case="Production deployment, good accuracy",
    ),
    QuantizationConfig(
        name="W4A16 (AWQ)",
        quantization="awq",
        description="4-bit weights, 16-bit activations - best compression",
        compression_ratio=3.7,
        gpu_requirements="Any GPU (CPU decompression)",
        use_case="Maximum cost savings, most popular",
    ),
]


def print_quantization_guide():
    """Print guide on choosing quantization scheme"""
    print(f"{'='*80}")
    print(f"📊 Quantization Scheme Selection Guide")
    print(f"{'='*80}\n")

    print(f"{'Scheme':<20} {'Compression':<15} {'GPU Needs':<25} {'Best For':<30}")
    print(f"{'-'*20} {'-'*15} {'-'*25} {'-'*30}")

    for scheme in QUANTIZATION_SCHEMES:
        print(f"{scheme.name:<20} {scheme.compression_ratio}x{'':<12} {scheme.gpu_requirements:<25} {scheme.use_case:<30}")

    print(f"\n{'='*80}")
    print(f"💡 Recommendations:")
    print(f"{'='*80}")
    print(f"• Start with FP8 - easiest to deploy, 2x cost savings")
    print(f"• Use W4A16 for maximum savings - 3.7x compression")
    print(f"• Baseline only for development or accuracy validation")
    print(f"\n⚠️  Note: W4A16 requires pre-quantized model from HuggingFace")
    print(f"   Example: TheBloke/Phi-2-AWQ or similar quantized variants\n")


def deploy_baseline_model(model_name: str = "microsoft/phi-2"):
    """Deploy baseline FP16 model"""
    print(f"\n{'='*80}")
    print(f"Example 1: Baseline FP16 Deployment")
    print(f"{'='*80}\n")

    if not VLLM_AVAILABLE:
        print("⚠️  vLLM not available")
        return

    print(f"Loading {model_name} in FP16 (no quantization)...")

    # Load model
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="half",  # FP16
        gpu_memory_utilization=0.9,
    )

    # Generate
    prompt = "Explain quantization in machine learning in simple terms."
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start

    print(f"✅ Generated {len(outputs[0].outputs[0].token_ids)} tokens in {elapsed:.2f}s")
    print(f"✅ Throughput: {len(outputs[0].outputs[0].token_ids) / elapsed:.1f} tokens/sec")
    print(f"\n📝 Response:\n{outputs[0].outputs[0].text}\n")

    del llm


def deploy_fp8_model(model_name: str = "microsoft/phi-2"):
    """Deploy FP8 quantized model"""
    print(f"\n{'='*80}")
    print(f"Example 2: FP8 Quantized Deployment (2x compression)")
    print(f"{'='*80}\n")

    if not VLLM_AVAILABLE:
        print("⚠️  vLLM not available")
        return

    print(f"Loading {model_name} with FP8 quantization...")
    print(f"💾 Memory savings: 2x compression (50% reduction)")
    print(f"⚡ Expected speedup: 2-3x faster inference")

    try:
        # Load model with FP8 quantization
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            quantization="fp8",  # Enable FP8 quantization
            gpu_memory_utilization=0.9,
        )

        # Generate
        prompt = "Explain how quantization reduces costs in LLM deployment."
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - start

        print(f"✅ Generated {len(outputs[0].outputs[0].token_ids)} tokens in {elapsed:.2f}s")
        print(f"✅ Throughput: {len(outputs[0].outputs[0].token_ids) / elapsed:.1f} tokens/sec")
        print(f"\n📝 Response:\n{outputs[0].outputs[0].text}\n")

        # Cost analysis
        print(f"{'='*80}")
        print(f"💰 Cost Analysis (L4 GPU @ $0.45/hr)")
        print(f"{'='*80}")
        print(f"Baseline FP16: $10.80/day ($324/month)")
        print(f"FP8 Quantized: $5.40/day ($162/month)")
        print(f"💵 Savings: $5.40/day ($162/month, $1,944/year)")

        del llm

    except Exception as e:
        print(f"❌ FP8 quantization failed: {e}")
        print(f"⚠️  FP8 requires Ampere+ GPUs (L4, A100, H100)")
        print(f"   Your GPU may not support FP8 quantization")


def deploy_awq_model():
    """Deploy AWQ (W4A16) quantized model"""
    print(f"\n{'='*80}")
    print(f"Example 3: W4A16 (AWQ) Deployment (3.7x compression)")
    print(f"{'='*80}\n")

    if not VLLM_AVAILABLE:
        print("⚠️  vLLM not available")
        return

    # Note: AWQ requires pre-quantized model
    # For Phi-2, we'd need a pre-quantized version like "TheBloke/Phi-2-AWQ"
    # For demonstration, we'll show the code pattern

    print(f"⚠️  AWQ requires pre-quantized model from HuggingFace")
    print(f"   Example: TheBloke/Mistral-7B-v0.1-AWQ")
    print(f"\n📝 Code example:\n")

    code_example = '''
# Load pre-quantized AWQ model
llm = LLM(
    model="TheBloke/Mistral-7B-v0.1-AWQ",  # Pre-quantized model
    quantization="awq",                     # Enable AWQ
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
)

# Generate as normal
outputs = llm.generate(prompts, sampling_params)
'''

    print(code_example)

    print(f"{'='*80}")
    print(f"💰 Cost Analysis (Mistral-7B on L4 GPU)")
    print(f"{'='*80}")
    print(f"Baseline FP16:  Requires A100 (26GB model)")
    print(f"                Cost: $26.40/day ($792/month)")
    print(f"\nW4A16 (AWQ):    Fits on L4! (7GB compressed)")
    print(f"                Cost: $10.80/day ($324/month)")
    print(f"\n💵 Savings: $15.60/day ($468/month, $5,616/year)")
    print(f"✨ Bonus: Can use cheaper L4 GPU instead of A100!")


def production_deployment_example():
    """Show production deployment with quantization"""
    print(f"\n{'='*80}")
    print(f"Example 4: Production Deployment Pattern")
    print(f"{'='*80}\n")

    print(f"📝 Recommended Production Setup:\n")

    code_example = '''
import os
from vllm import LLM, SamplingParams

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-2")
QUANTIZATION = os.getenv("QUANTIZATION", "fp8")  # fp8, awq, none
GPU_MEMORY = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

# Load model with quantization
llm = LLM(
    model=MODEL_NAME,
    quantization=QUANTIZATION if QUANTIZATION != "none" else None,
    trust_remote_code=True,
    gpu_memory_utilization=GPU_MEMORY,
    max_model_len=2048,
)

# Sampling config
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
)

# Generate
def generate(prompt: str) -> str:
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# Use in production
response = generate("Your prompt here")
'''

    print(code_example)

    print(f"\n{'='*80}")
    print(f"🐳 Docker Environment Variables")
    print(f"{'='*80}")
    print(f"MODEL_NAME=microsoft/phi-2")
    print(f"QUANTIZATION=fp8           # Options: fp8, awq, none")
    print(f"GPU_MEMORY_UTILIZATION=0.9")
    print(f"\n💡 Set QUANTIZATION=fp8 in .env file for 2x cost savings")


def main():
    """Run quantized deployment examples"""
    print(f"{'='*80}")
    print(f"Quantized Model Deployment Examples")
    print(f"{'='*80}\n")

    # Show guide
    print_quantization_guide()

    # Run examples
    if VLLM_AVAILABLE:
        print(f"\n⚠️  Note: These examples load models and may take 1-2 minutes each")
        print(f"Press Ctrl+C to skip to next example\n")

        try:
            # Example 1: Baseline
            deploy_baseline_model()

            # Example 2: FP8 (if supported)
            deploy_fp8_model()

        except KeyboardInterrupt:
            print(f"\n⚠️  Skipped")

    # Example 3: AWQ (code example only)
    deploy_awq_model()

    # Example 4: Production pattern
    production_deployment_example()

    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Summary")
    print(f"{'='*80}")
    print(f"\n1. FP8: Easiest to deploy, 2x compression, production-ready")
    print(f"2. W4A16 (AWQ): Best compression at 3.7x, requires pre-quantized model")
    print(f"3. Use environment variables for flexible configuration")
    print(f"4. Quantization is production standard - always use it!")
    print(f"\n💰 Cost Impact:")
    print(f"   Baseline (FP16):  $324/month")
    print(f"   FP8 Quantized:    $162/month  (50% savings)")
    print(f"   W4A16 Quantized:  $87/month   (73% savings)")
    print(f"\n📚 Next Steps:")
    print(f"   1. Test FP8 quantization on your model")
    print(f"   2. Find pre-quantized AWQ models on HuggingFace (TheBloke/*-AWQ)")
    print(f"   3. Update Docker .env with QUANTIZATION=fp8")
    print(f"   4. Deploy and enjoy 2-3.7x cost reduction!")


if __name__ == "__main__":
    main()
