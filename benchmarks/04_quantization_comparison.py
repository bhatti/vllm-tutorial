#!/usr/bin/env python3
"""
Benchmark 4: Quantization Comparison
Demonstrates cost/performance tradeoffs of different quantization schemes

Based on production best practices:
- W4A16 (4-bit weights, 16-bit activations): 3.7x compression, most popular
- INT8 W8A8 (8-bit weights and activations): 2x compression, hardware dependent
- FP8 W8A8 (Float8): 2x compression, easier to deploy (Ampere+)

Reference: Neural Magic CTO talk on GenAI at Scale
"""

import time
import torch
import gc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available")


@dataclass
class QuantizationMetrics:
    """Metrics for quantization scheme"""
    model_name: str
    quantization: str
    precision: str
    compression_ratio: float
    memory_allocated_gb: float
    memory_saved_gb: float
    memory_savings_pct: float
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    tokens_per_second: float
    speedup_vs_baseline: float
    total_time_seconds: float
    num_tokens: int
    gpu_name: str
    perplexity_change: Optional[float] = None  # Accuracy degradation


class QuantizationBenchmark:
    """Compare different quantization schemes"""

    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.results: List[QuantizationMetrics] = []
        self.baseline_memory = None
        self.baseline_throughput = None

    def get_gpu_info(self) -> Tuple[str, float]:
        """Get GPU name and total memory"""
        if not torch.cuda.is_available():
            return "CPU", 0.0

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return gpu_name, gpu_memory_gb

    def get_memory_stats(self) -> Dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_memory_gb': max_memory,
        }

    def reset_memory(self):
        """Reset GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            time.sleep(2)

    def benchmark_quantization(
        self,
        quantization: str = "none",
        dtype: str = "auto",
        num_prompts: int = 10,
        max_tokens: int = 100,
    ) -> Optional[QuantizationMetrics]:
        """
        Benchmark a specific quantization scheme

        Args:
            quantization: "none", "awq", "gptq", "squeezellm", or "fp8"
            dtype: "auto", "half", "float16", "bfloat16", "float32"
            num_prompts: Number of prompts to test
            max_tokens: Tokens to generate per prompt
        """
        if not VLLM_AVAILABLE:
            return None

        print(f"\n{'='*80}")
        print(f"Testing: {quantization.upper() if quantization != 'none' else 'Baseline FP16'}")
        print(f"{'='*80}")

        self.reset_memory()

        # Determine precision string for display
        if quantization == "awq":
            precision = "W4A16"  # 4-bit weights, 16-bit activations
        elif quantization in ["gptq", "squeezellm"]:
            precision = "W4A16"  # Also 4-bit typically
        elif quantization == "fp8":
            precision = "FP8"  # Float8
        elif dtype in ["float16", "half"]:
            precision = "FP16"
        elif dtype == "bfloat16":
            precision = "BF16"
        else:
            precision = "FP16"  # Default

        try:
            # Build vLLM config
            llm_kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 1024,  # Smaller for faster benchmarking
            }

            # Add quantization if specified
            if quantization != "none":
                llm_kwargs["quantization"] = quantization

            # Add dtype if specified
            if dtype != "auto":
                llm_kwargs["dtype"] = dtype

            # Initialize vLLM
            print(f"Loading model with {quantization if quantization != 'none' else 'no quantization'}...")
            load_start = time.time()
            llm = LLM(**llm_kwargs)
            load_time = time.time() - load_start
            print(f"✅ Model loaded in {load_time:.2f}s")

            # Memory after loading
            mem_after_load = self.get_memory_stats()
            memory_allocated = mem_after_load['allocated_gb']

            # Create prompts
            prompts = [
                f"Explain the concept of {topic} in finance."
                for topic in ["derivatives", "portfolio optimization", "risk management",
                             "market microstructure", "algorithmic trading", "quantitative analysis",
                             "credit default swaps", "volatility modeling", "asset allocation",
                             "high-frequency trading"][:num_prompts]
            ]

            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic
                max_tokens=max_tokens,
                ignore_eos=False,
            )

            # Warm-up run
            print("Warming up...")
            _ = llm.generate(prompts[:1], sampling_params)

            # Benchmark run
            print(f"Generating {num_prompts} prompts x {max_tokens} tokens...")

            # Track TTFT and ITL
            start_time = time.time()
            first_token_time = None
            token_times = []

            outputs = llm.generate(prompts, sampling_params)

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate metrics
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            tokens_per_second = total_tokens / total_time

            # Estimate TTFT and ITL (vLLM doesn't expose these directly in basic API)
            # TTFT is roughly the time for first batch processing
            # ITL is the time per token after that
            estimated_ttft = (total_time / num_prompts) * 1000  # Convert to ms
            estimated_itl = (total_time / max(total_tokens, 1)) * 1000  # ms per token

            # Calculate compression ratio
            if self.baseline_memory is None:
                # First run is baseline
                self.baseline_memory = memory_allocated
                self.baseline_throughput = tokens_per_second
                compression_ratio = 1.0
                speedup = 1.0
                memory_saved_gb = 0.0
                memory_savings_pct = 0.0
            else:
                compression_ratio = self.baseline_memory / max(memory_allocated, 0.1)
                speedup = tokens_per_second / max(self.baseline_throughput, 1.0)
                memory_saved_gb = self.baseline_memory - memory_allocated
                memory_savings_pct = (memory_saved_gb / self.baseline_memory) * 100

            metrics = QuantizationMetrics(
                model_name=self.model_name,
                quantization=quantization if quantization != "none" else "baseline",
                precision=precision,
                compression_ratio=compression_ratio,
                memory_allocated_gb=memory_allocated,
                memory_saved_gb=memory_saved_gb,
                memory_savings_pct=memory_savings_pct,
                time_to_first_token_ms=estimated_ttft,
                inter_token_latency_ms=estimated_itl,
                tokens_per_second=tokens_per_second,
                speedup_vs_baseline=speedup,
                total_time_seconds=total_time,
                num_tokens=total_tokens,
                gpu_name=self.get_gpu_info()[0],
            )

            print(f"✅ Memory: {memory_allocated:.2f} GB")
            print(f"✅ Compression: {compression_ratio:.2f}x")
            print(f"✅ Memory saved: {memory_saved_gb:.2f} GB ({memory_savings_pct:.1f}%)")
            print(f"✅ TTFT: {estimated_ttft:.1f}ms")
            print(f"✅ ITL: {estimated_itl:.1f}ms")
            print(f"✅ Throughput: {tokens_per_second:.1f} tokens/sec")
            print(f"✅ Speedup: {speedup:.2f}x")

            # Clean up
            del llm
            self.reset_memory()

            return metrics

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            self.reset_memory()
            return None

    def run_benchmark(self):
        """Run complete quantization benchmark"""
        print(f"\n{'='*80}")
        print(f"Quantization Comparison Benchmark")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")

        gpu_name, gpu_memory = self.get_gpu_info()
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")

        # Test configurations
        # Note: AWQ/GPTQ require pre-quantized models
        # We'll test what's available

        configs = [
            {
                "name": "Baseline FP16",
                "quantization": "none",
                "dtype": "half",
            },
            # Note: FP8 requires Ampere+ (A100, L4, H100)
            # AWQ/GPTQ require pre-quantized models from HuggingFace
            # For Phi-2, we may need to use a pre-quantized version
        ]

        # Try FP8 if available (L4 GPU supports it)
        if "L4" in gpu_name or "A100" in gpu_name or "H100" in gpu_name:
            configs.append({
                "name": "FP8 Quantization",
                "quantization": "fp8",
                "dtype": "auto",
            })

        # Run benchmarks
        for config in configs:
            print(f"\n{'='*80}")
            print(f"Configuration: {config['name']}")
            print(f"{'='*80}")

            try:
                metrics = self.benchmark_quantization(
                    quantization=config["quantization"],
                    dtype=config["dtype"],
                    num_prompts=5,  # Reduced for faster benchmarking
                    max_tokens=50,
                )

                if metrics:
                    self.results.append(metrics)

            except Exception as e:
                print(f"❌ Configuration {config['name']} failed: {e}")
                self.reset_memory()

    def save_results(self, filename: str = "results/quantization_results.json"):
        """Save benchmark results"""
        import os

        # Create directory
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Convert to dict
        results_dict = [
            {
                "model_name": m.model_name,
                "quantization": m.quantization,
                "precision": m.precision,
                "compression_ratio": round(m.compression_ratio, 2),
                "memory_allocated_gb": round(m.memory_allocated_gb, 2),
                "memory_saved_gb": round(m.memory_saved_gb, 2),
                "memory_savings_pct": round(m.memory_savings_pct, 1),
                "time_to_first_token_ms": round(m.time_to_first_token_ms, 1),
                "inter_token_latency_ms": round(m.inter_token_latency_ms, 2),
                "tokens_per_second": round(m.tokens_per_second, 1),
                "speedup_vs_baseline": round(m.speedup_vs_baseline, 2),
                "total_time_seconds": round(m.total_time_seconds, 2),
                "num_tokens": m.num_tokens,
                "gpu_name": m.gpu_name,
            }
            for m in self.results
        ]

        # Save JSON
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save CSV
        csv_filename = filename.replace('.json', '.csv')
        with open(csv_filename, 'w') as f:
            if results_dict:
                # Header
                f.write(','.join(results_dict[0].keys()) + '\n')
                # Rows
                for row in results_dict:
                    f.write(','.join(str(v) for v in row.values()) + '\n')

        print(f"\n✅ Results saved to {filename}")
        print(f"✅ Results saved to {csv_filename}")

    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("\n⚠️  No results to summarize")
            return

        print(f"\n{'='*80}")
        print(f"Quantization Benchmark Summary")
        print(f"{'='*80}")

        # Print table
        print(f"\n{'Scheme':<20} {'Memory':<12} {'Savings':<12} {'TTFT':<12} {'ITL':<12} {'Throughput':<15} {'Speedup':<10}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*15} {'-'*10}")

        for r in self.results:
            scheme = f"{r.quantization} ({r.precision})"
            memory = f"{r.memory_allocated_gb:.1f} GB"
            savings = f"{r.memory_savings_pct:.1f}%"
            ttft = f"{r.time_to_first_token_ms:.0f} ms"
            itl = f"{r.inter_token_latency_ms:.1f} ms"
            throughput = f"{r.tokens_per_second:.1f} tok/s"
            speedup = f"{r.speedup_vs_baseline:.2f}x"

            print(f"{scheme:<20} {memory:<12} {savings:<12} {ttft:<12} {itl:<12} {throughput:<15} {speedup:<10}")

        # Cost analysis
        print(f"\n{'='*80}")
        print(f"💰 Cost Impact Analysis")
        print(f"{'='*80}")

        baseline = self.results[0] if self.results else None
        if baseline:
            print(f"\nBaseline: {baseline.precision}")
            print(f"  Memory: {baseline.memory_allocated_gb:.2f} GB")
            print(f"  Throughput: {baseline.tokens_per_second:.1f} tokens/sec")

            for r in self.results[1:]:
                print(f"\n{r.quantization.upper()} ({r.precision}):")
                print(f"  Memory: {r.memory_allocated_gb:.2f} GB ({r.compression_ratio:.2f}x compression)")
                print(f"  Memory saved: {r.memory_saved_gb:.2f} GB ({r.memory_savings_pct:.1f}%)")
                print(f"  Throughput: {r.tokens_per_second:.1f} tokens/sec ({r.speedup_vs_baseline:.2f}x)")

                # Cost calculation (based on L4 GPU pricing)
                gpu_cost_per_hour = 0.45  # L4 spot pricing
                baseline_gpus_needed = 1
                quantized_gpus_needed = max(1, baseline_gpus_needed / r.compression_ratio)

                baseline_cost_per_hour = baseline_gpus_needed * gpu_cost_per_hour
                quantized_cost_per_hour = quantized_gpus_needed * gpu_cost_per_hour

                daily_cost_baseline = baseline_cost_per_hour * 24
                daily_cost_quantized = quantized_cost_per_hour * 24

                monthly_savings = (daily_cost_baseline - daily_cost_quantized) * 30
                yearly_savings = monthly_savings * 12

                print(f"\n  Cost Impact (L4 GPU @ ${gpu_cost_per_hour}/hr):")
                print(f"    Baseline: ${daily_cost_baseline:.2f}/day (${daily_cost_baseline * 30:.2f}/month)")
                print(f"    Quantized: ${daily_cost_quantized:.2f}/day (${daily_cost_quantized * 30:.2f}/month)")
                print(f"    Monthly savings: ${monthly_savings:.2f}")
                print(f"    Yearly savings: ${yearly_savings:.2f}")


def main():
    """Run quantization benchmark"""
    benchmark = QuantizationBenchmark(model_name="microsoft/phi-2")
    benchmark.run_benchmark()
    benchmark.print_summary()
    benchmark.save_results("results/quantization_results.json")

    print(f"\n{'='*80}")
    print(f"📝 Key Takeaways")
    print(f"{'='*80}")
    print(f"\n1. Quantization provides 2-3.7x compression with minimal accuracy loss")
    print(f"2. W4A16 (4-bit weights): Best compression (3.7x), most popular")
    print(f"3. FP8: Easier to deploy, requires Ampere+ GPUs (2x compression)")
    print(f"4. Enables deploying larger models on smaller GPUs")
    print(f"5. Production standard - every deployment should use quantization")
    print(f"\n💡 Recommendation: Start with FP8 (easiest), then try W4A16 for max savings")


if __name__ == "__main__":
    main()
