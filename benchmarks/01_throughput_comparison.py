#!/usr/bin/env python3
"""
Benchmark 1: Throughput Comparison
Compare vLLM vs HuggingFace Transformers throughput
"""

import time
import torch
import pandas as pd
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("⚠️  vLLM not installed")
    VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not installed")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Benchmark result"""
    framework: str
    model_name: str
    batch_size: int
    total_prompts: int
    total_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    requests_per_second: float
    avg_latency_ms: float
    memory_used_gb: float


class ThroughputBenchmark:
    """Comprehensive throughput benchmark"""

    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.results: List[BenchmarkResult] = []

    def get_test_prompts(self, count: int) -> List[str]:
        """Generate test prompts"""
        base_prompts = [
            "Explain machine learning in simple terms.",
            "What are the benefits of cloud computing?",
            "Describe the process of photosynthesis.",
            "How does blockchain technology work?",
            "What is quantum computing?",
            "Explain the theory of relativity.",
            "What causes climate change?",
            "How do neural networks learn?",
            "What is the difference between AI and ML?",
            "Describe how vaccines work.",
        ]

        # Repeat prompts to get desired count
        prompts = (base_prompts * (count // len(base_prompts) + 1))[:count]
        return prompts

    def benchmark_vllm(self, batch_size: int, num_batches: int = 5) -> BenchmarkResult:
        """Benchmark vLLM throughput"""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available")

        print(f"\n{'='*60}")
        print(f"🚀 vLLM Benchmark - Batch Size: {batch_size}")
        print(f"{'='*60}")

        # Load model
        print("Loading model...")
        llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            trust_remote_code=True
        )

        # Generate prompts
        total_prompts = batch_size * num_batches
        all_prompts = self.get_test_prompts(total_prompts)

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
            top_p=0.95
        )

        # Warm-up
        print("Warming up...")
        _ = llm.generate(all_prompts[:2], sampling_params)
        torch.cuda.synchronize()

        # Benchmark
        print(f"Benchmarking {total_prompts} prompts...")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        outputs = llm.generate(all_prompts, sampling_params)

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        memory_used = torch.cuda.max_memory_allocated() / 1e9

        result = BenchmarkResult(
            framework="vLLM",
            model_name=self.model_name,
            batch_size=batch_size,
            total_prompts=total_prompts,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            tokens_per_second=total_tokens / total_time,
            requests_per_second=total_prompts / total_time,
            avg_latency_ms=(total_time / total_prompts) * 1000,
            memory_used_gb=memory_used
        )

        print(f"\n📊 Results:")
        print(f"  Throughput: {result.tokens_per_second:.1f} tokens/sec")
        print(f"  Requests/sec: {result.requests_per_second:.2f}")
        print(f"  Avg latency: {result.avg_latency_ms:.1f}ms")
        print(f"  Memory used: {result.memory_used_gb:.2f}GB")

        self.results.append(result)
        return result

    def benchmark_transformers(self, batch_size: int, num_batches: int = 2) -> BenchmarkResult:
        """Benchmark HuggingFace Transformers (limited to avoid OOM)"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")

        print(f"\n{'='*60}")
        print(f"🤗 Transformers Benchmark - Batch Size: {batch_size}")
        print(f"{'='*60}")

        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Generate prompts (fewer for transformers)
        total_prompts = min(batch_size * num_batches, 20)  # Limit to avoid OOM
        all_prompts = self.get_test_prompts(total_prompts)

        # Warm-up
        print("Warming up...")
        inputs = tokenizer(all_prompts[0], return_tensors="pt").to("cuda")
        _ = model.generate(**inputs, max_new_tokens=10)
        torch.cuda.synchronize()

        # Benchmark (process one at a time for transformers)
        print(f"Benchmarking {total_prompts} prompts (sequential)...")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        total_tokens = 0
        for prompt in all_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
            total_tokens += outputs.shape[1]

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        memory_used = torch.cuda.max_memory_allocated() / 1e9

        result = BenchmarkResult(
            framework="Transformers",
            model_name=self.model_name,
            batch_size=1,  # Transformers processes sequentially
            total_prompts=total_prompts,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            tokens_per_second=total_tokens / total_time,
            requests_per_second=total_prompts / total_time,
            avg_latency_ms=(total_time / total_prompts) * 1000,
            memory_used_gb=memory_used
        )

        print(f"\n📊 Results:")
        print(f"  Throughput: {result.tokens_per_second:.1f} tokens/sec")
        print(f"  Requests/sec: {result.requests_per_second:.2f}")
        print(f"  Avg latency: {result.avg_latency_ms:.1f}ms")
        print(f"  Memory used: {result.memory_used_gb:.2f}GB")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        self.results.append(result)
        return result

    def run_full_benchmark(self):
        """Run comprehensive throughput benchmark"""
        print("=" * 80)
        print("🔬 Comprehensive Throughput Benchmark")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Test different batch sizes with vLLM
        # Use batch sizes that work well on L4 (24GB): 1, 16, 32, 50
        if VLLM_AVAILABLE:
            for batch_size in [1, 16, 32, 50]:
                try:
                    self.benchmark_vllm(batch_size=batch_size, num_batches=1)
                    torch.cuda.empty_cache()
                    time.sleep(2)
                except Exception as e:
                    print(f"❌ vLLM batch_size={batch_size} failed: {e}")
                    torch.cuda.empty_cache()

        # Test transformers (limited)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.benchmark_transformers(batch_size=1, num_batches=2)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ Transformers benchmark failed: {e}")

    def save_results(self, filename: str = "throughput_results.json"):
        """Save results to file"""
        import os

        # Create directory if it doesn't exist
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        results_dict = [asdict(r) for r in self.results]

        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n💾 Results saved to {filename}")

        # Also save as CSV
        df = pd.DataFrame(results_dict)
        csv_filename = filename.replace('.json', '.csv')
        df.to_csv(csv_filename, index=False)
        print(f"💾 Results saved to {csv_filename}")

    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "="*80)
        print("📊 Benchmark Summary")
        print("="*80)

        df = pd.DataFrame([asdict(r) for r in self.results])

        print("\n" + df.to_string(index=False))

        # Calculate speedup
        vllm_results = [r for r in self.results if r.framework == "vLLM"]
        hf_results = [r for r in self.results if r.framework == "Transformers"]

        if vllm_results and hf_results:
            vllm_best = max(vllm_results, key=lambda r: r.tokens_per_second)
            hf_best = max(hf_results, key=lambda r: r.tokens_per_second)

            speedup = vllm_best.tokens_per_second / hf_best.tokens_per_second

            print("\n" + "="*80)
            print("🚀 Performance Comparison")
            print("="*80)
            print(f"vLLM (best):        {vllm_best.tokens_per_second:.1f} tokens/sec")
            print(f"Transformers (best): {hf_best.tokens_per_second:.1f} tokens/sec")
            print(f"Speedup:            {speedup:.1f}x")
            print(f"\n💰 For 1M tokens:")
            print(f"  vLLM time:        {1_000_000 / vllm_best.tokens_per_second:.1f}s")
            print(f"  Transformers time: {1_000_000 / hf_best.tokens_per_second:.1f}s")
            print(f"  Time saved:       {(1_000_000 / hf_best.tokens_per_second) - (1_000_000 / vllm_best.tokens_per_second):.1f}s")


def main():
    """Run throughput benchmark"""
    benchmark = ThroughputBenchmark(model_name="microsoft/phi-2")

    try:
        benchmark.run_full_benchmark()
        benchmark.print_summary()
        benchmark.save_results("results/throughput_results.json")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Throughput benchmark complete!")


if __name__ == "__main__":
    main()
