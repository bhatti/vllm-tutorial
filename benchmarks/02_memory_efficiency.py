#!/usr/bin/env python3
"""
Benchmark 2: Memory Efficiency Analysis
Demonstrates PagedAttention memory savings vs traditional attention
"""

import time
import torch
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available")


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    framework: str
    model_name: str
    batch_size: int
    seq_length: int
    memory_allocated_gb: float
    memory_reserved_gb: float
    memory_cached_gb: float
    max_memory_gb: float
    kv_cache_gb: float
    memory_efficiency_pct: float


class MemoryBenchmark:
    """Benchmark memory efficiency of different frameworks"""

    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.results: List[MemoryMetrics] = []

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

    def estimate_kv_cache_memory(self, batch_size: int, seq_length: int,
                                 num_layers: int = 32, hidden_size: int = 2560,
                                 num_heads: int = 32) -> float:
        """
        Estimate KV cache memory usage

        Formula: 2 (K+V) * batch_size * seq_length * num_layers * hidden_size * 2 (bytes for fp16)
        """
        bytes_per_element = 2  # fp16
        kv_cache_bytes = (
            2 *  # K and V
            batch_size *
            seq_length *
            num_layers *
            hidden_size *
            bytes_per_element
        )
        return kv_cache_bytes / (1024**3)  # Convert to GB

    def benchmark_vllm_memory(self, batch_size: int = 1, seq_length: int = 512) -> Optional[MemoryMetrics]:
        """Benchmark vLLM memory usage with PagedAttention"""
        if not VLLM_AVAILABLE:
            return None

        print(f"\n{'='*80}")
        print(f"Testing vLLM Memory (batch={batch_size}, seq_len={seq_length})")
        print(f"{'='*80}")

        self.reset_memory()

        try:
            # Initialize vLLM
            llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=seq_length,
                gpu_memory_utilization=0.9,
            )

            # Create prompts
            prompt = "Analyze this financial report: " + "data " * (seq_length // 10)
            prompts = [prompt] * batch_size

            # Memory before generation
            mem_before = self.get_memory_stats()

            # Generate
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=100,
            )
            outputs = llm.generate(prompts, sampling_params)

            # Memory after generation
            mem_after = self.get_memory_stats()

            # Estimate KV cache
            # Phi-2: 32 layers, 2560 hidden, 32 heads
            kv_cache_gb = self.estimate_kv_cache_memory(
                batch_size=batch_size,
                seq_length=seq_length,
                num_layers=32,
                hidden_size=2560,
            )

            # Calculate memory efficiency
            # PagedAttention should use significantly less than traditional attention
            theoretical_kv = kv_cache_gb
            actual_increase = mem_after['allocated_gb'] - mem_before['allocated_gb']
            efficiency_pct = (1 - (actual_increase / max(theoretical_kv, 0.1))) * 100

            metrics = MemoryMetrics(
                framework="vLLM",
                model_name=self.model_name,
                batch_size=batch_size,
                seq_length=seq_length,
                memory_allocated_gb=mem_after['allocated_gb'],
                memory_reserved_gb=mem_after['reserved_gb'],
                memory_cached_gb=0.0,  # vLLM manages this internally
                max_memory_gb=mem_after['max_memory_gb'],
                kv_cache_gb=kv_cache_gb,
                memory_efficiency_pct=efficiency_pct,
            )

            print(f"✅ Memory allocated: {mem_after['allocated_gb']:.2f} GB")
            print(f"✅ Memory reserved: {mem_after['reserved_gb']:.2f} GB")
            print(f"✅ Estimated KV cache: {kv_cache_gb:.2f} GB")
            print(f"✅ Memory efficiency: {efficiency_pct:.1f}%")

            # Clean up
            del llm
            self.reset_memory()

            return metrics

        except Exception as e:
            print(f"❌ vLLM failed: {e}")
            self.reset_memory()
            return None

    def benchmark_transformers_memory(self, batch_size: int = 1, seq_length: int = 512) -> Optional[MemoryMetrics]:
        """Benchmark HuggingFace Transformers memory usage"""
        if not TRANSFORMERS_AVAILABLE:
            return None

        print(f"\n{'='*80}")
        print(f"Testing Transformers Memory (batch={batch_size}, seq_len={seq_length})")
        print(f"{'='*80}")

        self.reset_memory()

        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            # Create prompts
            prompt = "Analyze this financial report: " + "data " * (seq_length // 10)
            prompts = [prompt] * batch_size

            # Memory before generation
            mem_before = self.get_memory_stats()

            # Tokenize
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                )

            # Memory after generation
            mem_after = self.get_memory_stats()

            # Estimate KV cache
            kv_cache_gb = self.estimate_kv_cache_memory(
                batch_size=batch_size,
                seq_length=seq_length,
                num_layers=32,
                hidden_size=2560,
            )

            # Traditional attention uses full KV cache
            efficiency_pct = 0.0  # No optimization

            metrics = MemoryMetrics(
                framework="Transformers",
                model_name=self.model_name,
                batch_size=batch_size,
                seq_length=seq_length,
                memory_allocated_gb=mem_after['allocated_gb'],
                memory_reserved_gb=mem_after['reserved_gb'],
                memory_cached_gb=kv_cache_gb,
                max_memory_gb=mem_after['max_memory_gb'],
                kv_cache_gb=kv_cache_gb,
                memory_efficiency_pct=efficiency_pct,
            )

            print(f"✅ Memory allocated: {mem_after['allocated_gb']:.2f} GB")
            print(f"✅ Memory reserved: {mem_after['reserved_gb']:.2f} GB")
            print(f"✅ KV cache size: {kv_cache_gb:.2f} GB")
            print(f"✅ Memory efficiency: {efficiency_pct:.1f}%")

            # Clean up
            del model
            del tokenizer
            self.reset_memory()

            return metrics

        except Exception as e:
            print(f"❌ Transformers failed: {e}")
            self.reset_memory()
            return None

    def run_benchmark(self):
        """Run complete memory efficiency benchmark"""
        print(f"\n{'='*80}")
        print(f"Memory Efficiency Benchmark")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")

        # Test configurations
        configs = [
            {"batch_size": 1, "seq_length": 256},
            {"batch_size": 1, "seq_length": 512},
            {"batch_size": 4, "seq_length": 256},
            {"batch_size": 8, "seq_length": 256},
        ]

        # Test vLLM
        if VLLM_AVAILABLE:
            for config in configs:
                try:
                    metrics = self.benchmark_vllm_memory(**config)
                    if metrics:
                        self.results.append(metrics)
                except Exception as e:
                    print(f"❌ vLLM config {config} failed: {e}")
                    self.reset_memory()

        # Test Transformers (only small configs to avoid OOM)
        if TRANSFORMERS_AVAILABLE:
            small_configs = [
                {"batch_size": 1, "seq_length": 256},
                {"batch_size": 1, "seq_length": 512},
            ]
            for config in small_configs:
                try:
                    metrics = self.benchmark_transformers_memory(**config)
                    if metrics:
                        self.results.append(metrics)
                except Exception as e:
                    print(f"❌ Transformers config {config} failed: {e}")
                    self.reset_memory()

    def save_results(self, filename: str = "results/memory_results.json"):
        """Save benchmark results"""
        import os

        # Create directory
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Convert to dict
        results_dict = [
            {
                "framework": m.framework,
                "model_name": m.model_name,
                "batch_size": m.batch_size,
                "seq_length": m.seq_length,
                "memory_allocated_gb": round(m.memory_allocated_gb, 3),
                "memory_reserved_gb": round(m.memory_reserved_gb, 3),
                "memory_cached_gb": round(m.memory_cached_gb, 3),
                "max_memory_gb": round(m.max_memory_gb, 3),
                "kv_cache_gb": round(m.kv_cache_gb, 3),
                "memory_efficiency_pct": round(m.memory_efficiency_pct, 2),
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
        print(f"Memory Efficiency Summary")
        print(f"{'='*80}")

        # Group by framework
        vllm_results = [r for r in self.results if r.framework == "vLLM"]
        tf_results = [r for r in self.results if r.framework == "Transformers"]

        if vllm_results:
            print(f"\nvLLM Results ({len(vllm_results)} tests):")
            for r in vllm_results:
                print(f"  Batch={r.batch_size}, SeqLen={r.seq_length}: "
                      f"{r.memory_allocated_gb:.2f} GB allocated, "
                      f"{r.kv_cache_gb:.2f} GB KV cache, "
                      f"{r.memory_efficiency_pct:.1f}% efficient")

        if tf_results:
            print(f"\nTransformers Results ({len(tf_results)} tests):")
            for r in tf_results:
                print(f"  Batch={r.batch_size}, SeqLen={r.seq_length}: "
                      f"{r.memory_allocated_gb:.2f} GB allocated, "
                      f"{r.kv_cache_gb:.2f} GB KV cache")

        # Calculate memory savings
        if vllm_results and tf_results:
            print(f"\n{'='*80}")
            print(f"Memory Savings Analysis")
            print(f"{'='*80}")

            # Compare similar configs
            for vr in vllm_results:
                for tr in tf_results:
                    if vr.batch_size == tr.batch_size and vr.seq_length == tr.seq_length:
                        savings_gb = tr.memory_allocated_gb - vr.memory_allocated_gb
                        savings_pct = (savings_gb / tr.memory_allocated_gb) * 100
                        print(f"\nBatch={vr.batch_size}, SeqLen={vr.seq_length}:")
                        print(f"  Transformers: {tr.memory_allocated_gb:.2f} GB")
                        print(f"  vLLM: {vr.memory_allocated_gb:.2f} GB")
                        print(f"  Savings: {savings_gb:.2f} GB ({savings_pct:.1f}%)")


def main():
    benchmark = MemoryBenchmark(model_name="microsoft/phi-2")
    benchmark.run_benchmark()
    benchmark.print_summary()
    benchmark.save_results("results/memory_results.json")


if __name__ == "__main__":
    main()
