"""
Comprehensive vLLM Benchmarking Suite
Compares vLLM vs Transformers performance across multiple dimensions
Following TDD principles with reproducible benchmarks
"""
import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from datetime import datetime

# Import libraries to benchmark
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import GPUtil


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs"""

    model_name: str = "microsoft/phi-2"
    prompt_lengths: List[int] = None  # [10, 50, 100, 500, 1000]
    output_lengths: List[int] = None  # [50, 100, 200, 500]
    batch_sizes: List[int] = None  # [1, 4, 8, 16, 32]
    num_iterations: int = 10
    warmup_iterations: int = 2
    temperature: float = 0.7
    top_p: float = 0.95
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.prompt_lengths is None:
            self.prompt_lengths = [10, 50, 100, 500, 1000]
        if self.output_lengths is None:
            self.output_lengths = [50, 100, 200, 500]
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    framework: str
    model_name: str
    prompt_length: int
    output_length: int
    batch_size: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    total_tokens: int
    timestamp: datetime


class BenchmarkSuite:
    """Main benchmarking suite for vLLM vs Transformers"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.vllm_model = None
        self.hf_model = None
        self.hf_tokenizer = None

    def setup(self):
        """Initialize models for benchmarking"""
        print(f"Setting up models: {self.config.model_name}")

        # Setup vLLM
        print("Loading vLLM model...")
        self.vllm_model = LLM(
            model=self.config.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

        # Setup Transformers
        print("Loading Transformers model...")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.config.device == "cuda":
            self.hf_model = self.hf_model.cuda()

        # Warmup
        print("Running warmup iterations...")
        self._warmup()

    def _warmup(self):
        """Warmup models before benchmarking"""
        warmup_prompt = "The quick brown fox jumps over the lazy dog."

        # Warmup vLLM
        for _ in range(self.config.warmup_iterations):
            self.vllm_model.generate(
                [warmup_prompt],
                SamplingParams(
                    temperature=self.config.temperature,
                    max_tokens=50,
                ),
            )

        # Warmup HuggingFace
        inputs = self.hf_tokenizer(warmup_prompt, return_tensors="pt")
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                self.hf_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=self.config.temperature,
                    do_sample=True,
                )

    def generate_prompts(self, prompt_length: int, batch_size: int) -> List[str]:
        """Generate test prompts of specified length"""
        base_prompts = [
            "Analyze the financial report showing",
            "The quarterly earnings indicate that",
            "Market volatility has increased due to",
            "Investment strategies should consider",
            "Risk assessment reveals the following",
        ]

        prompts = []
        for i in range(batch_size):
            base = base_prompts[i % len(base_prompts)]
            # Extend prompt to desired length
            words_needed = prompt_length - len(base.split())
            if words_needed > 0:
                extension = " financial data" * (words_needed // 2)
                prompt = base + extension
            else:
                prompt = " ".join(base.split()[:prompt_length])
            prompts.append(prompt)

        return prompts

    def benchmark_vllm(
        self,
        prompts: List[str],
        output_length: int,
    ) -> Tuple[float, float, float, float, float]:
        """Benchmark vLLM performance"""
        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=output_length,
        )

        # Measure initial state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        if self.config.device == "cuda":
            torch.cuda.synchronize()
            gpu_initial = GPUtil.getGPUs()[0].memoryUsed

        # Run inference
        start_time = time.perf_counter()
        outputs = self.vllm_model.generate(prompts, sampling_params)

        if self.config.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = (end_time - start_time) * 1000  # ms
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        # Calculate throughput
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / (total_time / 1000)  # tokens/sec

        # GPU utilization
        gpu_util = 0
        if self.config.device == "cuda":
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_util = gpus[0].load * 100
                gpu_memory = gpus[0].memoryUsed - gpu_initial

        # Estimate time to first token (simplified)
        time_to_first_token = total_time / (len(prompts) * 10)  # Rough estimate
        inter_token_latency = total_time / total_tokens if total_tokens > 0 else 0

        return (
            total_time,
            throughput,
            memory_used,
            gpu_util,
            time_to_first_token,
            inter_token_latency,
            total_tokens,
        )

    def benchmark_transformers(
        self,
        prompts: List[str],
        output_length: int,
    ) -> Tuple[float, float, float, float, float]:
        """Benchmark HuggingFace Transformers performance"""
        # Tokenize inputs
        inputs = self.hf_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Measure initial state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        if self.config.device == "cuda":
            torch.cuda.synchronize()
            gpu_initial = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

        # Run inference
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=output_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )

        if self.config.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Calculate metrics
        total_time = (end_time - start_time) * 1000
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        # Calculate throughput
        total_tokens = sum(
            len(output) - len(inputs.input_ids[i])
            for i, output in enumerate(outputs)
        )
        throughput = total_tokens / (total_time / 1000)

        # GPU utilization
        gpu_util = 0
        if self.config.device == "cuda":
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_util = gpus[0].load * 100
                gpu_memory = gpus[0].memoryUsed - gpu_initial

        # Estimate time to first token
        time_to_first_token = total_time / (len(prompts) * 10)
        inter_token_latency = total_time / total_tokens if total_tokens > 0 else 0

        return (
            total_time,
            throughput,
            memory_used,
            gpu_util,
            time_to_first_token,
            inter_token_latency,
            total_tokens,
        )

    def run_comparison(self):
        """Run complete comparison benchmark"""
        print("\n" + "=" * 60)
        print("Starting vLLM vs Transformers Benchmark")
        print("=" * 60)

        for prompt_length in self.config.prompt_lengths:
            for output_length in self.config.output_lengths:
                for batch_size in self.config.batch_sizes:
                    print(
                        f"\nBenchmarking: prompt_len={prompt_length}, "
                        f"output_len={output_length}, batch={batch_size}"
                    )

                    prompts = self.generate_prompts(prompt_length, batch_size)

                    # Benchmark vLLM
                    vllm_times = []
                    for i in range(self.config.num_iterations):
                        metrics = self.benchmark_vllm(prompts, output_length)
                        vllm_times.append(metrics[0])

                        if i == self.config.num_iterations - 1:  # Last iteration
                            self.results.append(
                                BenchmarkResult(
                                    framework="vLLM",
                                    model_name=self.config.model_name,
                                    prompt_length=prompt_length,
                                    output_length=output_length,
                                    batch_size=batch_size,
                                    latency_ms=np.mean(vllm_times),
                                    throughput_tokens_per_sec=metrics[1],
                                    memory_usage_mb=metrics[2],
                                    gpu_utilization_percent=metrics[3],
                                    time_to_first_token_ms=metrics[4],
                                    inter_token_latency_ms=metrics[5],
                                    total_tokens=metrics[6],
                                    timestamp=datetime.now(),
                                )
                            )

                    # Benchmark Transformers
                    hf_times = []
                    for i in range(self.config.num_iterations):
                        metrics = self.benchmark_transformers(prompts, output_length)
                        hf_times.append(metrics[0])

                        if i == self.config.num_iterations - 1:  # Last iteration
                            self.results.append(
                                BenchmarkResult(
                                    framework="Transformers",
                                    model_name=self.config.model_name,
                                    prompt_length=prompt_length,
                                    output_length=output_length,
                                    batch_size=batch_size,
                                    latency_ms=np.mean(hf_times),
                                    throughput_tokens_per_sec=metrics[1],
                                    memory_usage_mb=metrics[2],
                                    gpu_utilization_percent=metrics[3],
                                    time_to_first_token_ms=metrics[4],
                                    inter_token_latency_ms=metrics[5],
                                    total_tokens=metrics[6],
                                    timestamp=datetime.now(),
                                )
                            )

                    # Print immediate comparison
                    vllm_avg = np.mean(vllm_times)
                    hf_avg = np.mean(hf_times)
                    speedup = hf_avg / vllm_avg

                    print(f"  vLLM avg: {vllm_avg:.2f}ms")
                    print(f"  Transformers avg: {hf_avg:.2f}ms")
                    print(f"  Speedup: {speedup:.2f}x")

    def analyze_results(self) -> pd.DataFrame:
        """Analyze benchmark results and create summary"""
        df = pd.DataFrame([r.__dict__ for r in self.results])

        # Calculate speedups
        pivot = df.pivot_table(
            values="latency_ms",
            index=["prompt_length", "output_length", "batch_size"],
            columns="framework",
        )

        if "vLLM" in pivot.columns and "Transformers" in pivot.columns:
            pivot["speedup"] = pivot["Transformers"] / pivot["vLLM"]

        # Summary statistics
        summary = {
            "avg_vllm_latency_ms": df[df["framework"] == "vLLM"]["latency_ms"].mean(),
            "avg_transformers_latency_ms": df[df["framework"] == "Transformers"]["latency_ms"].mean(),
            "avg_vllm_throughput": df[df["framework"] == "vLLM"]["throughput_tokens_per_sec"].mean(),
            "avg_transformers_throughput": df[df["framework"] == "Transformers"]["throughput_tokens_per_sec"].mean(),
            "avg_speedup": pivot["speedup"].mean() if "speedup" in pivot.columns else 0,
            "max_speedup": pivot["speedup"].max() if "speedup" in pivot.columns else 0,
            "min_speedup": pivot["speedup"].min() if "speedup" in pivot.columns else 0,
        }

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        for key, value in summary.items():
            print(f"{key}: {value:.2f}")

        return df

    def generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization plots for benchmark results"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Latency comparison by batch size
        ax1 = axes[0, 0]
        pivot_latency = df.pivot_table(
            values="latency_ms",
            index="batch_size",
            columns="framework",
            aggfunc="mean",
        )
        pivot_latency.plot(kind="bar", ax=ax1)
        ax1.set_title("Average Latency by Batch Size")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_xlabel("Batch Size")
        ax1.legend(title="Framework")

        # 2. Throughput comparison
        ax2 = axes[0, 1]
        pivot_throughput = df.pivot_table(
            values="throughput_tokens_per_sec",
            index="batch_size",
            columns="framework",
            aggfunc="mean",
        )
        pivot_throughput.plot(kind="bar", ax=ax2)
        ax2.set_title("Throughput by Batch Size")
        ax2.set_ylabel("Tokens/sec")
        ax2.set_xlabel("Batch Size")

        # 3. Memory usage comparison
        ax3 = axes[0, 2]
        pivot_memory = df.pivot_table(
            values="memory_usage_mb",
            index="output_length",
            columns="framework",
            aggfunc="mean",
        )
        pivot_memory.plot(kind="line", ax=ax3, marker="o")
        ax3.set_title("Memory Usage by Output Length")
        ax3.set_ylabel("Memory (MB)")
        ax3.set_xlabel("Output Length (tokens)")

        # 4. Speedup heatmap
        ax4 = axes[1, 0]
        speedup_pivot = df.pivot_table(
            values="latency_ms",
            index="prompt_length",
            columns=["framework", "batch_size"],
        )

        # Calculate speedup matrix
        vllm_data = speedup_pivot["vLLM"]
        trans_data = speedup_pivot["Transformers"]
        speedup_matrix = trans_data / vllm_data

        sns.heatmap(speedup_matrix, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax4)
        ax4.set_title("vLLM Speedup Matrix (vs Transformers)")
        ax4.set_ylabel("Prompt Length")
        ax4.set_xlabel("Batch Size")

        # 5. Time to First Token comparison
        ax5 = axes[1, 1]
        pivot_ttft = df.pivot_table(
            values="time_to_first_token_ms",
            index="batch_size",
            columns="framework",
            aggfunc="mean",
        )
        pivot_ttft.plot(kind="bar", ax=ax5)
        ax5.set_title("Time to First Token")
        ax5.set_ylabel("Time (ms)")
        ax5.set_xlabel("Batch Size")

        # 6. GPU Utilization
        ax6 = axes[1, 2]
        pivot_gpu = df.pivot_table(
            values="gpu_utilization_percent",
            index="batch_size",
            columns="framework",
            aggfunc="mean",
        )
        pivot_gpu.plot(kind="bar", ax=ax6)
        ax6.set_title("GPU Utilization")
        ax6.set_ylabel("GPU Usage (%)")
        ax6.set_xlabel("Batch Size")

        plt.suptitle(f"vLLM vs Transformers Benchmark: {self.config.model_name}", fontsize=16)
        plt.tight_layout()

        # Save figure
        output_path = Path("benchmark_results")
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(output_path / f"benchmark_comparison_{timestamp}.png", dpi=300)
        print(f"\nVisualization saved to benchmark_results/benchmark_comparison_{timestamp}.png")

        return fig

    def save_results(self, df: pd.DataFrame):
        """Save benchmark results to files"""
        output_path = Path("benchmark_results")
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as CSV
        csv_path = output_path / f"benchmark_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Save as JSON
        json_path = output_path / f"benchmark_data_{timestamp}.json"
        df.to_json(json_path, orient="records", indent=2)

        # Save summary
        summary_path = output_path / f"benchmark_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write("vLLM vs Transformers Benchmark Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Device: {self.config.device}\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")

            # Write aggregate statistics
            for framework in ["vLLM", "Transformers"]:
                framework_df = df[df["framework"] == framework]
                f.write(f"\n{framework} Statistics:\n")
                f.write(f"  Average Latency: {framework_df['latency_ms'].mean():.2f}ms\n")
                f.write(f"  Average Throughput: {framework_df['throughput_tokens_per_sec'].mean():.2f} tokens/sec\n")
                f.write(f"  Average Memory: {framework_df['memory_usage_mb'].mean():.2f}MB\n")
                f.write(f"  Average GPU Utilization: {framework_df['gpu_utilization_percent'].mean():.2f}%\n")

        print(f"Summary saved to {summary_path}")


def run_benchmarks(
    model_name: str = "microsoft/phi-2",
    quick_mode: bool = False,
):
    """Main entry point for running benchmarks"""
    # Configure benchmark
    if quick_mode:
        # Quick benchmark for testing
        config = BenchmarkConfig(
            model_name=model_name,
            prompt_lengths=[50, 100],
            output_lengths=[50, 100],
            batch_sizes=[1, 4],
            num_iterations=3,
            warmup_iterations=1,
        )
    else:
        # Full benchmark
        config = BenchmarkConfig(
            model_name=model_name,
            num_iterations=10,
        )

    # Run benchmark suite
    suite = BenchmarkSuite(config)

    try:
        # Setup models
        suite.setup()

        # Run comparison
        suite.run_comparison()

        # Analyze results
        df = suite.analyze_results()

        # Generate visualizations
        suite.generate_visualizations(df)

        # Save results
        suite.save_results(df)

        return df

    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise
    finally:
        # Cleanup
        if suite.vllm_model:
            del suite.vllm_model
        if suite.hf_model:
            del suite.hf_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Benchmarking Suite")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with reduced iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    print(f"Running benchmark on {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = run_benchmarks(
        model_name=args.model,
        quick_mode=args.quick,
    )

    print("\n✅ Benchmark complete!")