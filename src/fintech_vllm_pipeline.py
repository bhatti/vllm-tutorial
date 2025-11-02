"""
vLLM-Powered Enterprise Analysis Pipeline
Production-ready implementation for high-throughput financial document processing
Author: Your Blog Series
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid

# For comparison with traditional approach
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of financial documents we process"""
    EARNINGS_REPORT = "earnings_report"
    SEC_FILING = "sec_filing"  # 10-K, 10-Q, 8-K
    NEWS_ARTICLE = "news_article"
    ANALYST_REPORT = "analyst_report"
    REGULATORY_FILING = "regulatory_filing"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class FinancialDocument:
    """Financial document with metadata"""
    id: str
    content: str
    doc_type: DocumentType
    ticker: Optional[str] = None
    timestamp: Optional[datetime] = None
    priority: int = 5  # 1-10, higher is more urgent
    metadata: Dict = None


@dataclass
class AnalysisResult:
    """Result of financial document analysis"""
    document_id: str
    sentiment_score: float  # -1 to 1
    risk_score: float  # 0 to 1
    key_insights: List[str]
    trading_signals: Dict[str, Any]
    compliance_flags: List[str]
    processing_time_ms: float
    tokens_processed: int
    cost_estimate: float


class FinTechVLLMPipeline:
    """
    High-performance financial document analysis pipeline using vLLM
    Designed for production use with cost optimization
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 2048,
        tensor_parallel_size: int = 1,
        use_quantization: bool = False,
        quantization_method: Optional[str] = None  # "awq" or "gptq"
    ):
        """
        Initialize vLLM pipeline for FinTech analysis

        Args:
            model_name: HuggingFace model ID or local path
            gpu_memory_utilization: Fraction of GPU memory to use (0.95 = 95%)
            max_model_len: Maximum sequence length
            tensor_parallel_size: Number of GPUs for tensor parallelism
            use_quantization: Whether to use quantized model
            quantization_method: Quantization method if applicable
        """
        self.model_name = model_name

        # Configure vLLM with production settings
        vllm_kwargs = {
            "model": model_name,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "swap_space": 4,  # GB of CPU swap space for memory overflow
            "enforce_eager": False,  # Use CUDA graphs for better performance
            "trust_remote_code": True,  # Required for some models
        }

        if use_quantization and quantization_method:
            vllm_kwargs["quantization"] = quantization_method
            logger.info(f"Using {quantization_method} quantization")

        # Initialize vLLM engine
        logger.info(f"Initializing vLLM with model: {model_name}")
        self.llm = LLM(**vllm_kwargs)

        # Cost tracking (example rates)
        self.cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens
        self.total_tokens_processed = 0
        self.total_cost = 0.0

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time_ms": 0,
            "avg_latency_ms": 0,
            "throughput_tokens_per_sec": 0
        }

        # Prompt templates for different analysis types
        self.prompt_templates = self._load_prompt_templates()

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load optimized prompt templates for financial analysis"""
        return {
            "earnings_analysis": """Analyze this earnings report and provide:
1. Revenue growth (YoY and QoQ)
2. Profit margins and trends
3. Key business metrics
4. Management guidance
5. Risk factors mentioned

Document: {content}

Analysis:""",

            "risk_assessment": """Assess financial risks in this document:
1. Market risks
2. Credit risks
3. Operational risks
4. Regulatory risks
5. Strategic risks

Document: {content}

Risk Assessment:""",

            "sentiment_extraction": """Extract market sentiment from this financial document.
Provide:
1. Overall sentiment (bullish/bearish/neutral)
2. Confidence level (0-100%)
3. Key positive factors
4. Key negative factors
5. Trading recommendation

Document: {content}

Sentiment Analysis:""",

            "compliance_check": """Check this document for regulatory compliance issues:
1. SEC filing requirements
2. Risk disclosure completeness
3. Financial reporting standards
4. Material information disclosure
5. Forward-looking statement disclaimers

Document: {content}

Compliance Analysis:""",

            "trading_signals": """Extract actionable trading signals from this document:
1. Buy/Sell/Hold recommendation
2. Price target if mentioned
3. Key catalysts
4. Time horizon
5. Risk/Reward assessment

Document: {content}

Trading Signals:"""
        }

    def analyze_document(
        self,
        document: FinancialDocument,
        analysis_types: List[str] = None,
        temperature: float = 0.1,  # Low temperature for factual analysis
        max_tokens: int = 500,
        top_p: float = 0.95
    ) -> AnalysisResult:
        """
        Analyze a single financial document

        Args:
            document: Financial document to analyze
            analysis_types: Types of analysis to perform
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            AnalysisResult with all insights
        """
        if analysis_types is None:
            analysis_types = ["sentiment_extraction", "risk_assessment", "trading_signals"]

        start_time = time.time()

        # Prepare prompts for batch processing
        prompts = []
        for analysis_type in analysis_types:
            if analysis_type in self.prompt_templates:
                prompt = self.prompt_templates[analysis_type].format(
                    content=document.content[:1500]  # Truncate for context window
                )
                prompts.append(prompt)

        # vLLM batch processing - this is where the magic happens!
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["</s>", "\n\n\n"],  # Stop sequences
            skip_special_tokens=True
        )

        # Generate completions for all prompts in parallel
        outputs = self.llm.generate(prompts, sampling_params)

        # Process results
        results = {}
        total_tokens = 0

        for i, output in enumerate(outputs):
            analysis_type = analysis_types[i]
            generated_text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)
            total_tokens += token_count
            results[analysis_type] = generated_text

        # Extract structured insights from generated text
        sentiment_score = self._extract_sentiment_score(
            results.get("sentiment_extraction", "")
        )
        risk_score = self._extract_risk_score(
            results.get("risk_assessment", "")
        )
        key_insights = self._extract_key_insights(results)
        trading_signals = self._extract_trading_signals(
            results.get("trading_signals", "")
        )
        compliance_flags = self._extract_compliance_flags(
            results.get("compliance_check", "")
        )

        # Calculate metrics
        processing_time_ms = (time.time() - start_time) * 1000
        cost_estimate = (total_tokens / 1000) * self.cost_per_1k_tokens

        # Update global metrics
        self._update_metrics(total_tokens, processing_time_ms, cost_estimate)

        return AnalysisResult(
            document_id=document.id,
            sentiment_score=sentiment_score,
            risk_score=risk_score,
            key_insights=key_insights,
            trading_signals=trading_signals,
            compliance_flags=compliance_flags,
            processing_time_ms=processing_time_ms,
            tokens_processed=total_tokens,
            cost_estimate=cost_estimate
        )

    def batch_analyze_documents(
        self,
        documents: List[FinancialDocument],
        batch_size: int = 32,
        priority_queue: bool = True
    ) -> List[AnalysisResult]:
        """
        Analyze multiple documents with optimized batching

        This is where vLLM really shines - continuous batching allows
        processing many documents simultaneously with minimal overhead

        Args:
            documents: List of documents to analyze
            batch_size: Maximum batch size for vLLM
            priority_queue: Process high-priority documents first

        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch analysis of {len(documents)} documents")

        # Sort by priority if requested
        if priority_queue:
            documents = sorted(documents, key=lambda x: x.priority, reverse=True)

        # Prepare all prompts upfront for maximum efficiency
        all_prompts = []
        doc_prompt_mapping = []  # Track which prompts belong to which document

        for doc in documents:
            doc_prompts = []
            for analysis_type in ["sentiment_extraction", "risk_assessment"]:
                prompt = self.prompt_templates[analysis_type].format(
                    content=doc.content[:1500]
                )
                all_prompts.append(prompt)
                doc_prompts.append(len(all_prompts) - 1)
            doc_prompt_mapping.append((doc, doc_prompts))

        # vLLM's continuous batching processes all prompts efficiently
        logger.info(f"Processing {len(all_prompts)} prompts with vLLM continuous batching")

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=300,
            stop=["</s>", "\n\n\n"]
        )

        start_time = time.time()
        outputs = self.llm.generate(all_prompts, sampling_params)
        batch_time = time.time() - start_time

        # Process results
        results = []
        for doc, prompt_indices in doc_prompt_mapping:
            doc_outputs = [outputs[i] for i in prompt_indices]

            # Create analysis result
            sentiment_text = doc_outputs[0].outputs[0].text if len(doc_outputs) > 0 else ""
            risk_text = doc_outputs[1].outputs[0].text if len(doc_outputs) > 1 else ""

            total_tokens = sum(
                len(output.outputs[0].token_ids) for output in doc_outputs
            )

            result = AnalysisResult(
                document_id=doc.id,
                sentiment_score=self._extract_sentiment_score(sentiment_text),
                risk_score=self._extract_risk_score(risk_text),
                key_insights=self._extract_insights_from_text(sentiment_text + risk_text),
                trading_signals={"recommendation": "HOLD"},  # Simplified for demo
                compliance_flags=[],
                processing_time_ms=(batch_time / len(documents)) * 1000,
                tokens_processed=total_tokens,
                cost_estimate=(total_tokens / 1000) * self.cost_per_1k_tokens
            )
            results.append(result)

        # Log performance metrics
        total_tokens = sum(r.tokens_processed for r in results)
        throughput = total_tokens / batch_time
        logger.info(f"Batch processing completed:")
        logger.info(f"  - Documents: {len(documents)}")
        logger.info(f"  - Total time: {batch_time:.2f}s")
        logger.info(f"  - Throughput: {throughput:.1f} tokens/sec")
        logger.info(f"  - Avg latency: {(batch_time/len(documents)*1000):.1f}ms per doc")

        return results

    def streaming_analysis(
        self,
        document_stream: asyncio.Queue,
        result_callback: callable
    ):
        """
        Process documents from a streaming source (e.g., news feed)
        Demonstrates vLLM's ability to handle continuous workloads
        """
        async def process_stream():
            while True:
                try:
                    # Get document from stream
                    document = await asyncio.wait_for(
                        document_stream.get(),
                        timeout=1.0
                    )

                    # Analyze document
                    result = self.analyze_document(document)

                    # Send result to callback
                    await result_callback(result)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Stream processing error: {e}")

        # Run stream processor
        asyncio.create_task(process_stream())

    def _extract_sentiment_score(self, text: str) -> float:
        """Extract sentiment score from generated text"""
        text_lower = text.lower()
        if "bullish" in text_lower or "positive" in text_lower:
            return 0.7
        elif "bearish" in text_lower or "negative" in text_lower:
            return -0.7
        else:
            return 0.0

    def _extract_risk_score(self, text: str) -> float:
        """Extract risk score from generated text"""
        text_lower = text.lower()
        high_risk_keywords = ["high risk", "significant risk", "major concern"]
        medium_risk_keywords = ["moderate risk", "some risk", "potential issue"]

        for keyword in high_risk_keywords:
            if keyword in text_lower:
                return 0.8

        for keyword in medium_risk_keywords:
            if keyword in text_lower:
                return 0.5

        return 0.2

    def _extract_key_insights(self, results: Dict[str, str]) -> List[str]:
        """Extract key insights from all analysis results"""
        insights = []
        for analysis_type, text in results.items():
            # Simple extraction - in production, use more sophisticated NLP
            lines = text.split("\n")
            for line in lines[:3]:  # Top 3 lines
                if len(line.strip()) > 20:
                    insights.append(line.strip())
        return insights[:5]  # Return top 5 insights

    def _extract_insights_from_text(self, text: str) -> List[str]:
        """Extract insights from combined text"""
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
        return lines[:5]

    def _extract_trading_signals(self, text: str) -> Dict[str, Any]:
        """Extract trading signals from generated text"""
        text_lower = text.lower()

        signal = {
            "recommendation": "HOLD",
            "confidence": 0.5,
            "price_target": None,
            "timeframe": "medium-term"
        }

        if "buy" in text_lower or "long" in text_lower:
            signal["recommendation"] = "BUY"
            signal["confidence"] = 0.7
        elif "sell" in text_lower or "short" in text_lower:
            signal["recommendation"] = "SELL"
            signal["confidence"] = 0.7

        return signal

    def _extract_compliance_flags(self, text: str) -> List[str]:
        """Extract compliance issues from generated text"""
        flags = []
        text_lower = text.lower()

        compliance_keywords = {
            "missing disclosure": "DISCLOSURE_MISSING",
            "incomplete filing": "FILING_INCOMPLETE",
            "regulatory concern": "REGULATORY_ISSUE"
        }

        for keyword, flag in compliance_keywords.items():
            if keyword in text_lower:
                flags.append(flag)

        return flags

    def _update_metrics(self, tokens: int, time_ms: float, cost: float):
        """Update performance metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_time_ms"] += time_ms
        self.metrics["avg_latency_ms"] = (
            self.metrics["total_time_ms"] / self.metrics["total_requests"]
        )
        if self.metrics["total_time_ms"] > 0:
            self.metrics["throughput_tokens_per_sec"] = (
                self.metrics["total_tokens"] / (self.metrics["total_time_ms"] / 1000)
            )
        self.total_tokens_processed += tokens
        self.total_cost += cost

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "model": self.model_name,
            "metrics": self.metrics,
            "total_tokens_processed": self.total_tokens_processed,
            "total_cost": round(self.total_cost, 4),
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "avg_cost_per_request": round(
                self.total_cost / max(1, self.metrics["total_requests"]), 6
            )
        }


class TraditionalPipeline:
    """
    Traditional HuggingFace Transformers pipeline for comparison
    This shows what most people use before discovering vLLM
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """Initialize traditional pipeline with HuggingFace Transformers"""
        logger.info(f"Loading model with HuggingFace Transformers: {model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def analyze_document(
        self,
        document: FinancialDocument,
        max_length: int = 500
    ) -> Tuple[str, float]:
        """
        Analyze document with traditional approach
        Note: This processes one document at a time, no batching optimization
        """
        start_time = time.time()

        prompt = f"Analyze this financial document:\n{document.content[:1000]}\n\nAnalysis:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.1,
                do_sample=True,
                top_p=0.95
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time

        return generated_text, processing_time


def benchmark_comparison():
    """
    Direct comparison between vLLM and traditional approach
    This demonstrates the massive performance difference
    """
    # Create sample documents
    sample_documents = [
        FinancialDocument(
            id=f"doc_{i}",
            content=f"""
            Q3 2024 Earnings Report for TechCorp Inc.

            Revenue: $5.2B (up 23% YoY)
            Operating Income: $1.1B (up 31% YoY)
            Net Income: $890M (up 28% YoY)
            EPS: $2.34 (beat estimates by $0.12)

            CEO Commentary: "We saw strong growth across all segments, particularly
            in our cloud services division which grew 45% YoY. We're raising our
            full-year guidance based on continued momentum."

            Risk Factors: Increased competition in cloud services, potential economic
            headwinds, and regulatory scrutiny in key markets.

            Forward Guidance: Q4 revenue expected between $5.5B-$5.7B.
            """,
            doc_type=DocumentType.EARNINGS_REPORT,
            ticker="TECH",
            priority=8
        ) for i in range(10)  # Create 10 identical documents for testing
    ]

    # Test vLLM
    logger.info("=" * 60)
    logger.info("Testing vLLM Pipeline")
    logger.info("=" * 60)

    vllm_pipeline = FinTechVLLMPipeline(
        model_name="microsoft/phi-2",  # Smaller model for testing
        gpu_memory_utilization=0.9
    )

    vllm_start = time.time()
    vllm_results = vllm_pipeline.batch_analyze_documents(sample_documents)
    vllm_time = time.time() - vllm_start

    logger.info(f"vLLM Results:")
    logger.info(f"  - Total time: {vllm_time:.2f}s")
    logger.info(f"  - Docs/second: {len(sample_documents)/vllm_time:.2f}")
    logger.info(f"  - Performance report: {vllm_pipeline.get_performance_report()}")

    # Test traditional approach (only first 3 docs due to speed)
    logger.info("\n" + "=" * 60)
    logger.info("Testing Traditional Pipeline (HuggingFace)")
    logger.info("=" * 60)

    traditional_pipeline = TraditionalPipeline("microsoft/phi-2")

    traditional_start = time.time()
    traditional_results = []
    for doc in sample_documents[:3]:  # Only process 3 due to slowness
        result, proc_time = traditional_pipeline.analyze_document(doc)
        traditional_results.append((result, proc_time))
        logger.info(f"  - Document {doc.id}: {proc_time:.2f}s")
    traditional_time = time.time() - traditional_start

    logger.info(f"\nTraditional Results:")
    logger.info(f"  - Total time for 3 docs: {traditional_time:.2f}s")
    logger.info(f"  - Estimated time for 10 docs: {(traditional_time/3)*10:.2f}s")
    logger.info(f"  - Docs/second: {3/traditional_time:.2f}")

    # Calculate speedup
    vllm_speed = len(sample_documents) / vllm_time
    traditional_speed = 3 / traditional_time
    speedup = vllm_speed / traditional_speed

    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 60)
    logger.info(f"vLLM Throughput: {vllm_speed:.2f} docs/second")
    logger.info(f"Traditional Throughput: {traditional_speed:.2f} docs/second")
    logger.info(f"🚀 vLLM SPEEDUP: {speedup:.1f}x faster!")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run benchmark comparison
    logger.info("Starting FinTech vLLM Pipeline Demonstration")
    logger.info("This will compare vLLM vs traditional approaches")
    logger.info("")

    # Note: This requires GPU with vLLM installed
    # For CPU testing, comment out the benchmark and run individual components

    try:
        benchmark_comparison()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.info("Note: This demo requires a GPU with vLLM installed")
        logger.info("You can test this in Google Colab with a free GPU")
