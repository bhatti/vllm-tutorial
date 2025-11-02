"""
Unit tests for FinTechVLLMPipeline with >90% coverage requirement
Following TDD principles: Write test first, then implementation
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List
import asyncio
from datetime import datetime
import numpy as np

# Import modules to test
from src.fintech_vllm_pipeline import (
    FinTechVLLMPipeline,
    FinancialDocument,
    AnalysisResult,
    DocumentPriority,
    SentimentScore,
    RiskLevel,
    BatchProcessor,
    CacheManager,
    MetricsCollector,
)


class TestFinancialDocument:
    """Test suite for FinancialDocument data model"""

    def test_create_financial_document(self):
        """Test creating a financial document"""
        # Arrange
        doc_data = {
            "id": "doc_001",
            "content": "Q3 2024 Earnings: Revenue $5.2B (+23% YoY)",
            "doc_type": "earnings_report",
            "ticker": "TECH",
            "timestamp": datetime.now(),
            "priority": 8,
        }

        # Act
        doc = FinancialDocument(**doc_data)

        # Assert
        assert doc.id == "doc_001"
        assert doc.ticker == "TECH"
        assert doc.priority == 8
        assert doc.doc_type == "earnings_report"

    def test_document_validation(self):
        """Test document validation rules"""
        # Test invalid priority
        with pytest.raises(ValueError):
            FinancialDocument(
                id="test",
                content="test",
                doc_type="report",
                priority=11,  # Invalid: > 10
            )

        # Test empty content
        with pytest.raises(ValueError):
            FinancialDocument(
                id="test",
                content="",  # Invalid: empty
                doc_type="report",
            )

    def test_document_serialization(self):
        """Test document serialization to dict"""
        # Arrange
        doc = FinancialDocument(
            id="doc_001",
            content="Test content",
            doc_type="report",
            ticker="TEST",
        )

        # Act
        doc_dict = doc.to_dict()

        # Assert
        assert isinstance(doc_dict, dict)
        assert doc_dict["id"] == "doc_001"
        assert doc_dict["ticker"] == "TEST"
        assert "timestamp" in doc_dict


class TestAnalysisResult:
    """Test suite for AnalysisResult data model"""

    def test_create_analysis_result(self):
        """Test creating an analysis result"""
        # Arrange
        result_data = {
            "document_id": "doc_001",
            "summary": "Strong Q3 performance with revenue beat",
            "sentiment": SentimentScore(
                score=0.85,
                confidence=0.92,
                label="positive",
            ),
            "risk_level": RiskLevel.LOW,
            "key_metrics": {
                "revenue": 5.2e9,
                "growth": 0.23,
                "eps": 2.34,
            },
            "recommendations": ["BUY", "HOLD"],
            "processing_time_ms": 150,
            "model_used": "llama-13b",
        }

        # Act
        result = AnalysisResult(**result_data)

        # Assert
        assert result.document_id == "doc_001"
        assert result.sentiment.score == 0.85
        assert result.risk_level == RiskLevel.LOW
        assert "revenue" in result.key_metrics
        assert result.processing_time_ms == 150

    def test_sentiment_score_validation(self):
        """Test sentiment score validation"""
        # Test valid sentiment
        sentiment = SentimentScore(score=0.7, confidence=0.9, label="positive")
        assert 0 <= sentiment.score <= 1
        assert 0 <= sentiment.confidence <= 1

        # Test invalid score
        with pytest.raises(ValueError):
            SentimentScore(score=1.5, confidence=0.9, label="positive")

    def test_risk_level_classification(self):
        """Test risk level classification logic"""
        # Arrange test cases
        test_cases = [
            (0.2, RiskLevel.LOW),
            (0.4, RiskLevel.MODERATE),
            (0.7, RiskLevel.HIGH),
            (0.95, RiskLevel.CRITICAL),
        ]

        for risk_score, expected_level in test_cases:
            # Act
            level = RiskLevel.from_score(risk_score)

            # Assert
            assert level == expected_level


class TestFinTechVLLMPipeline:
    """Test suite for main FinTech vLLM pipeline"""

    @pytest.fixture
    def mock_vllm(self):
        """Create mock vLLM instance"""
        mock = MagicMock()
        mock.generate.return_value = [
            MagicMock(
                outputs=[
                    MagicMock(
                        text="Analysis: Strong performance with 23% YoY growth",
                        token_ids=[1, 2, 3, 4, 5],
                    )
                ]
            )
        ]
        return mock

    @pytest.fixture
    def pipeline(self, mock_vllm, mock_langfuse):
        """Create pipeline instance with mocked dependencies"""
        with patch("src.fintech_vllm_pipeline.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_vllm
            pipeline = FinTechVLLMPipeline(
                model_name="mistral-7b",
                langfuse_config={"public_key": "test", "secret_key": "test"},
            )
            pipeline.langfuse = mock_langfuse
            return pipeline

    def test_analyze_single_document(self, pipeline):
        """Test analyzing a single financial document"""
        # Arrange
        document = FinancialDocument(
            id="doc_001",
            content="""Q3 2024 Earnings Report
            Revenue: $5.2B (+23% YoY)
            Net Income: $890M (+28% YoY)
            EPS: $2.34 (beat by $0.12)""",
            doc_type="earnings_report",
            ticker="TECH",
        )

        # Act
        result = pipeline.analyze_document(document)

        # Assert
        assert isinstance(result, AnalysisResult)
        assert result.document_id == "doc_001"
        assert result.summary is not None
        assert result.sentiment is not None
        assert result.risk_level in RiskLevel
        assert result.model_used == "mistral-7b"
        assert result.processing_time_ms > 0

    def test_batch_document_processing(self, pipeline):
        """Test batch processing of multiple documents"""
        # Arrange
        documents = [
            FinancialDocument(
                id=f"doc_{i:03d}",
                content=f"Financial report {i} with metrics",
                doc_type="report",
                ticker=f"TICK{i}",
                priority=i,
            )
            for i in range(5)
        ]

        # Act
        results = pipeline.batch_analyze_documents(
            documents, batch_size=2, max_workers=2
        )

        # Assert
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.document_id == f"doc_{i:03d}"
            assert result.summary is not None

    def test_priority_based_processing(self, pipeline):
        """Test that high priority documents are processed first"""
        # Arrange
        documents = [
            FinancialDocument(
                id="low_priority",
                content="Low priority content",
                doc_type="report",
                priority=2,
            ),
            FinancialDocument(
                id="high_priority",
                content="High priority content",
                doc_type="alert",
                priority=9,
            ),
            FinancialDocument(
                id="medium_priority",
                content="Medium priority content",
                doc_type="update",
                priority=5,
            ),
        ]

        # Act
        with patch.object(pipeline, "_track_processing_order") as mock_track:
            results = pipeline.batch_analyze_documents(documents)

            # Assert - High priority processed first
            processing_order = [call[0][0] for call in mock_track.call_args_list]
            assert processing_order[0].id == "high_priority"
            assert processing_order[-1].id == "low_priority"

    def test_sentiment_analysis_accuracy(self, pipeline):
        """Test sentiment analysis for different content types"""
        # Arrange test cases
        test_cases = [
            ("Revenue exceeded expectations by 15%", "positive", 0.7),
            ("Significant losses reported in Q3", "negative", 0.7),
            ("Performance remained stable", "neutral", 0.5),
        ]

        for content, expected_label, min_confidence in test_cases:
            # Arrange
            doc = FinancialDocument(
                id="test", content=content, doc_type="report"
            )

            # Act
            result = pipeline.analyze_document(doc)

            # Assert
            assert result.sentiment.label == expected_label
            assert result.sentiment.confidence >= min_confidence

    def test_risk_assessment(self, pipeline):
        """Test risk level assessment for various scenarios"""
        # Arrange test cases with content and expected risk
        risk_scenarios = [
            ("Strong cash flow and minimal debt", RiskLevel.LOW),
            ("Some market volatility expected", RiskLevel.MODERATE),
            ("Significant regulatory challenges", RiskLevel.HIGH),
            ("Bankruptcy proceedings initiated", RiskLevel.CRITICAL),
        ]

        for content, expected_risk in risk_scenarios:
            # Arrange
            doc = FinancialDocument(
                id="test", content=content, doc_type="risk_assessment"
            )

            # Act
            result = pipeline.analyze_document(doc)

            # Assert
            assert result.risk_level == expected_risk

    def test_key_metrics_extraction(self, pipeline):
        """Test extraction of key financial metrics"""
        # Arrange
        doc = FinancialDocument(
            id="test",
            content="""
            Revenue: $5.2B
            Net Income: $890M
            EPS: $2.34
            Debt-to-Equity: 0.45
            P/E Ratio: 18.5
            """,
            doc_type="earnings_report",
        )

        # Act
        result = pipeline.analyze_document(doc)

        # Assert
        assert "revenue" in result.key_metrics
        assert result.key_metrics["revenue"] == 5.2e9
        assert "eps" in result.key_metrics
        assert result.key_metrics["eps"] == 2.34
        assert "debt_to_equity" in result.key_metrics

    def test_error_handling(self, pipeline):
        """Test error handling for invalid inputs"""
        # Test with None document
        with pytest.raises(ValueError):
            pipeline.analyze_document(None)

        # Test with malformed document
        with pytest.raises(ValueError):
            doc = FinancialDocument(id="test", content="", doc_type="report")
            pipeline.analyze_document(doc)

    def test_caching_mechanism(self, pipeline):
        """Test that caching prevents duplicate processing"""
        # Arrange
        doc = FinancialDocument(
            id="cached_doc", content="Test content", doc_type="report"
        )

        # Act - Process document twice
        result1 = pipeline.analyze_document(doc)
        result2 = pipeline.analyze_document(doc)

        # Assert - Second call should use cache
        assert result1.document_id == result2.document_id
        assert result1.summary == result2.summary
        # Processing time should be much lower for cached result
        assert result2.processing_time_ms < result1.processing_time_ms


class TestBatchProcessor:
    """Test suite for batch processing functionality"""

    @pytest.fixture
    def batch_processor(self):
        """Create batch processor instance"""
        return BatchProcessor(batch_size=3, max_workers=2)

    def test_batch_creation(self, batch_processor):
        """Test creating batches from documents"""
        # Arrange
        documents = [f"doc_{i}" for i in range(10)]

        # Act
        batches = batch_processor.create_batches(documents)

        # Assert
        assert len(batches) == 4  # 10 docs / 3 batch_size = 4 batches
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1  # Last batch has remainder

    def test_parallel_batch_execution(self, batch_processor):
        """Test parallel execution of batches"""
        # Arrange
        documents = [f"doc_{i}" for i in range(6)]

        def process_func(doc):
            return f"processed_{doc}"

        # Act
        results = batch_processor.process_parallel(documents, process_func)

        # Assert
        assert len(results) == 6
        for i, result in enumerate(results):
            assert result == f"processed_doc_{i}"

    def test_batch_error_recovery(self, batch_processor):
        """Test error recovery in batch processing"""
        # Arrange
        documents = ["doc_1", "doc_error", "doc_3"]

        def process_func(doc):
            if "error" in doc:
                raise Exception("Processing failed")
            return f"processed_{doc}"

        # Act
        results = batch_processor.process_with_recovery(documents, process_func)

        # Assert
        assert len(results) == 3
        assert results[0] == "processed_doc_1"
        assert results[1] is None  # Failed document
        assert results[2] == "processed_doc_3"


class TestCacheManager:
    """Test suite for caching functionality"""

    @pytest.fixture
    def cache_manager(self, mock_redis):
        """Create cache manager with mocked Redis"""
        return CacheManager(redis_client=mock_redis, ttl_seconds=3600)

    def test_cache_set_and_get(self, cache_manager):
        """Test setting and retrieving from cache"""
        # Arrange
        key = "test_key"
        value = {"result": "test_data"}

        # Act
        cache_manager.set(key, value)
        retrieved = cache_manager.get(key)

        # Assert
        assert retrieved == value

    def test_cache_expiration(self, cache_manager):
        """Test cache expiration behavior"""
        # Arrange
        key = "expiring_key"
        value = {"data": "temporary"}

        # Act
        cache_manager.set(key, value, ttl=1)  # 1 second TTL
        import time

        time.sleep(1.5)
        retrieved = cache_manager.get(key)

        # Assert
        assert retrieved is None  # Should be expired

    def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation"""
        # Arrange
        key = "invalid_key"
        value = {"data": "to_invalidate"}

        # Act
        cache_manager.set(key, value)
        cache_manager.invalidate(key)
        retrieved = cache_manager.get(key)

        # Assert
        assert retrieved is None


class TestMetricsCollector:
    """Test suite for metrics collection"""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance"""
        return MetricsCollector()

    def test_latency_tracking(self, metrics_collector):
        """Test latency metrics tracking"""
        # Arrange & Act
        metrics_collector.record_latency("analyze_document", 150)
        metrics_collector.record_latency("analyze_document", 200)
        metrics_collector.record_latency("analyze_document", 175)

        # Assert
        stats = metrics_collector.get_latency_stats("analyze_document")
        assert stats["count"] == 3
        assert stats["mean"] == 175
        assert stats["p50"] == 175
        assert stats["p95"] <= 200

    def test_throughput_calculation(self, metrics_collector):
        """Test throughput metrics calculation"""
        # Arrange & Act
        for _ in range(100):
            metrics_collector.increment_counter("requests_processed")

        import time

        time.sleep(1)
        throughput = metrics_collector.get_throughput("requests_processed")

        # Assert
        assert throughput > 0
        assert throughput <= 100  # Should be ~100 per second

    def test_error_rate_tracking(self, metrics_collector):
        """Test error rate calculation"""
        # Arrange & Act
        for i in range(100):
            metrics_collector.increment_counter("total_requests")
            if i % 10 == 0:  # 10% errors
                metrics_collector.increment_counter("failed_requests")

        # Assert
        error_rate = metrics_collector.get_error_rate()
        assert error_rate == 0.11  # 11/100 = 11%

    def test_metrics_reset(self, metrics_collector):
        """Test metrics reset functionality"""
        # Arrange
        metrics_collector.record_latency("test_operation", 100)
        metrics_collector.increment_counter("test_counter")

        # Act
        metrics_collector.reset()

        # Assert
        stats = metrics_collector.get_latency_stats("test_operation")
        assert stats["count"] == 0
        assert metrics_collector.get_counter("test_counter") == 0


# Integration tests
@pytest.mark.integration
class TestFinTechPipelineIntegration:
    """Integration tests for the complete pipeline"""

    @pytest.fixture
    def integrated_pipeline(self):
        """Create fully integrated pipeline"""
        with patch("src.fintech_vllm_pipeline.LLM") as mock_llm:
            # Mock vLLM responses
            mock_llm.return_value.generate.return_value = [
                MagicMock(
                    outputs=[
                        MagicMock(
                            text="Comprehensive analysis of financial data",
                            token_ids=list(range(50)),
                        )
                    ]
                )
            ]
            return FinTechVLLMPipeline(model_name="llama-13b")

    @pytest.mark.asyncio
    async def test_end_to_end_document_flow(self, integrated_pipeline):
        """Test complete document processing flow"""
        # Arrange
        documents = [
            FinancialDocument(
                id="earnings_001",
                content="Q4 2024: Record revenue of $10B",
                doc_type="earnings_report",
                ticker="MEGA",
                priority=9,
            ),
            FinancialDocument(
                id="risk_001",
                content="Market volatility increasing",
                doc_type="risk_assessment",
                ticker="MEGA",
                priority=7,
            ),
        ]

        # Act
        results = integrated_pipeline.batch_analyze_documents(documents)

        # Assert
        assert len(results) == 2

        # Check earnings report analysis
        earnings_result = results[0]
        assert earnings_result.document_id == "earnings_001"
        assert earnings_result.sentiment.label in ["positive", "neutral"]
        assert earnings_result.risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]

        # Check risk assessment analysis
        risk_result = results[1]
        assert risk_result.document_id == "risk_001"
        assert risk_result.risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH]

    def test_performance_benchmark(self, integrated_pipeline, benchmark):
        """Benchmark pipeline performance"""
        # Arrange
        doc = FinancialDocument(
            id="perf_test",
            content="Performance test document with financial data",
            doc_type="report",
        )

        # Act
        result = benchmark(integrated_pipeline.analyze_document, doc)

        # Assert
        assert result is not None
        assert benchmark.stats["mean"] < 1.0  # Should process in <1 second
        assert benchmark.stats["stddev"] < 0.1  # Low variance


# Performance and stress tests
@pytest.mark.slow
class TestPerformanceAndStress:
    """Performance and stress testing"""

    def test_large_batch_processing(self):
        """Test processing large batches of documents"""
        # Arrange
        documents = [
            FinancialDocument(
                id=f"stress_{i:05d}",
                content=f"Document {i} with extensive financial data " * 100,
                doc_type="report",
                ticker=f"TICK{i % 100}",
                priority=i % 10,
            )
            for i in range(1000)
        ]

        with patch("src.fintech_vllm_pipeline.LLM"):
            pipeline = FinTechVLLMPipeline(model_name="llama-13b")

            # Act
            import time

            start = time.time()
            results = pipeline.batch_analyze_documents(
                documents, batch_size=50, max_workers=4
            )
            elapsed = time.time() - start

            # Assert
            assert len(results) == 1000
            assert elapsed < 60  # Should process 1000 docs in <60 seconds

    def test_memory_efficiency(self):
        """Test memory usage remains bounded"""
        import psutil
        import gc

        # Get initial memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process many documents
        with patch("src.fintech_vllm_pipeline.LLM"):
            pipeline = FinTechVLLMPipeline(model_name="llama-13b")

            for batch in range(10):
                documents = [
                    FinancialDocument(
                        id=f"mem_{batch}_{i}",
                        content="x" * 10000,  # Large content
                        doc_type="report",
                    )
                    for i in range(100)
                ]
                pipeline.batch_analyze_documents(documents)
                gc.collect()

        # Check memory didn't grow excessively
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        assert memory_growth < 500  # Should not grow more than 500MB