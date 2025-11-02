"""
Pytest configuration and shared fixtures for vLLM project tests
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure pytest settings
pytest_plugins = ["pytest_asyncio", "pytest_benchmark", "pytest_mock"]


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_prompts() -> List[str]:
    """Sample prompts for testing"""
    return [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "Analyze the financial implications of rising interest rates.",
        "Create a risk assessment for a portfolio.",
    ]


@pytest.fixture(scope="session")
def sample_financial_documents() -> List[Dict[str, Any]]:
    """Sample financial documents for testing"""
    return [
        {
            "id": "doc_001",
            "content": """Q3 2024 Earnings Report
            Revenue: $5.2B (+23% YoY)
            Net Income: $890M (+28% YoY)
            EPS: $2.34 (beat by $0.12)
            Guidance: Q4 revenue $5.5B-$5.7B""",
            "doc_type": "earnings_report",
            "ticker": "TECH",
            "priority": 8,
        },
        {
            "id": "doc_002",
            "content": """Risk Assessment Report
            Market Risk: HIGH - Volatility index above 30
            Credit Risk: MODERATE - Some exposure to subprime
            Operational Risk: LOW - Strong controls in place""",
            "doc_type": "risk_assessment",
            "ticker": "FINC",
            "priority": 9,
        },
    ]


@pytest.fixture
def mock_llm(mocker: MockerFixture) -> MagicMock:
    """Mock vLLM instance for testing"""
    mock = MagicMock()
    mock.generate.return_value = [
        MagicMock(
            outputs=[
                MagicMock(
                    text="Test response",
                    token_ids=[1, 2, 3, 4, 5],
                )
            ]
        )
    ]
    return mock


@pytest.fixture
def mock_gpu_available(mocker: MockerFixture) -> None:
    """Mock GPU availability"""
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.get_device_name", return_value="NVIDIA L4")
    mocker.patch(
        "torch.cuda.get_device_properties",
        return_value=MagicMock(total_memory=24 * 1024**3),
    )


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Mock model configuration"""
    return {
        "name": "test-model",
        "tier": "small",
        "cost_per_1k_tokens": 0.001,
        "avg_latency_ms": 100,
        "max_context_length": 2048,
        "capabilities": ["general", "analysis"],
        "is_vllm": True,
    }


@pytest.fixture
def mock_langfuse(mocker: MockerFixture) -> MagicMock:
    """Mock Langfuse client"""
    mock = MagicMock()
    mock.trace.return_value = MagicMock()
    mock.generation.return_value = MagicMock()
    mock.score.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_redis(mocker: MockerFixture) -> MagicMock:
    """Mock Redis client"""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.expire.return_value = True
    return mock


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch) -> None:
    """Reset environment variables for each test"""
    # Clear any existing env vars that might affect tests
    env_vars_to_clear = [
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "REDIS_URL",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    # Set test environment
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create temporary directory for model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Create mock model files
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "pytorch_model.bin").write_bytes(b"mock model data")

    return model_dir


@pytest.fixture
def benchmark_results() -> Dict[str, Any]:
    """Mock benchmark results for testing"""
    return {
        "vllm": {
            "throughput_tokens_per_sec": 1200,
            "latency_p95_ms": 85,
            "memory_usage_gb": 15.2,
        },
        "transformers": {
            "throughput_tokens_per_sec": 50,
            "latency_p95_ms": 2000,
            "memory_usage_gb": 18.5,
        },
        "speedup": 24,
    }


@pytest.fixture(scope="function")
def mock_api_client(mocker: MockerFixture) -> MagicMock:
    """Mock API client for testing endpoints"""
    from httpx import Response

    mock = MagicMock()
    mock.post.return_value = Response(
        status_code=200,
        json={"text": "Generated text", "tokens": 50, "latency_ms": 100},
    )
    mock.get.return_value = Response(
        status_code=200,
        json={"status": "healthy", "models": ["phi-2", "mistral-7b"]},
    )
    return mock


# Async fixtures
@pytest.fixture
async def async_mock_llm() -> MagicMock:
    """Async mock vLLM for async tests"""
    mock = MagicMock()

    async def async_generate(*args, **kwargs):
        return [
            MagicMock(
                outputs=[
                    MagicMock(
                        text="Async test response",
                        token_ids=[1, 2, 3, 4, 5],
                    )
                ]
            )
        ]

    mock.generate = async_generate
    return mock


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")


# Test data generators
@pytest.fixture
def generate_routing_request():
    """Factory for generating routing requests"""

    def _generate(
        complexity: str = "simple",
        user_tier: str = "basic",
        content: str = None,
    ) -> Dict[str, Any]:
        contents = {
            "simple": "What is the capital of France?",
            "moderate": "Analyze this earnings report and provide insights.",
            "complex": "Design a distributed system for high-frequency trading.",
            "critical": "Provide medical diagnosis for these symptoms.",
        }

        return {
            "id": f"test_{complexity}",
            "content": content or contents.get(complexity, "Test content"),
            "user_id": f"user_{user_tier}",
            "metadata": {"user_tier": user_tier},
            "max_latency_ms": 1000,
            "max_cost": 0.10,
        }

    return _generate


# Performance monitoring for tests
@pytest.fixture
def performance_monitor(request):
    """Monitor test performance"""
    import time
    import psutil

    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    yield

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    duration = end_time - start_time
    memory_used = end_memory - start_memory

    # Add performance info to test report
    if hasattr(request.node, "performance"):
        request.node.performance = {
            "duration_seconds": duration,
            "memory_used_mb": memory_used,
        }

    # Fail if test is too slow (configurable)
    max_duration = request.node.get_closest_marker("max_duration")
    if max_duration and duration > max_duration.args[0]:
        pytest.fail(f"Test too slow: {duration:.2f}s > {max_duration.args[0]}s")


# Coverage utilities
@pytest.fixture(scope="session")
def coverage_threshold() -> float:
    """Minimum coverage threshold"""
    return 90.0


def pytest_sessionfinish(session, exitstatus):
    """Check coverage after all tests"""
    # Only check coverage if explicitly running with --cov flag
    if exitstatus == 0 and hasattr(session.config.option, 'cov_source'):
        try:
            import coverage
            cov = coverage.Coverage()
            cov.load()
            total_coverage = cov.report()

            if total_coverage < 90:
                print(f"\n⚠️  Coverage {total_coverage:.1f}% is below 90% threshold!")
                session.exitstatus = 1
        except coverage.exceptions.NoDataError:
            # No coverage data available, which is fine if not running with --cov
            pass