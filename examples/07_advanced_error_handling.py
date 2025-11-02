#!/usr/bin/env python3
"""
Example 7: Advanced Error Handling and Resilience Patterns
Production-grade error handling for vLLM deployments

Patterns covered:
- Retry with exponential backoff
- Circuit breaker
- Fallback strategies
- Rate limiting
- Timeout handling
"""

import time
import asyncio
from typing import Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta


class ErrorType(Enum):
    """Common error types in LLM serving"""
    TIMEOUT = "timeout"
    OOM = "out_of_memory"
    MODEL_ERROR = "model_error"
    RATE_LIMIT = "rate_limit"
    INVALID_INPUT = "invalid_input"


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


def retry_with_backoff(config: RetryConfig = RetryConfig()):
    """
    Decorator: Retry with exponential backoff

    Example:
        @retry_with_backoff()
        def generate_text(prompt):
            return llm.generate(prompt)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            delay = config.initial_delay

            for attempt in range(config.max_retries):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    if attempt == config.max_retries - 1:
                        raise  # Last attempt, re-raise

                    error_type = classify_error(e)

                    # Don't retry on invalid input
                    if error_type == ErrorType.INVALID_INPUT:
                        raise

                    print(f"⚠️  Attempt {attempt + 1} failed: {error_type.value}")
                    print(f"   Retrying in {delay:.1f}s...")
                    time.sleep(delay)

                    # Exponential backoff
                    delay = min(delay * config.exponential_base, config.max_delay)

            raise RuntimeError(f"Failed after {config.max_retries} retries")

        return wrapper
    return decorator


def classify_error(error: Exception) -> ErrorType:
    """Classify error type for appropriate handling"""
    error_str = str(error).lower()

    if "timeout" in error_str:
        return ErrorType.TIMEOUT
    elif "out of memory" in error_str or "oom" in error_str:
        return ErrorType.OOM
    elif "rate limit" in error_str:
        return ErrorType.RATE_LIMIT
    elif "invalid" in error_str or "validation" in error_str:
        return ErrorType.INVALID_INPUT
    else:
        return ErrorType.MODEL_ERROR


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance

    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, reject requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""

        if self.state == "OPEN":
            # Check if timeout elapsed
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
                print("🔄 Circuit breaker: HALF_OPEN (testing)")
            else:
                raise RuntimeError("Circuit breaker OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)

            # Success - reset on HALF_OPEN or CLOSED
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                print("✅ Circuit breaker: CLOSED (recovered)")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                print(f"🔴 Circuit breaker: OPEN (threshold {self.failure_threshold} reached)")

            raise


class RateLimiter:
    """
    Token bucket rate limiter

    Limits requests per second to prevent overload
    """

    def __init__(self, max_requests: int, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = time.time()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens, return True if allowed"""

        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens based on elapsed time
        self.tokens = min(
            self.max_requests,
            self.tokens + (elapsed / self.time_window) * self.max_requests
        )
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            return False

    def wait_for_token(self, tokens: int = 1):
        """Wait until token is available"""
        while not self.acquire(tokens):
            time.sleep(0.1)


def fallback_strategy_example():
    """
    Example: Fallback strategies when primary fails

    Strategies:
    1. Fallback to cached response
    2. Fallback to simpler model
    3. Fallback to template response
    """
    print(f"\n{'='*80}")
    print(f"Fallback Strategies")
    print(f"{'='*80}\n")

    cache = {
        "What is risk management?": "Risk management involves identifying, assessing, and controlling threats..."
    }

    def generate_with_fallback(prompt: str) -> str:
        """Try primary, fallback to cache, then template"""

        # Try primary model
        try:
            # Simulate LLM call
            if "error" in prompt.lower():
                raise RuntimeError("Model error")

            return f"[Primary Model] Response to: {prompt}"

        except Exception as e:
            print(f"⚠️  Primary model failed: {e}")

            # Fallback 1: Check cache
            if prompt in cache:
                print(f"✅ Returning cached response")
                return f"[Cached] {cache[prompt]}"

            # Fallback 2: Template response
            print(f"✅ Returning template response")
            return f"I apologize, but I'm unable to process your request right now. Please try again later."

    # Test scenarios
    print("Test 1: Normal request")
    print(generate_with_fallback("What is portfolio optimization?"))

    print(f"\nTest 2: Cached request")
    print(generate_with_fallback("What is risk management?"))

    print(f"\nTest 3: Error with fallback")
    print(generate_with_fallback("This will error"))


def timeout_handling_example():
    """Example: Timeout handling for long-running requests"""
    print(f"\n{'='*80}")
    print(f"Timeout Handling")
    print(f"{'='*80}\n")

    import signal

    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("Request timed out")

    def generate_with_timeout(prompt: str, timeout_seconds: int = 30):
        """Generate with timeout"""

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            # Simulate LLM call
            time.sleep(0.5)  # Simulated processing
            result = f"Response to: {prompt}"

            # Cancel timeout
            signal.alarm(0)
            return result

        except TimeoutError:
            print(f"❌ Request timed out after {timeout_seconds}s")
            # Return partial result or error
            return "Request timed out. Please try a shorter prompt."

    print("Test: Normal request")
    print(generate_with_timeout("Quick question", timeout_seconds=5))


def production_example():
    """Complete production error handling example"""
    print(f"\n{'='*80}")
    print(f"Production Error Handling Pattern")
    print(f"{'='*80}\n")

    code_example = '''
from fastapi import FastAPI, HTTPException
from circuitbreaker import CircuitBreaker, CircuitBreakerError

app = FastAPI()

# Initialize components
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
rate_limiter = RateLimiter(max_requests=100, time_window=1.0)

@app.post("/generate")
@retry_with_backoff(max_retries=3)
async def generate(request: GenerateRequest):
    """Generate with full error handling"""

    # Rate limiting
    if not rate_limiter.acquire():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        # Circuit breaker protection
        result = circuit_breaker.call(
            llm.generate,
            request.prompt,
            max_tokens=request.max_tokens
        )

        return {"text": result, "status": "success"}

    except CircuitBreakerError:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

    except Exception as e:
        # Log error
        logger.error(f"Generation failed: {e}")

        # Return fallback response
        return {
            "text": "I apologize, but I'm unable to process your request.",
            "status": "error",
            "fallback": True
        }
'''

    print(code_example)


def main():
    """Run error handling examples"""
    print(f"{'='*80}")
    print(f"Advanced Error Handling & Resilience Patterns")
    print(f"{'='*80}\n")

    fallback_strategy_example()
    timeout_handling_example()
    production_example()

    print(f"\n{'='*80}")
    print(f"✅ Summary")
    print(f"{'='*80}")
    print(f"\n1. Retry with exponential backoff - Handle transient failures")
    print(f"2. Circuit breaker - Prevent cascading failures")
    print(f"3. Rate limiting - Prevent overload")
    print(f"4. Fallback strategies - Graceful degradation")
    print(f"5. Timeout handling - Prevent hanging requests")
    print(f"\n💡 Production Recommendation:")
    print(f"   - Use all patterns together for maximum resilience")
    print(f"   - Log all errors for debugging")
    print(f"   - Monitor error rates and circuit breaker state")
    print(f"   - Have fallback responses for critical paths")


if __name__ == "__main__":
    main()
