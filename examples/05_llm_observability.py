#!/usr/bin/env python3
"""
Example 5: Advanced LLM Observability with Langfuse and Phoenix
Demonstrates production-grade LLM tracing and analytics

Tools covered:
- Langfuse: Full LLM observability platform
- Arize Phoenix: Open-source LLM monitoring
- Custom metrics integration
"""

import time
import os
from typing import Dict, List, Optional
from datetime import datetime

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("⚠️  Langfuse not available. Install: pip install langfuse")

try:
    import phoenix as px
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("⚠️  Phoenix not available. Install: pip install arize-phoenix")


def langfuse_example():
    """
    Example: Track LLM requests with Langfuse

    Langfuse provides:
    - Request tracing
    - Cost tracking
    - User analytics
    - Prompt versioning
    - A/B testing support
    """
    if not LANGFUSE_AVAILABLE:
        print("\n⚠️  Langfuse not installed")
        return

    print(f"\n{'='*80}")
    print(f"Example 1: Langfuse LLM Observability")
    print(f"{'='*80}\n")

    # Initialize Langfuse
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-demo"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-demo"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
    )

    print("✅ Langfuse initialized")
    print(f"   Host: {os.getenv('LANGFUSE_HOST', 'http://localhost:3001')}\n")

    # Example: Track a generation
    print("📝 Tracking LLM generation...\n")

    # Create a trace
    trace = langfuse.trace(
        name="financial_analysis",
        user_id="user_123",
        metadata={
            "environment": "production",
            "model": "phi-2",
            "use_case": "fintech",
        }
    )

    # Create a generation within the trace
    generation = trace.generation(
        name="analyze_earnings",
        model="microsoft/phi-2",
        model_parameters={
            "temperature": 0.7,
            "max_tokens": 500,
        },
        input="Analyze the quarterly earnings report for Tesla",
        metadata={
            "document_id": "earnings_q4_2024",
            "department": "risk_analysis",
        }
    )

    # Simulate LLM call
    time.sleep(0.5)
    output = "Tesla's Q4 2024 earnings show strong revenue growth..."

    # End the generation with output and metrics
    generation.end(
        output=output,
        usage={
            "input_tokens": 15,
            "output_tokens": 150,
            "total_tokens": 165,
        },
        metadata={
            "latency_ms": 500,
            "cost_usd": 0.000033,
        }
    )

    print("✅ Generation tracked in Langfuse")
    print(f"   Trace ID: {trace.id}")
    print(f"   Input tokens: 15")
    print(f"   Output tokens: 150")
    print(f"   Cost: $0.000033\n")

    # Example: Track multiple steps (RAG pipeline)
    print("📝 Tracking RAG pipeline with multiple steps...\n")

    trace2 = langfuse.trace(
        name="rag_pipeline",
        user_id="user_456",
    )

    # Step 1: Document retrieval
    retrieval = trace2.span(
        name="vector_search",
        input="Risk factors for portfolio",
    )
    retrieval.end(
        output="Found 5 relevant documents",
        metadata={"num_docs": 5, "latency_ms": 50}
    )

    # Step 2: LLM generation
    generation2 = trace2.generation(
        name="synthesize_answer",
        model="microsoft/phi-2",
        input="Synthesize risk factors from 5 documents",
    )
    generation2.end(
        output="Key risk factors include...",
        usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
    )

    print("✅ RAG pipeline tracked")
    print(f"   Steps: vector_search → synthesize_answer")
    print(f"   Total cost: $0.000140\n")

    # Example: Score the output quality
    print("📝 Scoring output quality...\n")

    langfuse.score(
        trace_id=trace2.id,
        name="answer_quality",
        value=0.9,  # 0-1 scale
        comment="High quality, factually accurate",
    )

    langfuse.score(
        trace_id=trace2.id,
        name="user_feedback",
        value=1.0,  # User thumbs up
    )

    print("✅ Quality scores added")
    print(f"   Answer quality: 0.9/1.0")
    print(f"   User feedback: 👍\n")

    # Flush to ensure data is sent
    langfuse.flush()

    print(f"{'='*80}")
    print(f"🎯 Langfuse Features Demonstrated:")
    print(f"{'='*80}")
    print(f"✅ Request tracing with metadata")
    print(f"✅ Cost tracking per request")
    print(f"✅ Multi-step pipeline tracking (RAG)")
    print(f"✅ Quality scoring and user feedback")
    print(f"\n💡 View dashboard: http://localhost:3001")


def phoenix_example():
    """
    Example: Track LLM requests with Arize Phoenix

    Phoenix provides:
    - Open-source LLM monitoring
    - Trace visualization
    - Embedding analysis
    - Drift detection
    """
    if not PHOENIX_AVAILABLE:
        print("\n⚠️  Phoenix not installed")
        return

    print(f"\n{'='*80}")
    print(f"Example 2: Arize Phoenix LLM Monitoring")
    print(f"{'='*80}\n")

    # Launch Phoenix (will start local server)
    print("🚀 Launching Phoenix server...")
    session = px.launch_app()
    print(f"✅ Phoenix running at: {session.url}\n")

    print(f"{'='*80}")
    print(f"🎯 Phoenix Features:")
    print(f"{'='*80}")
    print(f"✅ Open-source (no external dependencies)")
    print(f"✅ Real-time trace visualization")
    print(f"✅ Embedding drift detection")
    print(f"✅ Model performance tracking")
    print(f"\n💡 Integration with OpenTelemetry for auto-tracing")
    print(f"💡 View dashboard: {session.url}")


def custom_observability_example():
    """
    Example: Custom observability integration

    Shows how to integrate vLLM with custom monitoring
    """
    print(f"\n{'='*80}")
    print(f"Example 3: Custom Observability Integration")
    print(f"{'='*80}\n")

    print("📝 Custom integration pattern:\n")

    code_example = '''
# In your FastAPI server (src/api_server.py)

from fastapi import FastAPI, Request
import time

app = FastAPI()

@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Custom observability middleware"""

    # Start timing
    start_time = time.time()

    # Track request metadata
    metadata = {
        "path": request.url.path,
        "method": request.method,
        "user_id": request.headers.get("X-User-ID"),
    }

    # Process request
    response = await call_next(request)

    # Calculate metrics
    latency_ms = (time.time() - start_time) * 1000

    # Send to observability platform
    if LANGFUSE_AVAILABLE:
        langfuse.trace(
            name=f"{request.method} {request.url.path}",
            metadata=metadata,
            duration_ms=latency_ms,
        )

    # Or send to Phoenix
    if PHOENIX_AVAILABLE:
        px.log_trace(
            span_kind="SERVER",
            attributes=metadata,
            duration_ms=latency_ms,
        )

    return response
'''

    print(code_example)

    print(f"\n{'='*80}")
    print(f"💡 Integration Options:")
    print(f"{'='*80}")
    print(f"1. Middleware: Auto-track all requests")
    print(f"2. Decorator: Track specific endpoints")
    print(f"3. Manual: Explicit trace creation")


def comparison_guide():
    """Compare Langfuse vs Phoenix"""
    print(f"\n{'='*80}")
    print(f"🔍 Langfuse vs Phoenix Comparison")
    print(f"{'='*80}\n")

    print(f"{'Feature':<30} {'Langfuse':<20} {'Phoenix':<20}")
    print(f"{'-'*30} {'-'*20} {'-'*20}")
    print(f"{'Deployment':<30} {'Cloud/Self-hosted':<20} {'Local/Self-hosted':<20}")
    print(f"{'Cost':<30} {'Freemium':<20} {'Free (OSS)':<20}")
    print(f"{'Setup':<30} {'Easy':<20} {'Very Easy':<20}")
    print(f"{'Tracing':<30} {'✅ Full':<20} {'✅ Full':<20}")
    print(f"{'Cost Tracking':<30} {'✅ Built-in':<20} {'❌ Manual':<20}")
    print(f"{'User Analytics':<30} {'✅ Yes':<20} {'❌ No':<20}")
    print(f"{'Prompt Versioning':<30} {'✅ Yes':<20} {'❌ No':<20}")
    print(f"{'Embedding Analysis':<30} {'❌ No':<20} {'✅ Yes':<20}")
    print(f"{'Drift Detection':<30} {'❌ No':<20} {'✅ Yes':<20}")

    print(f"\n{'='*80}")
    print(f"📊 Recommendations:")
    print(f"{'='*80}")
    print(f"Use Langfuse when:")
    print(f"  • Need cost tracking per user/request")
    print(f"  • Want prompt versioning and A/B testing")
    print(f"  • Team collaboration on prompts")
    print(f"  • Production fintech application (we use this)")
    print(f"\nUse Phoenix when:")
    print(f"  • Need embedding drift detection")
    print(f"  • Want fully open-source solution")
    print(f"  • Local development and testing")
    print(f"  • Model performance debugging")


def main():
    """Run LLM observability examples"""
    print(f"{'='*80}")
    print(f"Advanced LLM Observability Examples")
    print(f"{'='*80}\n")

    print(f"💡 Start monitoring stack first:")
    print(f"   cd deployment")
    print(f"   docker-compose -f docker-compose.monitoring.yml --profile langfuse up -d\n")

    print(f"Or for Phoenix:")
    print(f"   docker-compose -f docker-compose.monitoring.yml --profile phoenix up -d\n")

    # Run examples
    langfuse_example()
    phoenix_example()
    custom_observability_example()
    comparison_guide()

    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Summary")
    print(f"{'='*80}")
    print(f"\n1. Langfuse: Full LLM observability platform")
    print(f"   - Cost tracking, user analytics, prompt versioning")
    print(f"   - Dashboard: http://localhost:3001")
    print(f"\n2. Phoenix: Open-source LLM monitoring")
    print(f"   - Trace visualization, embedding analysis")
    print(f"   - Dashboard: http://localhost:6006")
    print(f"\n3. Custom: Integrate with any platform")
    print(f"   - Middleware pattern for auto-tracking")
    print(f"\n💰 Production Recommendation: Use Langfuse")
    print(f"   - Built-in cost tracking")
    print(f"   - Perfect for FinTech compliance")
    print(f"   - Team collaboration features")
    print(f"\n📚 Next Steps:")
    print(f"   1. Start Langfuse: docker-compose --profile langfuse up -d")
    print(f"   2. Get API keys from dashboard")
    print(f"   3. Add to .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY")
    print(f"   4. Integrate with FastAPI middleware")


if __name__ == "__main__":
    main()
