#!/bin/bash
# End-to-End Demo: Complete vLLM Production Stack
#
# This demonstrates:
# 1. Intelligent routing between models
# 2. Budget tracking
# 3. Observability metrics
# 4. Prefix caching
# 5. Error handling
#
# Requirements:
# - vLLM installed in venv
# - GPU available (L4 or better)
# - ~20GB GPU memory

set -e

echo "========================================"
echo "vLLM Production Stack - End-to-End Demo"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not activated${NC}"
    echo "Please activate venv first:"
    echo "  source venv/bin/activate"
    exit 1
fi

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  nvidia-smi not found - GPU may not be available${NC}"
fi

echo -e "${BLUE}Step 1: Running End-to-End Integration Test${NC}"
echo "This will:"
echo "  - Process 11 Enterprise queries (simple → complex)"
echo "  - Route to appropriate models based on complexity"
echo "  - Track costs and enforce budget ($10/day limit)"
echo "  - Use prefix caching for 80% cost reduction"
echo "  - Collect all observability metrics"
echo ""

python3 tests/integration/test_end_to_end_fintech.py

echo ""
echo -e "${GREEN}✅ End-to-End Test Complete!${NC}"
echo ""

# Check if results were created
if [ -d "test_results" ]; then
    echo -e "${BLUE}Step 2: Test Results${NC}"
    echo "Results saved to: test_results/"

    # Show latest result file
    LATEST_RESULT=$(ls -t test_results/e2e_results_*.json | head -1)
    if [ -f "$LATEST_RESULT" ]; then
        echo ""
        echo "Latest result: $LATEST_RESULT"
        echo ""
        echo "Summary (from JSON):"
        cat "$LATEST_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
summary = data['summary']
print(f\"  Requests: {summary['total_requests']} total, {summary['successful_requests']} successful\")
print(f\"  Cost: \${summary['total_cost_usd']:.6f} (Budget: \${data['config']['daily_budget_usd']:.2f})\")
print(f\"  Tokens: {summary['total_tokens']:,}\")
print(f\"  Avg latency: {summary['avg_latency_ms']:.1f}ms\")
print(f\"  Cache hit rate: {summary['cache_hit_rate']*100:.1f}%\")
print(f\"  Models used: {', '.join(summary['models_used'].keys())}\")
"
    fi
fi

echo ""
echo -e "${BLUE}Step 3: What This Demonstrates${NC}"
echo ""
echo "✅ Intelligent Routing:"
echo "   - Simple queries → Phi-2 (fast, cheap)"
echo "   - Medium complexity → Mistral-7B (balanced)"
echo "   - Complex analysis → Llama-3-8B (capable)"
echo ""
echo "✅ Cost Optimization:"
echo "   - Prefix caching: 80% cost reduction for repeated prompts"
echo "   - FP8 quantization: 50% memory reduction"
echo "   - Budget enforcement: Stops at \$10/day limit"
echo ""
echo "✅ Observability:"
echo "   - Per-request metrics (latency, tokens, cost)"
echo "   - Model distribution tracking"
echo "   - Cache hit rate monitoring"
echo "   - Budget utilization tracking"
echo ""
echo "✅ Production Resilience:"
echo "   - Budget exceeded handling"
echo "   - Error tracking and reporting"
echo "   - Graceful degradation"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Demo Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results in test_results/ directory"
echo "  2. Start monitoring stack: docker-compose -f deployment/docker-compose.monitoring.yml up -d"
echo "  3. View Grafana dashboard: http://localhost:3000"
echo "  4. Run production API server: python src/api_server.py"
echo ""
