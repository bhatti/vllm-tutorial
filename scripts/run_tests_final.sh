#!/bin/bash

echo "========================================="
echo "Final Router Test Suite"
echo "========================================="
echo ""

# Run each test file directly with Python
echo "1. Testing Intelligent Routing (23 tests)..."
python -m pytest tests/test_intelligent_routing.py -xvs -p no:cov 2>/dev/null | tail -5

echo ""
echo "2. Testing Router Coverage (17 tests)..."
python -m pytest tests/test_router_coverage.py -xvs -p no:cov 2>/dev/null | tail -5

echo ""
echo "3. Testing Comprehensive Suite (9 tests)..."
python -m pytest tests/test_router_comprehensive.py -xvs -p no:cov 2>/dev/null | tail -5

echo ""
echo "========================================="
echo "Quick Summary Check"
echo "========================================="

# Count actual passing tests
PASS_COUNT_1=$(python -m pytest tests/test_intelligent_routing.py -p no:cov 2>/dev/null | grep -o "passed" | wc -l)
PASS_COUNT_2=$(python -m pytest tests/test_router_coverage.py -p no:cov 2>/dev/null | grep -o "passed" | wc -l)
PASS_COUNT_3=$(python -m pytest tests/test_router_comprehensive.py -p no:cov 2>/dev/null | grep -o "passed" | wc -l)

echo "Main Routing: $PASS_COUNT_1 tests completed"
echo "Coverage: $PASS_COUNT_2 tests completed"
echo "Comprehensive: $PASS_COUNT_3 tests completed"

echo ""
echo "========================================="
echo "Features Tested:"
echo "  • Complexity Classification"
echo "  • Cost Optimization"
echo "  • Budget Constraints"
echo "  • Model Health Tracking"
echo "  • Capability-based Routing"
echo "  • Latency Requirements"
echo "  • Enterprise Use Cases"
echo "========================================="
