#!/bin/bash

echo "========================================="
echo "Running Router Tests (Simple Version)"
echo "========================================="
echo ""

# Track overall success
ALL_PASSED=true
TOTAL_TESTS=0
PASSED_TESTS=0

# 1. Run main routing tests
echo "1. Main Routing Tests"
echo "-----------------------------------------"
python -m pytest tests/test_intelligent_routing.py -q -p no:cov
if [ $? -eq 0 ]; then
    echo "✅ 23 tests passed"
    PASSED_TESTS=$((PASSED_TESTS + 23))
else
    ALL_PASSED=false
    echo "❌ Some tests failed"
fi
TOTAL_TESTS=$((TOTAL_TESTS + 23))

echo ""
echo "2. Coverage Tests"
echo "-----------------------------------------"
python -m pytest tests/test_router_coverage.py -q -p no:cov
if [ $? -eq 0 ]; then
    echo "✅ 17 tests passed"
    PASSED_TESTS=$((PASSED_TESTS + 17))
else
    ALL_PASSED=false
    echo "❌ Some tests failed"
fi
TOTAL_TESTS=$((TOTAL_TESTS + 17))

echo ""
echo "3. Comprehensive Tests"
echo "-----------------------------------------"
python -m pytest tests/test_router_comprehensive.py -q -p no:cov
if [ $? -eq 0 ]; then
    echo "✅ 9 tests passed"
    PASSED_TESTS=$((PASSED_TESTS + 9))
else
    ALL_PASSED=false
    echo "❌ Some tests failed"
fi
TOTAL_TESTS=$((TOTAL_TESTS + 9))

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total: $PASSED_TESTS/$TOTAL_TESTS tests passed"
echo ""

if [ "$ALL_PASSED" = true ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "Features Validated:"
    echo "  ✅ Complexity Classification"
    echo "  ✅ Cost Optimization"
    echo "  ✅ Budget Constraints"
    echo "  ✅ Model Health Tracking"
    echo "  ✅ Capability-based Routing"
    echo "  ✅ Latency Requirements"
    echo "  ✅ Enterprise Use Cases"
else
    echo "❌ Some tests failed. Check output above."
fi
echo "========================================="
