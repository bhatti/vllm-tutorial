#!/bin/bash

echo "========================================"
echo "Running Tests with Coverage Analysis"
echo "========================================"
echo ""

# First run tests without coverage to ensure they pass
echo "1. Running Tests (without coverage)..."
echo "----------------------------------------"
python -m pytest tests/test_intelligent_routing.py -v

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed! Now checking coverage..."
    echo ""

    # Run with coverage
    echo "2. Running Tests with Coverage Analysis..."
    echo "----------------------------------------"
    python -m pytest tests/test_intelligent_routing.py -v \
        --cov=src.intelligent_router_simple \
        --cov-report=term-missing \
        --cov-report=term:skip-covered \
        --cov-fail-under=90

    echo ""
    echo "========================================"
    echo "3. Testing with Simple Router Script..."
    echo "----------------------------------------"
    python tests/test_intelligent_router.py

    echo ""
    echo "========================================"
    echo "4. Running All Feature Tests..."
    echo "----------------------------------------"
    python tests/test_all_features.py
else
    echo ""
    echo "❌ Tests failed. Fix tests before checking coverage."
    exit 1
fi

echo ""
echo "========================================"
echo "Summary:"
echo "- test_intelligent_routing.py: Full TDD test suite with pytest"
echo "- test_intelligent_router.py: Simple functional tests"
echo "- test_all_features.py: Integration test runner"
echo ""
echo "TDD Requirement: >90% coverage target"
echo "========================================"