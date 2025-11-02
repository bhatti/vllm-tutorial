#!/bin/bash

echo "========================================="
echo "Complete Test Suite - Dependency Injection Fix"
echo "========================================="
echo ""
echo "This script verifies the FastAPI dependency injection fix"
echo "for the 7 failing model loading API tests."
echo ""

# Function to print section headers
print_header() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
    echo ""
}

# Check if we're on GCP VM (optional)
if command -v nvidia-smi &> /dev/null; then
    print_header "GPU Information"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Run the 7 API tests that were previously failing
print_header "Testing Previously Failing API Endpoints"

FAILED_TESTS=0

echo "Test 1/7: test_get_loaded_models_endpoint"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_get_loaded_models_endpoint -xvs -p no:cov
if [ $? -ne 0 ]; then FAILED_TESTS=$((FAILED_TESTS+1)); fi
echo ""

echo "Test 2/7: test_generate_endpoint"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_generate_endpoint -xvs -p no:cov
if [ $? -ne 0 ]; then FAILED_TESTS=$((FAILED_TESTS+1)); fi
echo ""

echo "Test 3/7: test_generate_fallback_to_mock"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_generate_fallback_to_mock -xvs -p no:cov
if [ $? -ne 0 ]; then FAILED_TESTS=$((FAILED_TESTS+1)); fi
echo ""

echo "Test 4/7: test_load_model_endpoint"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_load_model_endpoint -xvs -p no:cov
if [ $? -ne 0 ]; then FAILED_TESTS=$((FAILED_TESTS+1)); fi
echo ""

echo "Test 5/7: test_unload_model_endpoint"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_unload_model_endpoint -xvs -p no:cov
if [ $? -ne 0 ]; then FAILED_TESTS=$((FAILED_TESTS+1)); fi
echo ""

echo "Test 6/7: test_model_not_found"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_model_not_found -xvs -p no:cov
if [ $? -ne 0 ]; then FAILED_TESTS=$((FAILED_TESTS+1)); fi
echo ""

echo "Test 7/7: test_model_loading_error"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_model_loading_error -xvs -p no:cov
if [ $? -ne 0 ]; then FAILED_TESTS=$((FAILED_TESTS+1)); fi
echo ""

print_header "Individual Test Results"
if [ $FAILED_TESTS -eq 0 ]; then
    echo "✅ SUCCESS: All 7 previously failing tests now pass!"
else
    echo "❌ FAILURE: $FAILED_TESTS out of 7 tests failed"
    exit 1
fi

# Run all API tests together
print_header "Running All API Tests Together"
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI -v -p no:cov
if [ $? -ne 0 ]; then
    echo "❌ Some tests failed when run together"
    exit 1
fi

# Run complete test suite
print_header "Running Complete Test Suite"
python -m pytest tests/test_model_loading.py -v -p no:cov
if [ $? -ne 0 ]; then
    echo "❌ Some tests in the complete suite failed"
    exit 1
fi

# Check test coverage
print_header "Test Coverage Report"
python -m pytest tests/test_model_loading.py --cov=src --cov-report=term-missing -p no:warnings

print_header "All Tests Passed Successfully!"
echo "✅ All 7 previously failing API tests now pass"
echo "✅ Complete test suite passes"
echo "✅ Dependency injection pattern properly implemented"
echo ""
echo "The fix involved:"
echo "1. Updated API endpoints to use FastAPI's Depends() for dependency injection"
echo "2. Updated tests to use app.dependency_overrides for proper mocking"
echo "3. Avoided global variable manipulation and lifespan event issues"
echo ""
echo "See API_TEST_FIX_SUMMARY.md for complete details."
