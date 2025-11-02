#!/bin/bash

echo "========================================="
echo "Testing Model Loading API Fixes"
echo "========================================="
echo ""

# Run only the API tests to verify fixes
echo "Testing individual API endpoints..."
echo ""

echo "1. Testing get_loaded_models endpoint..."
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_get_loaded_models_endpoint -xvs -p no:cov
echo ""

echo "2. Testing generate endpoint..."
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_generate_endpoint -xvs -p no:cov
echo ""

echo "3. Testing generate fallback to mock..."
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_generate_fallback_to_mock -xvs -p no:cov
echo ""

echo "4. Testing load_model endpoint..."
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_load_model_endpoint -xvs -p no:cov
echo ""

echo "5. Testing unload_model endpoint..."
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_unload_model_endpoint -xvs -p no:cov
echo ""

echo "6. Testing model_not_found error..."
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_model_not_found -xvs -p no:cov
echo ""

echo "7. Testing model_loading_error..."
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_model_loading_error -xvs -p no:cov
echo ""

echo "========================================="
echo "Running all API tests together..."
echo "========================================="
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI -v -p no:cov

echo ""
echo "========================================="
echo "Summary: Test execution complete"
echo "========================================="