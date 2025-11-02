#!/bin/bash

echo "========================================="
echo "Testing Model Loading Fixes"
echo "========================================="
echo ""

# Run only the API tests to verify fixes
python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_get_loaded_models_endpoint -xvs -p no:cov
echo ""

python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_generate_endpoint -xvs -p no:cov
echo ""

python -m pytest tests/test_model_loading.py::TestModelLoadingAPI::test_generate_fallback_to_mock -xvs -p no:cov
echo ""

echo "========================================="
echo "Running full model loading test suite..."
echo "========================================="
python -m pytest tests/test_model_loading.py -v -p no:cov