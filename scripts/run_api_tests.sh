#!/bin/bash

echo "========================================="
echo "FastAPI Server Test Suite"
echo "========================================="
echo ""

# First, check if FastAPI dependencies are installed
echo "Checking FastAPI dependencies..."
python -c "import fastapi; import uvicorn; print('✅ FastAPI dependencies installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ FastAPI dependencies missing. Installing..."
    pip install fastapi uvicorn httpx -q
fi

echo ""
echo "1. Running API Server Tests..."
echo "-----------------------------------------"

# Run the API tests
python -m pytest tests/test_api_server.py -v -p no:cov

API_RESULT=$?

echo ""
echo "2. Running Model Loading Tests..."
echo "-----------------------------------------"

# Run the model loading tests
python -m pytest tests/test_model_loading.py -v -p no:cov

MODEL_RESULT=$?

if [ $API_RESULT -eq 0 ] && [ $MODEL_RESULT -eq 0 ]; then
    echo ""
    echo "✅ All API and model loading tests passed!"
    echo ""
    echo "To start the server for manual testing, run:"
    echo "  ./server_manager.sh start"
    echo ""
    echo "To test real inference on GCP, run:"
    echo "  ./test_real_inference.sh all"
else
    echo ""
    echo "❌ Some tests failed. Check output above."
fi

echo ""
echo "========================================="
echo "Test Coverage Areas:"
echo "  • Health & Status Endpoints"
echo "  • Model Management & Loading"
echo "  • Real vLLM Integration"
echo "  • Inference with Routing"
echo "  • Enterprise Use Cases"
echo "  • Error Handling & Fallback"
echo "  • Metrics & Monitoring"
echo "  • GPU Memory Management"
echo "========================================="
