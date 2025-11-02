#!/bin/bash

echo "========================================="
echo "GCP VM Test Suite for vLLM Project"
echo "========================================="
echo ""
echo "This script helps test the FastAPI server incrementally on GCP"
echo ""

# Function to check dependencies
check_deps() {
    echo "1. Checking Dependencies..."
    echo "-----------------------------------------"
    python -c "import fastapi; print('  ✅ FastAPI installed')" 2>/dev/null || echo "  ❌ FastAPI missing"
    python -c "import uvicorn; print('  ✅ Uvicorn installed')" 2>/dev/null || echo "  ❌ Uvicorn missing"
    python -c "import httpx; print('  ✅ HTTPX installed')" 2>/dev/null || echo "  ❌ HTTPX missing"
    python -c "import vllm; print('  ✅ vLLM installed')" 2>/dev/null || echo "  ❌ vLLM missing"
    echo ""
}

# Function to run basic tests
run_basic_tests() {
    echo "2. Running Basic Health Tests..."
    echo "-----------------------------------------"
    python -m pytest tests/test_api_server.py::TestHealthEndpoints -v -p no:cov 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ Health endpoints working"
    else
        echo "  ❌ Health endpoints failed"
    fi
    echo ""
}

# Function to start server in background
start_server() {
    echo "3. Starting API Server..."
    echo "-----------------------------------------"
    ./server_manager.sh start
    echo ""
}

# Function to test with curl
test_with_curl() {
    echo "4. Testing Endpoints with curl..."
    echo "-----------------------------------------"

    echo "  Testing /health:"
    curl -s http://localhost:8000/health | python -m json.tool | head -5

    echo ""
    echo "  Testing /models:"
    curl -s http://localhost:8000/models | python -m json.tool | head -10

    echo ""
    echo "  Testing simple inference:"
    curl -X POST http://localhost:8000/v1/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "What is the interest rate?",
            "user_id": "test_user"
        }' -s | python -m json.tool | head -10
    echo ""
}

# Function to test FinTech examples
test_fintech() {
    echo "5. Testing FinTech Use Cases..."
    echo "-----------------------------------------"

    echo "  Earnings Analysis:"
    curl -X POST http://localhost:8000/v1/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Analyze Q3 earnings: Revenue $5.2B, EPS $2.34",
            "user_id": "analyst_001",
            "metadata": {"user_tier": "premium"}
        }' -s | python -m json.tool | grep -E "(complexity|model_used)"

    echo ""
    echo "  Risk Assessment:"
    curl -X POST http://localhost:8000/v1/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Calculate VAR for portfolio",
            "user_id": "risk_001",
            "metadata": {"user_tier": "enterprise"}
        }' -s | python -m json.tool | grep -E "(complexity|model_used)"
    echo ""
}

# Function to stop server
stop_server() {
    echo "6. Stopping Server..."
    echo "-----------------------------------------"
    ./server_manager.sh stop
    echo ""
}

# Main menu
case "$1" in
    deps)
        check_deps
        ;;
    basic)
        run_basic_tests
        ;;
    start)
        start_server
        ;;
    test)
        test_with_curl
        ;;
    fintech)
        test_fintech
        ;;
    stop)
        stop_server $2
        ;;
    all)
        check_deps
        run_basic_tests
        start_server
        test_with_curl
        test_fintech
        echo "Server still running. Stop with: ./server_manager.sh stop"
        ;;
    *)
        echo "Usage: $0 {deps|basic|start|test|fintech|stop|all}"
        echo ""
        echo "  deps    - Check dependencies"
        echo "  basic   - Run basic health tests"
        echo "  start   - Start API server"
        echo "  test    - Test with curl"
        echo "  fintech - Test FinTech examples"
        echo "  stop    - Stop server"
        echo "  all     - Run all tests"
        echo ""
        echo "Recommended sequence:"
        echo "  1. ./test_on_gcp.sh deps"
        echo "  2. ./test_on_gcp.sh basic"
        echo "  3. ./test_on_gcp.sh start"
        echo "  4. ./test_on_gcp.sh test"
        echo "  5. ./test_on_gcp.sh fintech"
        echo "  6. ./test_on_gcp.sh stop"
        echo ""
        echo "Or use server_manager.sh directly:"
        echo "  ./server_manager.sh {start|stop|restart|status|logs|test}"
        ;;
esac