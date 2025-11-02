#!/bin/bash

echo "========================================="
echo "Real Model Inference Test for vLLM"
echo "========================================="
echo ""
echo "This script tests real model loading and inference on GCP VM"
echo ""

# Function to test model loading
test_model_loading() {
    echo "1. Testing Model Loading..."
    echo "-----------------------------------------"

    echo "  Loading tiny model:"
    curl -X POST http://localhost:8000/models/tiny-model/load \
        -H "Content-Type: application/json" \
        -s | python -m json.tool

    echo ""
    echo "  Checking loaded models:"
    curl -s http://localhost:8000/models/loaded | python -m json.tool
    echo ""
}

# Function to test real inference
test_real_inference() {
    echo "2. Testing Real Inference..."
    echo "-----------------------------------------"

    echo "  Simple query (should use tiny-model):"
    curl -X POST http://localhost:8000/v1/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "What is 2+2?",
            "user_id": "test_user",
            "max_tokens": 50
        }' -s | python -m json.tool

    echo ""
    echo "  Financial query (should use appropriate model):"
    curl -X POST http://localhost:8000/v1/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Explain the impact of interest rate changes on bond prices",
            "user_id": "analyst_001",
            "max_tokens": 100,
            "metadata": {"user_tier": "premium"}
        }' -s | python -m json.tool
    echo ""
}

# Function to test model unloading
test_model_unloading() {
    echo "3. Testing Model Unloading..."
    echo "-----------------------------------------"

    echo "  Unloading tiny model:"
    curl -X POST http://localhost:8000/models/tiny-model/unload \
        -H "Content-Type: application/json" \
        -s | python -m json.tool

    echo ""
    echo "  Checking loaded models after unload:"
    curl -s http://localhost:8000/models/loaded | python -m json.tool
    echo ""
}

# Function to test GPU memory info
check_gpu_memory() {
    echo "4. GPU Memory Status..."
    echo "-----------------------------------------"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv
    else
        echo "  nvidia-smi not available"
    fi
    echo ""
}

# Main menu
case "$1" in
    load)
        test_model_loading
        ;;
    inference)
        test_real_inference
        ;;
    unload)
        test_model_unloading
        ;;
    gpu)
        check_gpu_memory
        ;;
    all)
        check_gpu_memory
        test_model_loading
        sleep 2
        test_real_inference
        sleep 2
        test_model_unloading
        check_gpu_memory
        ;;
    *)
        echo "Usage: $0 {load|inference|unload|gpu|all}"
        echo ""
        echo "  load      - Test model loading"
        echo "  inference - Test real inference"
        echo "  unload    - Test model unloading"
        echo "  gpu       - Check GPU memory status"
        echo "  all       - Run all tests"
        echo ""
        echo "Make sure server is running: ./server_manager.sh start"
        ;;
esac