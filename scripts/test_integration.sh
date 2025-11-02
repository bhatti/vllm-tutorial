#!/bin/bash

echo "========================================="
echo "Running Integration Tests (No FastAPI/HTTP)"
echo "========================================="
echo ""
echo "These tests verify business logic directly without HTTP endpoints"
echo ""

# Run integration tests
python -m pytest tests/test_model_loading_integration.py -v -p no:cov

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ All integration tests passed!"
    echo "========================================="
    echo ""
    echo "Integration tests verify:"
    echo "- Model manager initialization and configuration"
    echo "- Model loading/unloading logic"
    echo "- Generation logic with mocking"
    echo "- Model statistics tracking"
    echo "- Intelligent routing decisions"
    echo "- Error handling"
    echo ""
    echo "These tests cover the same functionality as the API tests"
    echo "but test the business logic directly instead of through HTTP."
else
    echo ""
    echo "========================================="
    echo "❌ Some integration tests failed"
    echo "========================================="
    exit 1
fi
