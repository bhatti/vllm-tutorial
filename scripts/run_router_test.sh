#!/bin/bash

echo "=================================="
echo "Testing Intelligent Router"
echo "=================================="

# Run the intelligent router test
python tests/test_intelligent_router.py

echo ""
echo "=================================="
echo "To run with pytest:"
echo "python -m pytest tests/test_intelligent_routing.py -v"
echo "=================================="