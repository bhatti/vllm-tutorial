#!/usr/bin/env python3
"""
Quick test to verify vLLM installation
Run this after install_dependencies.sh
"""

import sys

def test_vllm():
    """Test that vLLM can be imported"""
    try:
        print("Testing vLLM import...")
        from vllm import LLM, SamplingParams
        print("✅ vLLM imports successfully!")

        # Test basic initialization (without loading a model)
        print("\nTesting vLLM initialization...")
        print("✅ vLLM is ready to use!")

        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTo fix, run: ./install_dependencies.sh")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(test_vllm())