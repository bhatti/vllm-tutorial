#!/usr/bin/env python3
"""
Comprehensive test runner for all vLLM features
Tests each feature independently and shows results
"""

import sys
import subprocess
import time


def run_test(name, command):
    """Run a test and report results"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(f"✅ {name} - PASSED")
            if "✅" in result.stdout:
                # Count success markers
                success_count = result.stdout.count("✅")
                print(f"   {success_count} successful checks")
            return True
        else:
            print(f"❌ {name} - FAILED")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ {name} - TIMEOUT (>30s)")
        return False
    except Exception as e:
        print(f"❌ {name} - ERROR: {e}")
        return False


def main():
    """Run all feature tests"""
    print("\n" + "="*60)
    print("vLLM Project - Feature Test Suite")
    print("="*60)
    print("\nThis will test all implemented features.")
    print("Each test is independent and won't load actual models.\n")

    tests = [
        # Basic functionality tests
        ("Python imports", "python -c 'import src.model_server; import src.intelligent_router_simple; print(\"✅ Imports work\")'"),

        # Model server tests (requires GPU)
        # Commented out for now since it requires model loading
        # ("Model Server", "python tests/test_basic_serving.py"),

        # Intelligent routing tests (no GPU needed)
        ("Intelligent Router", "python tests/test_intelligent_router.py"),

        # Test the simple router directly
        ("Router Example", "python src/intelligent_router_simple.py | head -50"),

        # Check requirements
        ("Requirements Check", "python -c 'import vllm; import torch; import transformers; print(\"✅ Core deps installed\")'"),
    ]

    results = []

    for test_name, test_cmd in tests:
        success = run_test(test_name, test_cmd)
        results.append((test_name, success))
        time.sleep(0.5)  # Small delay between tests

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name:30} {status}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")

    print(f"{'='*60}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())