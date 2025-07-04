#!/usr/bin/env python3
"""
Validate CI setup locally before pushing to GitHub.

This script simulates the GitHub Actions workflow to ensure
everything works correctly in the CI environment.
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()

    print(f"Exit code: {result.returncode}")
    print(f"Duration: {end_time - start_time:.2f}s")

    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    return result.returncode == 0

def main():
    """Validate CI setup."""
    print("SlicerUltrasound CI Validation")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists("run_all_tests.py"):
        print("❌ Error: run_all_tests.py not found. Run from project root.")
        return 1

    # Simulate GitHub Actions workflow
    steps = [
        ("python -m pip install --upgrade pip", "Upgrade pip"),
        ("pip install -r requirements-test.txt", "Install test dependencies"),
        ("python run_all_tests.py", "Run automated test runner"),
        ("find AnnotateUltrasound/tests -name 'test_*.py' -type f | wc -l", "Count AnnotateUltrasound tests"),
        ("find AnonymizeUltrasound/tests -name 'test_*.py' -type f | wc -l", "Count AnonymizeUltrasound tests"),
        ("find tests -name 'test_*.py' -type f | wc -l", "Count core/shared tests"),
        ("find AnonymizeUltrasound/common/tests -name 'test_*.py' -type f | wc -l", "Count common tests"),
    ]

    results = []
    for cmd, description in steps:
        success = run_command(cmd, description)
        results.append((description, success))

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed} steps")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print(f"\n🎉 CI VALIDATION PASSED! 🎉")
        print("GitHub Actions workflow should work correctly.")
        return 0
    else:
        print(f"\n💥 {failed} STEP(S) FAILED 💥")
        print("Fix issues before pushing to GitHub.")
        return 1

if __name__ == "__main__":
    sys.exit(main())