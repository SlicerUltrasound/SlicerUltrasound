#!/usr/bin/env python3
"""
Run all unit tests for SlicerUltrasound project.

This script runs tests from all module directories in the correct order
to avoid import conflicts.
"""

import subprocess
import sys
import os
import time

def run_tests_in_directory(directory, description):
    """Run tests in a specific directory."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"Directory: {directory}")
    print(f"{'='*60}")

    start_time = time.time()

    if directory == "tests":
        # Run from root directory
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"],
                              capture_output=False)
    else:
        # Change to module directory and run tests
        original_dir = os.getcwd()
        try:
            module_dir = directory.split('/')[0]
            os.chdir(module_dir)
            result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"],
                                  capture_output=False)
        finally:
            os.chdir(original_dir)

    end_time = time.time()
    duration = end_time - start_time

    if result.returncode == 0:
        print(f"\n✅ {description} PASSED ({duration:.2f}s)")
        return True
    else:
        print(f"\n❌ {description} FAILED ({duration:.2f}s)")
        return False

def run_common_tests():
    """Run tests in AnonymizeUltrasound/common/tests/."""
    print(f"\n{'='*60}")
    print(f"Running AnonymizeUltrasound Common Tests")
    print(f"Directory: AnonymizeUltrasound/common/tests/")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run([sys.executable, "-m", "pytest",
                           "AnonymizeUltrasound/common/tests/", "-v"],
                          capture_output=False)
    end_time = time.time()
    duration = end_time - start_time

    if result.returncode == 0:
        print(f"\n✅ AnonymizeUltrasound Common Tests PASSED ({duration:.2f}s)")
        return True
    else:
        print(f"\n❌ AnonymizeUltrasound Common Tests FAILED ({duration:.2f}s)")
        return False

def main():
    """Run all unit tests."""
    print("SlicerUltrasound Unit Test Runner")
    print("=" * 60)

    overall_start = time.time()
    results = []

    # Test directories and descriptions
    test_suites = [
        ("AnnotateUltrasound/tests", "AnnotateUltrasound Module Tests (43 tests)"),
        ("AnonymizeUltrasound/tests", "AnonymizeUltrasound Module Tests (9 tests)"),
        ("tests", "Core/Shared Tests (38 tests)"),
    ]

    # Run each test suite
    for directory, description in test_suites:
        success = run_tests_in_directory(directory, description)
        results.append((description, success))

    # Run common tests separately
    success = run_common_tests()
    results.append(("AnonymizeUltrasound Common Tests (5 tests)", success))

    # Summary
    overall_end = time.time()
    total_duration = overall_end - overall_start

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
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

    print(f"\nTotal: {passed + failed} test suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Total Tests: ~95 unit tests")

    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n💥 {failed} TEST SUITE(S) FAILED 💥")
        return 1

if __name__ == "__main__":
    sys.exit(main())