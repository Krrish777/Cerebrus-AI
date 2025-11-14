"""
Test runner script for Cerebrus AI project.

This script provides a convenient way to run tests with different configurations.
"""

import sys
import pytest
from pathlib import Path


def run_tests():
    """Run all tests with appropriate configuration"""
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))
    
    # Test configuration
    test_args = [
        str(project_root / "tests"),  # Test directory
        "-v",                         # Verbose output
        "--tb=short",                 # Short traceback format
        "--strict-markers",           # Require markers to be defined
        "-x",                         # Stop on first failure
        "--color=yes",               # Colored output
    ]
    
    print("Running Cerebrus AI Tests...")
    print(f"Project root: {project_root}")
    print(f"Test arguments: {' '.join(test_args)}")
    print("-" * 50)
    
    # Run pytest
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


def run_specific_test(test_file=None, test_function=None):
    """Run a specific test file or function"""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))
    
    test_args = ["-v", "--tb=short"]
    
    if test_file:
        test_path = project_root / "tests" / test_file
        if not test_path.exists():
            print(f"Test file not found: {test_path}")
            return 1
        test_args.append(str(test_path))
        
        if test_function:
            test_args[-1] += f"::{test_function}"
    
    return pytest.main(test_args)


def run_coverage_tests():
    """Run tests with coverage reporting"""
    try:
        import pytest_cov
    except ImportError:
        print("pytest-cov not installed. Install with: pip install pytest-cov")
        return 1
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))
    
    test_args = [
        str(project_root / "tests"),
        "-v",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-branch"
    ]
    
    return pytest.main(test_args)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Cerebrus AI tests")
    parser.add_argument("--file", "-f", help="Specific test file to run")
    parser.add_argument("--function", "-t", help="Specific test function to run")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    
    args = parser.parse_args()
    
    if args.coverage:
        exit_code = run_coverage_tests()
    elif args.file:
        exit_code = run_specific_test(args.file, args.function)
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)