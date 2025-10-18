#!/usr/bin/env python3
"""
Test runner script for the NLP Document Library API
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests for the API"""
    project_root = Path(__file__).parent
    venv_python = project_root / "venv311" / "bin" / "python"

    if not venv_python.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        print(
            "Run: python3.11 -m venv venv311 && source venv311/bin/activate && pip install -r requirements.txt"
        )
        return False

    print("🧪 Running API tests...")
    print("=" * 50)

    try:
        # Run unit tests
        print("\n📋 Unit Tests:")
        result = subprocess.run(
            [str(venv_python), "-m", "pytest", "tests/unit/", "-v"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        unit_success = result.returncode == 0

        # Run integration tests
        print("\n🔗 Integration Tests:")
        result = subprocess.run(
            [str(venv_python), "-m", "pytest", "tests/integration/", "-v"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        integration_success = result.returncode == 0

        # Summary
        print("\n" + "=" * 50)
        if unit_success and integration_success:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
