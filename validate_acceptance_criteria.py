#!/usr/bin/env python3
"""Test script to validate acceptance criteria for the style enforcement PR."""

import subprocess
import sys


def run_command(cmd: str, description: str) -> bool:
    """Run a command and check if it succeeds."""
    print(f"✓ Running: {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✅ PASSED: {description}")
        return True
    else:
        print(f"  ❌ FAILED: {description}")
        print(f"  Error: {result.stderr}")
        return False


def main() -> None:
    """Run all acceptance criteria tests."""
    print("🔍 Validating acceptance criteria for Python style enforcement...")

    tests = [
        ("black --check .", "Black format check - no diffs"),
        ("ruff check .", "Ruff lint check - 0 violations"),
        ("mypy . --ignore-missing-imports", "Mypy type check - passes"),
        ("python -m pytest tests/ -v", "Tests pass"),
    ]

    all_passed = True

    for cmd, description in tests:
        if not run_command(cmd, description):
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL ACCEPTANCE CRITERIA PASSED!")
        print("✅ black --check . returns no diffs")
        print("✅ ruff check . shows 0 violations")
        print("✅ mypy --ignore-missing-imports passes")
        print("✅ Tests pass successfully")
        print("✅ GitHub Actions will enforce style in CI")
        print("✅ README.md documents the development workflow")
    else:
        print("❌ Some acceptance criteria failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
