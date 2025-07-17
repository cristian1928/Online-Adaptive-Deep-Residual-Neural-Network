import subprocess
import sys
from pathlib import Path


def test_type_hints_are_clean() -> None:
    """Fail if `mypy --strict` reports any type issues."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(repo_root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Type‑hint errors detected:\n{result.stdout}\n{result.stderr}"
    )