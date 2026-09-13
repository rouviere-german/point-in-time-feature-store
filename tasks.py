"""Cross-platform task runner. The Makefile delegates here, so both always agree.

Usage: uv run python tasks.py <target> [<target> ...]

Targets:
  lint       ruff check + ruff format --check
  format     ruff format + ruff check --fix
  typecheck  mypy --strict (configured in pyproject.toml)
  test       pytest with coverage (fails under the floor in pyproject.toml)
  bench      benchmarks with regression thresholds (bench/thresholds.json)
  demo       offline demo (FakeProvider, no API key)
  bandit     bandit static analysis of src/
  pip-audit  pip-audit of the locked dependency set
  audit      bandit + pip-audit
  baseline   scripts/check_baseline.py
  record     optional, budget-capped cassette recording (needs ANTHROPIC_API_KEY)
  check      lint + typecheck + test + audit + baseline
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _run(*command: str) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def lint() -> None:
    _run(PY, "-m", "ruff", "check", ".")
    _run(PY, "-m", "ruff", "format", "--check", ".")


def format_code() -> None:
    _run(PY, "-m", "ruff", "format", ".")
    _run(PY, "-m", "ruff", "check", "--fix", ".")


def typecheck() -> None:
    _run(PY, "-m", "mypy")


def test() -> None:
    _run(PY, "-m", "pytest", "--cov", "--cov-report=term-missing", "--cov-report=xml")


def bench() -> None:
    _run(PY, "bench/bench_baseline.py", "--check")


def demo() -> None:
    package = str(_pyproject()["tool"]["portfolio"]["package"])
    _run(PY, "-m", package, "demo")


def bandit() -> None:
    _run(PY, "-m", "bandit", "-c", "pyproject.toml", "-r", "src", "-q")


def pip_audit() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("pip-audit needs uv on PATH (https://docs.astral.sh/uv/)")
    ignores = _pyproject()["tool"]["portfolio"]["audit"]["ignore_vulns"]
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements-audit.txt"
        _run(
            uv,
            "export",
            "--frozen",
            "--format",
            "requirements-txt",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        )
        command = [
            uv,
            "tool",
            "run",
            "--from",
            "pip-audit>=2.7,<3",
            "pip-audit",
            "--requirement",
            str(requirements),
            "--disable-pip",
            "--progress-spinner",
            "off",
        ]
        for entry in ignores:
            command += ["--ignore-vuln", str(entry["id"])]
        _run(*command)


def audit() -> None:
    bandit()
    pip_audit()


def baseline() -> None:
    _run(PY, "scripts/check_baseline.py")


def record() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("record needs ANTHROPIC_API_KEY; tests, demo and bench never do.")
    raise SystemExit(
        "The template has no recording workflow. Repos add bench/record.py that wraps "
        "AnthropicProvider in RecordingProvider with a BudgetLedger cap."
    )


def check() -> None:
    for task in (lint, typecheck, test, audit, baseline):
        task()


TASKS: dict[str, Callable[[], None]] = {
    "lint": lint,
    "format": format_code,
    "typecheck": typecheck,
    "test": test,
    "bench": bench,
    "demo": demo,
    "bandit": bandit,
    "pip-audit": pip_audit,
    "audit": audit,
    "baseline": baseline,
    "record": record,
    "check": check,
}


def main(argv: list[str]) -> int:
    if not argv or any(name not in TASKS for name in argv):
        print(__doc__)
        return 2
    for name in argv:
        try:
            TASKS[name]()
        except subprocess.CalledProcessError as exc:
            print(f"task '{name}' failed (exit code {exc.returncode})", file=sys.stderr)
            return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
