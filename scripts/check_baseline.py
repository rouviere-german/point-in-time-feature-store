#!/usr/bin/env python3
"""Fail if any piece of the shared portfolio baseline is missing (standard library only).

Usage: python scripts/check_baseline.py [--root PATH]

The kind of repo comes from ``[tool.portfolio] kind`` in pyproject.toml: ``template`` (both kits),
``agentic`` or ``ml``. Evidence references in CLAIMS.md and docs/THREAT_MODEL.md, written as
backticked ``tests/...py::test_name``, ``bench/...py`` or ``.github/workflows/ci.yml#job``, must
resolve to a real test function, file or CI job.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

COMMON_FILES = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CLAIMS.md",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    ".pre-commit-config.yaml",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    "tasks.py",
    ".github/workflows/ci.yml",
    ".github/dependabot.yml",
    ".semgrep/rules.yml",
    "docs/THREAT_MODEL.md",
    "scripts/check_baseline.py",
)
CORE_MODULES = (
    "core/config.py",
    "core/logging.py",
    "core/redaction.py",
    "core/tracing.py",
    "core/metrics.py",
    "core/app.py",
    "core/security.py",
    "core/ratelimit.py",
    "core/middleware.py",
    "core/errors.py",
    "core/healthcheck.py",
)
AGENTIC_MODULES = (
    "agentic/providers/base.py",
    "agentic/providers/fake.py",
    "agentic/providers/replay.py",
    "agentic/providers/anthropic.py",
    "agentic/envelope.py",
    "agentic/tools.py",
    "agentic/approval.py",
    "agentic/budget.py",
    "agentic/audit.py",
)
ML_MODULES = (
    "ml/contracts.py",
    "ml/manifest.py",
    "ml/registry.py",
    "ml/drift.py",
    "ml/serialization.py",
)
README_SECTIONS = ("Architecture", "Quickstart", "Results", "Security", "Non-goals")
CI_REQUIREMENTS = (
    ("ruff lint", r"ruff check"),
    ("ruff format check", r"ruff format --check"),
    ("mypy", r"\bmypy\b"),
    ("pytest", r"\bpytest\b"),
    ("coverage", r"--cov\b"),
    ("bandit", r"\bbandit\b"),
    ("semgrep", r"\bsemgrep\b"),
    ("semgrep repo rules", r"\.semgrep/rules\.yml"),
    ("pip-audit", r"pip-audit"),
    ("gitleaks", r"gitleaks"),
    ("docker build", r"docker/build-push-action|docker build"),
    ("trivy", r"trivy"),
    ("syft SBOM", r"syft|anchore/sbom-action"),
    ("CycloneDX", r"cyclonedx"),
    ("baseline check", r"check_baseline\.py"),
    ("bench regression thresholds", r"bench_baseline\.py --check"),
    ("least-privilege permissions", r"^permissions:"),
)
TASK_TARGETS = ("lint", "typecheck", "test", "bench", "demo", "audit", "baseline")
STRIDE = (
    "Spoofing",
    "Tampering",
    "Repudiation",
    "Information disclosure",
    "Denial of service",
    "Elevation of privilege",
)
DEV_TOOLS = ("pytest", "pytest-cov", "hypothesis", "ruff", "mypy", "bandit")
# Built from parts so the global search-and-replace rename in TEMPLATE_GUIDE.md leaves these
# sentinels intact.
TEMPLATE_PROJECT = "portfolio" + "-template"
TEMPLATE_PACKAGE = "portfolio" + "_template"
_EVIDENCE = re.compile(
    r"`((?:tests|bench|scripts)/[A-Za-z0-9_./-]+\.py)(?:::([A-Za-z_][A-Za-z0-9_]*))?`"
    r"|`(\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml)#([A-Za-z0-9_-]+)`"
)
# Pickle-family imports, plus pandas' pickle I/O (the ML kit ships pandas).
_BANNED_SERIALIZATION = re.compile(
    r"^\s*(?:import|from)\s+(?:pickle|cPickle|_pickle|dill|joblib|cloudpickle|shelve)\b"
    r"|\b(?:read|to)_pickle\s*\(",
    re.MULTILINE,
)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, ok: object, message: str) -> None:
        self.checks += 1
        if not ok:
            self.failures.append(message)


def _read(root: Path, relative: str) -> str:
    path = root / relative
    # Normalize CRLF so Windows checkouts (core.autocrlf) are checked the same way.
    return path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.is_file() else ""


def _section(markdown: str, title: str) -> str | None:
    match = re.search(rf"^## {re.escape(title)}[ \t]*$(.*?)(?=^## |\Z)", markdown, re.M | re.S)
    return match.group(1) if match else None


def check_pyproject(root: Path, report: Report) -> tuple[str, str]:
    try:
        data: dict[str, Any] = tomllib.loads(_read(root, "pyproject.toml"))
    except tomllib.TOMLDecodeError as exc:
        report.require(False, f"pyproject.toml: invalid TOML ({exc})")
        return "template", TEMPLATE_PACKAGE
    tool = data.get("tool", {})
    portfolio = tool.get("portfolio", {})
    kind = str(portfolio.get("kind", ""))
    package = str(portfolio.get("package", ""))
    project = data.get("project", {})
    report.require(
        kind in {"template", "agentic", "ml"},
        "pyproject: [tool.portfolio] kind must be template, agentic or ml",
    )
    report.require(
        package and (root / "src" / package / "__init__.py").is_file(),
        f"pyproject: package src/{package}/ not found",
    )
    report.require(
        "3.12" in str(project.get("requires-python", "")),
        "pyproject: requires-python must target 3.12",
    )
    report.require(
        tool.get("mypy", {}).get("strict") is True, "pyproject: mypy strict = true is required"
    )
    fail_under = tool.get("coverage", {}).get("report", {}).get("fail_under", 0)
    report.require(
        isinstance(fail_under, int | float) and fail_under >= 85,
        "pyproject: coverage fail_under must be >= 85",
    )
    markers = " ".join(tool.get("pytest", {}).get("ini_options", {}).get("markers", []))
    report.require(
        "unit:" in markers and "integration:" in markers,
        "pyproject: pytest markers unit and integration are required",
    )
    dev = [str(item) for item in data.get("dependency-groups", {}).get("dev", [])]
    for name in DEV_TOOLS:
        report.require(
            any(re.match(rf"{re.escape(name)}(\[|[<>=!~ ]|$)", dep) for dep in dev),
            f"pyproject: dev dependency {name} is missing",
        )
    if kind != "template":
        report.require(
            project.get("name") != TEMPLATE_PROJECT and package != TEMPLATE_PACKAGE,
            "pyproject: instantiated repos must rename the project and package "
            "(see TEMPLATE_GUIDE.md)",
        )
    return kind, package


def check_files(root: Path, report: Report, kind: str, package: str) -> None:
    required = list(COMMON_FILES)
    if kind in {"template", "ml"}:
        required.append("MODEL_CARD.md")
    if kind == "template":
        required.append("TEMPLATE_GUIDE.md")
    for relative in required:
        report.require((root / relative).is_file(), f"missing required file: {relative}")
    modules = list(CORE_MODULES)
    if kind in {"template", "agentic"}:
        modules += AGENTIC_MODULES
    if kind in {"template", "ml"}:
        modules += ML_MODULES
    for module in modules:
        report.require(
            (root / "src" / package / module).is_file(),
            f"missing baseline module: src/{package}/{module}",
        )
    report.require(
        any((root / "bench").glob("*.py")), "bench/: at least one benchmark script is required"
    )
    adrs = [p for p in (root / "docs" / "adr").glob("*.md") if re.match(r"^\d{4}-.+\.md$", p.name)]
    report.require(adrs, "docs/adr/: at least one NNNN-title.md ADR is required")


def check_readme(root: Path, report: Report) -> None:
    readme = _read(root, "README.md")
    for title in README_SECTIONS:
        report.require(
            _section(readme, title) is not None, f"README.md: missing '## {title}' section"
        )
    architecture = _section(readme, "Architecture") or ""
    report.require(
        "```mermaid" in architecture, "README.md: Architecture section needs a mermaid diagram"
    )
    results = _section(readme, "Results") or ""
    report.require(
        "bench/" in results, "README.md: Results must cite the bench/ command that produced them"
    )


def check_ci(root: Path, report: Report) -> None:
    workflow = _read(root, ".github/workflows/ci.yml")
    for label, pattern in CI_REQUIREMENTS:
        report.require(
            re.search(pattern, workflow, re.M | re.I),
            f"ci.yml: {label} step is missing (pattern {pattern!r})",
        )
    dependabot = _read(root, ".github/dependabot.yml")
    for ecosystem in ("uv", "github-actions", "docker"):
        report.require(
            f'"{ecosystem}"' in dependabot or f" {ecosystem}\n" in dependabot,
            f"dependabot.yml: {ecosystem} ecosystem is missing",
        )


def check_containers(root: Path, report: Report) -> None:
    # Join "\" line continuations so multi-line instructions are matched as one line.
    dockerfile = re.sub(r"\\\n[ \t]*", " ", _read(root, "Dockerfile"))
    args = dict(re.findall(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)", dockerfile, re.M))
    stages: set[str] = set()
    froms = re.findall(
        r"^FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", dockerfile, re.M | re.I
    )
    resolved_images: list[str] = []
    for image, alias in froms:
        resolved = re.sub(
            r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda m: args.get(m.group(1), m.group(0)), image
        )
        if resolved not in stages:
            resolved_images.append(resolved)
            report.require(
                "@sha256:" in resolved, f"Dockerfile: base image {image} is not pinned by digest"
            )
        if alias:
            stages.add(alias)
    report.require(len(froms) >= 2, "Dockerfile: a multi-stage build is required")
    report.require(
        any(img.startswith("python:3.12-slim") for img in resolved_images),
        "Dockerfile: python:3.12-slim base is required",
    )
    users = re.findall(r"^USER\s+(\S+)", dockerfile, re.M)
    report.require(
        users and users[-1].split(":")[0] not in {"root", "0"},
        "Dockerfile: final USER must be non-root",
    )
    report.require(
        re.search(r"^HEALTHCHECK\s.*python", dockerfile, re.M),
        "Dockerfile: HEALTHCHECK must use a Python probe",
    )
    compose = _read(root, "docker-compose.yml")
    for needle in ("read_only: true", "no-new-privileges", "cap_drop"):
        report.require(needle in compose, f"docker-compose.yml: {needle} hardening is missing")


def check_hygiene(root: Path, report: Report, package: str) -> None:
    gitignore = _read(root, ".gitignore").splitlines()
    report.require(
        ".env" in [line.strip() for line in gitignore], ".gitignore: .env must be ignored"
    )
    for line in _read(root, ".env.example").splitlines():
        key, _, value = line.partition("=")
        if not line.startswith("#") and re.search(r"(API_KEY|SECRET|TOKEN|PASSWORD)$", key.strip()):
            report.require(not value.strip(), f".env.example: {key.strip()} must be empty")
    license_text = _read(root, "LICENSE")
    report.require(
        "MIT License" in license_text and "Copyright (c) 2026 Rouviere German" in license_text,
        "LICENSE: MIT with 'Copyright (c) 2026 Rouviere German' is required",
    )
    makefile, tasks = _read(root, "Makefile"), _read(root, "tasks.py")
    for target in TASK_TARGETS:
        report.require(
            re.search(rf"^{target}:", makefile, re.M), f"Makefile: target {target} is missing"
        )
        report.require(
            re.search(rf'"{target}"\s*:', tasks), f"tasks.py: target {target} is missing"
        )
    report.require(
        re.search(r"(?im)^#+ .*report", _read(root, "SECURITY.md")),
        "SECURITY.md: a vulnerability reporting section is required",
    )
    for path in sorted((root / "src" / package).rglob("*.py")):
        report.require(
            not _BANNED_SERIALIZATION.search(path.read_text(encoding="utf-8")),
            f"{path.relative_to(root)}: pickle-family serialization is banned",
        )


def check_evidence(root: Path, report: Report, document: str, *, minimum: int) -> None:
    text = _read(root, document)
    workflow = _read(root, ".github/workflows/ci.yml")
    references = list(_EVIDENCE.finditer(text))
    report.require(
        len(references) >= minimum, f"{document}: expected at least {minimum} evidence references"
    )
    for match in references:
        file_ref, test_name, workflow_ref, job = match.groups()
        if file_ref:
            path = root / file_ref
            report.require(path.is_file(), f"{document}: evidence file {file_ref} does not exist")
            if test_name and path.is_file():
                source = path.read_text(encoding="utf-8")
                report.require(
                    re.search(rf"^\s*(?:async\s+)?def {test_name}\(", source, re.M),
                    f"{document}: {file_ref}::{test_name} does not exist",
                )
        else:
            report.require(
                (root / workflow_ref).is_file()
                and re.search(rf"^  {re.escape(job)}:", workflow, re.M),
                f"{document}: CI job {workflow_ref}#{job} does not exist",
            )


def check_threat_model(root: Path, report: Report) -> None:
    threat_model = _read(root, "docs/THREAT_MODEL.md")
    for category in STRIDE:
        report.require(
            category.lower() in threat_model.lower(),
            f"THREAT_MODEL.md: STRIDE category '{category}' is missing",
        )
    report.require(
        re.search(r"^\|.*Control.*\|.*Test.*\|", threat_model, re.M),
        "THREAT_MODEL.md: a threat -> control -> test table is required",
    )


def run_checks(root: Path) -> Report:
    report = Report()
    kind, package = check_pyproject(root, report)
    check_files(root, report, kind, package)
    check_readme(root, report)
    check_ci(root, report)
    check_containers(root, report)
    check_hygiene(root, report, package)
    check_threat_model(root, report)
    check_evidence(root, report, "CLAIMS.md", minimum=1)
    check_evidence(root, report, "docs/THREAT_MODEL.md", minimum=1)
    return report


def check(root: Path) -> list[str]:
    """Return the list of baseline failures for the repository at ``root``."""
    return run_checks(root).failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    report = run_checks(args.root.resolve())
    if report.failures:
        print(f"baseline check FAILED ({len(report.failures)} of {report.checks} checks):")
        for failure in report.failures:
            print(f"  - {failure}")
        return 1
    print(f"baseline check OK ({report.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
