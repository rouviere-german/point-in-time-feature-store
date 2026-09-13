from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration


def load_checker(repo_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_baseline", repo_root / "scripts" / "check_baseline.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_copy(repo_root: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "repo"
    shutil.copytree(
        repo_root,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "var",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".hypothesis",
            "htmlcov",
            ".coverage*",
            "coverage.xml",
        ),
    )
    return destination


def failures(repo_root: Path, root: Path) -> list[str]:
    result: list[str] = load_checker(repo_root).check(root)
    return result


def test_template_passes_its_own_baseline(repo_root: Path) -> None:
    assert failures(repo_root, repo_root) == []


def test_missing_file_fails(repo_root: Path, repo_copy: Path) -> None:
    (repo_copy / "SECURITY.md").unlink()
    assert any("SECURITY.md" in item for item in failures(repo_root, repo_copy))


def test_missing_readme_section_fails(repo_root: Path, repo_copy: Path) -> None:
    readme = repo_copy / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("## Non-goals", "## Out of scope"),
        encoding="utf-8",
    )
    assert any("Non-goals" in item for item in failures(repo_root, repo_copy))


def test_missing_ci_scanner_fails(repo_root: Path, repo_copy: Path) -> None:
    workflow = repo_copy / ".github" / "workflows" / "ci.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace("trivy", "scanner"), encoding="utf-8"
    )
    assert any("trivy" in item for item in failures(repo_root, repo_copy))


def test_dangling_claim_reference_fails(repo_root: Path, repo_copy: Path) -> None:
    claims = repo_copy / "CLAIMS.md"
    claims.write_text(
        claims.read_text(encoding="utf-8")
        + "\n| Made-up claim | `tests/unit/test_redaction.py::test_does_not_exist` |\n",
        encoding="utf-8",
    )
    assert any("test_does_not_exist" in item for item in failures(repo_root, repo_copy))


def test_unpinned_base_image_fails(repo_root: Path, repo_copy: Path) -> None:
    dockerfile = repo_copy / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    dockerfile.write_text(text.replace("@sha256:", "-unpinned-"), encoding="utf-8")
    assert any("digest" in item for item in failures(repo_root, repo_copy))


def test_instantiated_repo_must_rename_the_package(repo_root: Path, repo_copy: Path) -> None:
    pyproject = repo_copy / "pyproject.toml"
    if 'kind = "template"' not in pyproject.read_text(encoding="utf-8"):
        pytest.skip("only meaningful in the template itself")
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('kind = "template"', 'kind = "agentic"'),
        encoding="utf-8",
    )
    assert any("rename" in item for item in failures(repo_root, repo_copy))


@pytest.mark.parametrize(
    "source",
    [
        "import _pickle\n",
        "import pandas as pd\n\nframe = pd.read_pickle('model.pkl')\n",
        "def save(frame, path):\n    frame.to_pickle(path)\n",
    ],
)
def test_pickle_family_serialization_in_source_fails(
    repo_root: Path, repo_copy: Path, source: str
) -> None:
    package = next(path.parent for path in (repo_copy / "src").glob("*/__init__.py"))
    (package / "artifact_io.py").write_text(source, encoding="utf-8")
    assert any("pickle" in item for item in failures(repo_root, repo_copy))


def test_main_exit_codes(
    repo_root: Path, repo_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checker = load_checker(repo_root)
    assert checker.main(["--root", str(repo_root)]) == 0
    (repo_copy / "LICENSE").unlink()
    assert checker.main(["--root", str(repo_copy)]) == 1
    assert "LICENSE" in capsys.readouterr().out
