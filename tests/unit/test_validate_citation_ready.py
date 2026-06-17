"""Unit tests for citation-loop readiness validation."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_citation_ready import (
    CITATION_ARTIFACTS,
    collect_version_mismatches,
    validate_citation_ready,
)


def test_collect_version_mismatches_empty_when_aligned():
    root = Path(__file__).resolve().parents[2]
    expected = _pyproject_version(root)
    assert collect_version_mismatches(root=root, expected_version=expected) == []


def _pyproject_version(root: Path) -> str:
    for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("missing version")


def test_collect_version_mismatches_detects_drift(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('version = "9.9.9"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("version: 0.1.0\n", encoding="utf-8")
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "arxiv_metadata.yaml").write_text('software_version: "0.1.0"\n', encoding="utf-8")
    (tmp_path / "src" / "presentation" / "http").mkdir(parents=True)
    (tmp_path / "src" / "presentation" / "http" / "app.py").write_text(
        'version="0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "RELEASE_NOTES.md").write_text("# v0.1.0\n", encoding="utf-8")
    mismatches = collect_version_mismatches(root=tmp_path, expected_version="0.1.0")
    assert any("pyproject" in m for m in mismatches)


def test_validate_citation_ready_passes_in_repo():
    ok, issues = validate_citation_ready(skip_release=True)
    assert ok, issues


def test_citation_artifacts_exist_in_repo():
    root = Path(__file__).resolve().parents[2]
    for rel in CITATION_ARTIFACTS:
        assert (root / rel).is_file(), f"missing citation artifact: {rel}"
