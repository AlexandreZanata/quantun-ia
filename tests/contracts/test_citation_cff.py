"""Contract tests for CITATION.cff metadata."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

CITATION_PATH = Path("CITATION.cff")
PYPROJECT_PATH = Path("pyproject.toml")
DOI_PATTERN = re.compile(r"^10\.5281/zenodo\.\d+$")


def _load_citation() -> dict:
    text = CITATION_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _pyproject_version() -> str:
    for line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("pyproject.toml missing version")


def test_citation_file_exists():
    assert CITATION_PATH.is_file()


def test_citation_required_fields():
    data = _load_citation()
    for field in ("cff-version", "title", "type", "version", "authors", "license"):
        assert field in data, f"CITATION.cff missing {field}"


def test_citation_version_matches_pyproject():
    data = _load_citation()
    assert data["version"] == _pyproject_version()


def test_citation_authors_have_names():
    data = _load_citation()
    authors = data["authors"]
    assert isinstance(authors, list) and authors
    for author in authors:
        assert author.get("family-names") or author.get("given-names")


def test_citation_doi_format_when_present():
    data = _load_citation()
    doi = data.get("doi")
    if doi is None:
        pytest.skip("DOI added after Zenodo sync (see docs/zenodo.md)")
    assert DOI_PATTERN.match(str(doi)), f"invalid Zenodo DOI format: {doi}"
