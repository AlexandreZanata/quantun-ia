"""Contract tests for paper/arxiv_metadata.yaml submission metadata."""

from __future__ import annotations

from pathlib import Path

import yaml

METADATA_PATH = Path("paper/arxiv_metadata.yaml")
PYPROJECT_PATH = Path("pyproject.toml")
ALLOWED_CATEGORIES = frozenset({"cs.LG", "quant-ph", "cs.AI", "stat.ML"})


def _load_metadata() -> dict:
    assert METADATA_PATH.is_file(), "paper/arxiv_metadata.yaml is required for arXiv submission"
    return yaml.safe_load(METADATA_PATH.read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    for line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("pyproject.toml missing version")


def test_arxiv_metadata_required_fields():
    data = _load_metadata()
    for field in ("title", "abstract", "authors", "categories", "comments", "software_version"):
        assert field in data, f"arxiv_metadata.yaml missing {field}"


def test_arxiv_metadata_software_version_matches_pyproject():
    data = _load_metadata()
    assert data["software_version"] == _pyproject_version()


def test_arxiv_metadata_categories_valid():
    data = _load_metadata()
    categories = data["categories"]
    assert isinstance(categories, list) and categories
    for cat in categories:
        assert cat in ALLOWED_CATEGORIES, f"unexpected arXiv category: {cat}"


def test_arxiv_metadata_authors_non_empty():
    data = _load_metadata()
    authors = data["authors"]
    assert isinstance(authors, list) and authors
    for author in authors:
        assert author.get("name"), "each author needs a name"


def test_arxiv_metadata_abstract_length():
    data = _load_metadata()
    abstract = data["abstract"].strip()
    assert 100 <= len(abstract) <= 1920, "arXiv abstract should be 100–1920 characters"


def test_arxiv_id_format_when_present():
    data = _load_metadata()
    arxiv_id = data.get("arxiv_id")
    if not arxiv_id:
        return
    assert str(arxiv_id).count(".") == 1, f"invalid arXiv ID format: {arxiv_id}"
