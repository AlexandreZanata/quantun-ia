"""Unit tests for citation finalization after Zenodo/arXiv moderation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.finalize_citation import (
    ARXIV_ID_PATTERN,
    DOI_PATTERN,
    apply_arxiv_id,
    apply_bib_doi,
    apply_citation_doi,
    finalize_citation,
)


def test_doi_pattern_accepts_zenodo():
    assert DOI_PATTERN.match("10.5281/zenodo.1234567")


def test_arxiv_pattern_accepts_standard_id():
    assert ARXIV_ID_PATTERN.match("2606.12345")


def test_apply_citation_doi_inserts_field():
    original = "version: 0.9.16\ndate-released: 2026-06-17\n# doi: placeholder\n"
    updated = apply_citation_doi(original, "10.5281/zenodo.1234567")
    assert "doi: 10.5281/zenodo.1234567" in updated
    assert "# doi:" not in updated


def test_apply_arxiv_id_updates_yaml():
    original = "arxiv_id: null\n"
    updated = apply_arxiv_id(original, "2606.12345")
    data = yaml.safe_load(updated)
    assert data["arxiv_id"] == "2606.12345"


def test_apply_bib_doi_adds_doi_field():
    original = "@misc{quantunia2026,\n  title={test},\n  note={pending},\n}\n"
    updated = apply_bib_doi(original, "10.5281/zenodo.99", "0.9.16")
    assert "doi = {10.5281/zenodo.99}" in updated
    assert "v0.9.16" in updated


def test_finalize_citation_writes_files(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('version = "0.9.16"\n', encoding="utf-8")
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "cff-version: 1.2.0\nversion: 0.9.16\ndate-released: 2026-06-17\n",
        encoding="utf-8",
    )
    arxiv = tmp_path / "paper" / "arxiv_metadata.yaml"
    arxiv.parent.mkdir(parents=True)
    arxiv.write_text("software_version: '0.9.16'\narxiv_id: null\n", encoding="utf-8")
    bib = tmp_path / "paper" / "references.bib"
    bib.write_text("@misc{quantunia2026,\n  title={t},\n}\n", encoding="utf-8")

    paths = finalize_citation(
        doi="10.5281/zenodo.1234567",
        arxiv_id="2606.12345",
        root=tmp_path,
    )
    assert len(paths) == 3
    assert "doi: 10.5281/zenodo.1234567" in citation.read_text(encoding="utf-8")
    meta = yaml.safe_load(arxiv.read_text(encoding="utf-8"))
    assert meta["arxiv_id"] == "2606.12345"


def test_finalize_citation_rejects_invalid_doi():
    with pytest.raises(ValueError, match="invalid Zenodo DOI"):
        apply_citation_doi("version: 0.9.16\n", "not-a-doi")
