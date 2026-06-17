"""Contract tests for unified citation loop documentation (Phase 25)."""

from __future__ import annotations

from pathlib import Path

import yaml

LOOP_DOC = Path("docs/citation_loop.md")
ZENODO_DOC = Path("docs/zenodo.md")
ARXIV_DOC = Path("docs/arxiv.md")
NARRATIVE_DOC = Path("docs/paper_narrative.md")


def test_citation_loop_doc_exists():
    assert LOOP_DOC.is_file()


def test_citation_loop_links_zenodo_and_arxiv():
    text = LOOP_DOC.read_text(encoding="utf-8")
    assert "zenodo.md" in text
    assert "arxiv.md" in text
    assert "CITATION.cff" in text
    assert "arxiv_metadata.yaml" in text


def test_citation_loop_documents_manual_doi_step():
    text = LOOP_DOC.read_text(encoding="utf-8")
    assert "doi:" in text.lower()
    assert "git tag" in text


def test_paper_narrative_doc_exists():
    assert NARRATIVE_DOC.is_file()
    text = NARRATIVE_DOC.read_text(encoding="utf-8")
    assert "Option C" in text
    assert "exp_011" in text


def test_arxiv_metadata_ready_for_upload():
    data = yaml.safe_load(Path("paper/arxiv_metadata.yaml").read_text(encoding="utf-8"))
    assert data.get("title")
    assert data.get("abstract")
    # arxiv_id remains null until manual upload
    assert data.get("arxiv_id") in (None, "")
