"""Unit tests for arXiv submission bundle preparation."""

from pathlib import Path

from scripts.prepare_arxiv_submission import (
    ARXIV_README,
    build_readme_text,
    collect_paper_sources,
    prepare_arxiv_submission,
)


def test_collect_paper_sources_includes_core_files(tmp_path: Path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(r"\documentclass{article}", encoding="utf-8")
    (paper / "references.bib").write_text("@misc{test}", encoding="utf-8")
    sections = paper / "sections"
    sections.mkdir()
    (sections / "intro.tex").write_text(r"\section{Intro}", encoding="utf-8")
    tables = paper / "tables"
    tables.mkdir()
    (tables / "summary.tex").write_text(r"\begin{table}", encoding="utf-8")

    sources = collect_paper_sources(paper_dir=paper)
    rel_names = {p.relative_to(paper).as_posix() for p in sources}
    assert "main.tex" in rel_names
    assert "references.bib" in rel_names
    assert "sections/intro.tex" in rel_names
    assert "tables/summary.tex" in rel_names


def test_build_readme_text_mentions_pdflatex():
    text = build_readme_text()
    assert "pdflatex" in text
    assert ARXIV_README in text


def test_prepare_arxiv_submission_creates_bundle(tmp_path: Path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(r"\documentclass{article}\begin{document}Hi\end{document}", encoding="utf-8")
    (paper / "references.bib").write_text("", encoding="utf-8")
    (paper / "arxiv_metadata.yaml").write_text("title: test\n", encoding="utf-8")
    dist = tmp_path / "dist" / "arxiv"

    artifacts = prepare_arxiv_submission(paper_dir=paper, dist_dir=dist, skip_pdf_check=True)

    assert (dist / ARXIV_README).is_file()
    assert (dist / "main.tex").is_file()
    assert (dist / "arxiv_metadata.yaml").is_file()
    tarball = dist / "quantun-ia-paper.tar.gz"
    assert tarball.is_file()
    assert len(artifacts) >= 3
