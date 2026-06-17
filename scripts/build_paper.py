#!/usr/bin/env python3
"""Build paper PDF from generated figures and LaTeX tables."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIGURES_SRC = ROOT / "figures"
FIGURES_DST = PAPER_DIR / "figures"


def sync_paper_assets(figures_src: Path = FIGURES_SRC, figures_dst: Path = FIGURES_DST) -> list[Path]:
    """Copy generated PDF figures into paper/figures/. Returns copied paths."""
    figures_dst.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    if figures_src.is_dir():
        for pdf in sorted(figures_src.glob("*.pdf")):
            dest = figures_dst / pdf.name
            shutil.copy2(pdf, dest)
            copied.append(dest)
    return copied


def build_paper(paper_dir: Path = PAPER_DIR, latex_cmd: str = "pdflatex") -> Path:
    """Run pdflatex/bibtex cycle. Returns path to main.pdf."""
    main_tex = paper_dir / "main.tex"
    if not main_tex.is_file():
        raise FileNotFoundError(f"missing paper entry point: {main_tex}")

    def _run_pdflatex() -> None:
        subprocess.run(
            [latex_cmd, "-interaction=nonstopmode", "main.tex"],
            cwd=paper_dir,
            check=False,
        )

    for _ in range(2):
        _run_pdflatex()

    bib = paper_dir / "main.bbl"
    if not bib.exists():
        subprocess.run(["bibtex", "main"], cwd=paper_dir, check=False)
    _run_pdflatex()

    pdf = paper_dir / "main.pdf"
    if not pdf.is_file():
        raise RuntimeError("pdflatex did not produce main.pdf")
    return pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync assets and build paper/main.pdf")
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only copy figures into paper/figures/",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Sync assets without running pdflatex",
    )
    args = parser.parse_args()

    copied = sync_paper_assets()
    print(f"Synced {len(copied)} figure(s) to {FIGURES_DST}")

    if args.sync_only or args.skip_build:
        return 0

    try:
        pdf = build_paper()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Paper build failed: {exc}", file=sys.stderr)
        return 1

    print(f"Paper built: {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
