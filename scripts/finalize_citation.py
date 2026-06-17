#!/usr/bin/env python3
"""Apply Zenodo DOI and arXiv ID to citation artifacts after moderation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DOI_PATTERN = re.compile(r"^10\.5281/zenodo\.\d+$")
ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")

CITATION_PATH = ROOT / "CITATION.cff"
ARXIV_META_PATH = ROOT / "paper" / "arxiv_metadata.yaml"
REFERENCES_BIB_PATH = ROOT / "paper" / "references.bib"
QUANTUNIA_BIB_KEY = "quantunia2026"


def apply_citation_doi(content: str, doi: str) -> str:
    """Insert or replace doi field and remove placeholder comments."""
    if not DOI_PATTERN.match(doi):
        raise ValueError(f"invalid Zenodo DOI format: {doi}")

    content = re.sub(r"^# doi:.*\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"^# After Zenodo.*\n", "", content, flags=re.MULTILINE)
    if re.search(r"^doi:", content, flags=re.MULTILINE):
        content = re.sub(r"^doi:.*$", f"doi: {doi}", content, flags=re.MULTILINE)
    elif re.search(r"^date-released:", content, flags=re.MULTILINE):
        content = re.sub(
            r"^(date-released: .+)$",
            rf"\1\ndoi: {doi}",
            content,
            flags=re.MULTILINE,
        )
    else:
        content = content.rstrip() + f"\ndoi: {doi}\n"
    return content


def apply_arxiv_id(content: str, arxiv_id: str) -> str:
    """Set arxiv_id in paper/arxiv_metadata.yaml."""
    if not ARXIV_ID_PATTERN.match(arxiv_id):
        raise ValueError(f"invalid arXiv ID format: {arxiv_id}")

    data = yaml.safe_load(content) or {}
    data["arxiv_id"] = arxiv_id
    header_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("#"):
            header_lines.append(line)
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if header_lines:
        return "\n".join(header_lines) + "\n\n" + body
    return body


def apply_bib_doi(content: str, doi: str, version: str) -> str:
    """Update quantunia2026 bib entry with Zenodo DOI."""
    if QUANTUNIA_BIB_KEY not in content:
        raise ValueError(f"missing @{QUANTUNIA_BIB_KEY} entry in references.bib")

    block_pattern = re.compile(
        rf"@misc\{{{QUANTUNIA_BIB_KEY},.*?\}}\n",
        re.DOTALL,
    )
    match = block_pattern.search(content)
    if not match:
        raise ValueError(f"could not parse @{QUANTUNIA_BIB_KEY} block")

    block = match.group(0)
    block = re.sub(r"^\s*doi\s*=\s*\{.*\},?\n", "", block, flags=re.MULTILINE)
    block = re.sub(
        r"^\s*note\s*=\s*\{.*\},?\n",
        f"  note={{Software v{version}; Zenodo archive}},\n",
        block,
        flags=re.MULTILINE,
    )
    if "doi = {" not in block:
        block = block.replace(
            f"@misc{{{QUANTUNIA_BIB_KEY},",
            f"@misc{{{QUANTUNIA_BIB_KEY},\n  doi = {{{doi}}},",
        )
    else:
        block = re.sub(r"doi = \{.*\}", f"doi = {{{doi}}}", block)

    return content[: match.start()] + block + content[match.end() :]


def finalize_citation(
    doi: str,
    *,
    arxiv_id: str | None = None,
    root: Path = ROOT,
    version: str | None = None,
) -> list[Path]:
    """Write DOI (and optional arXiv ID) into citation files. Returns updated paths."""
    citation_path = root / "CITATION.cff"
    arxiv_path = root / "paper" / "arxiv_metadata.yaml"
    bib_path = root / "paper" / "references.bib"

    if version is None:
        for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                version = line.split("=", 1)[1].strip().strip('"')
                break
        if version is None:
            raise ValueError("pyproject.toml missing version")

    updated: list[Path] = []

    citation_text = apply_citation_doi(citation_path.read_text(encoding="utf-8"), doi)
    citation_path.write_text(citation_text, encoding="utf-8")
    updated.append(citation_path)

    bib_text = apply_bib_doi(bib_path.read_text(encoding="utf-8"), doi, version)
    bib_path.write_text(bib_text, encoding="utf-8")
    updated.append(bib_path)

    if arxiv_id:
        arxiv_text = apply_arxiv_id(arxiv_path.read_text(encoding="utf-8"), arxiv_id)
        arxiv_path.write_text(arxiv_text, encoding="utf-8")
        updated.append(arxiv_path)

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Zenodo DOI and optional arXiv ID")
    parser.add_argument("--doi", required=True, help="Zenodo DOI, e.g. 10.5281/zenodo.1234567")
    parser.add_argument("--arxiv-id", help="Moderated arXiv ID, e.g. 2606.12345")
    args = parser.parse_args()

    try:
        paths = finalize_citation(doi=args.doi, arxiv_id=args.arxiv_id)
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    for path in paths:
        print(f"updated {path.relative_to(ROOT)}")
    print("Run: pytest tests/contracts/test_citation_cff.py tests/contracts/test_arxiv_metadata.py -v")
    print("Run: make citation-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
