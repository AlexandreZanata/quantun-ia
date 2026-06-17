#!/usr/bin/env python3
"""Validate version alignment and artifacts before Zenodo tag and arXiv upload."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from scripts.prepare_release import DEFAULT_DIST, RELEASE_VERSION, verify_manifest

ROOT = Path(__file__).resolve().parents[1]

CITATION_ARTIFACTS = (
    "CITATION.cff",
    "RELEASE_NOTES.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "docs/zenodo.md",
    "docs/arxiv.md",
    "docs/citation_loop.md",
    "docs/paper_narrative.md",
    "paper/arxiv_metadata.yaml",
)


def _pyproject_version(root: Path) -> str:
    for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise ValueError("pyproject.toml missing version")


def collect_version_mismatches(root: Path, expected_version: str) -> list[str]:
    """Return human-readable mismatches for version fields across citation artifacts."""
    issues: list[str] = []

    pyproject_ver = _pyproject_version(root)
    if pyproject_ver != expected_version:
        issues.append(f"pyproject.toml version {pyproject_ver} != expected {expected_version}")

    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    if str(citation.get("version")) != expected_version:
        issues.append(f"CITATION.cff version {citation.get('version')} != {expected_version}")

    arxiv_meta = yaml.safe_load((root / "paper" / "arxiv_metadata.yaml").read_text(encoding="utf-8"))
    if str(arxiv_meta.get("software_version")) != expected_version:
        issues.append(
            f"arxiv_metadata software_version {arxiv_meta.get('software_version')} != {expected_version}"
        )

    app_text = (root / "src" / "presentation" / "http" / "app.py").read_text(encoding="utf-8")
    match = re.search(r'version="([^"]+)"', app_text)
    if not match or match.group(1) != expected_version:
        found = match.group(1) if match else "missing"
        issues.append(f"app.py version {found} != {expected_version}")

    if RELEASE_VERSION != expected_version:
        issues.append(f"prepare_release RELEASE_VERSION {RELEASE_VERSION} != {expected_version}")

    notes = (root / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    if f"v{expected_version}" not in notes:
        issues.append(f"RELEASE_NOTES.md missing v{expected_version}")

    return issues


def validate_citation_ready(
    root: Path = ROOT,
    dist_dir: Path = DEFAULT_DIST,
    *,
    skip_release: bool = False,
) -> tuple[bool, list[str]]:
    """Check citation artifacts and optional release manifest. Returns (ok, issues)."""
    issues: list[str] = []
    expected = _pyproject_version(root)

    for rel in CITATION_ARTIFACTS:
        if not (root / rel).is_file():
            issues.append(f"missing artifact: {rel}")

    issues.extend(collect_version_mismatches(root, expected))

    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    if citation.get("doi") is None:
        issues.append("DOI not set in CITATION.cff (expected until Zenodo sync — informational)")

    arxiv_meta = yaml.safe_load((root / "paper" / "arxiv_metadata.yaml").read_text(encoding="utf-8"))
    if arxiv_meta.get("arxiv_id"):
        if str(arxiv_meta["arxiv_id"]).count(".") != 1:
            issues.append(f"invalid arxiv_id format: {arxiv_meta['arxiv_id']}")
    else:
        issues.append("arxiv_id not set (expected until arXiv upload — informational)")

    if not skip_release:
        manifest = dist_dir / "MANIFEST.txt"
        if not manifest.is_file():
            issues.append(f"release bundle missing — run `make release` ({dist_dir})")
        elif not verify_manifest(manifest, dist_dir):
            issues.append("MANIFEST.txt checksum verification failed")

    blocking = [i for i in issues if "informational" not in i]
    return len(blocking) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate citation-loop readiness")
    parser.add_argument(
        "--skip-release",
        action="store_true",
        help="Do not require dist/release MANIFEST.txt",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=DEFAULT_DIST,
        help="Release bundle directory",
    )
    args = parser.parse_args()

    ok, issues = validate_citation_ready(dist_dir=args.dist, skip_release=args.skip_release)
    for item in issues:
        prefix = "INFO" if "informational" in item else "ERROR"
        print(f"{prefix}: {item}")

    if ok:
        print(f"Citation loop ready for tag v{_pyproject_version(ROOT)} (paste DOI + arXiv ID after upload)")
        print("See docs/citation_loop.md")
        return 0

    print("Citation loop validation failed — fix blocking errors above", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
