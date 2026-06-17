#!/usr/bin/env python3
"""Bundle release artifacts for GitHub release and Zenodo archival."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = ROOT / "dist" / "release"


def prepare_release(dist_dir: Path = DEFAULT_DIST) -> list[Path]:
    """Export CSV, figures, LaTeX tables into dist/release/. Returns artifact paths."""
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True)

    artifacts: list[Path] = []

    # CSV export
    csv_out = dist_dir / "results.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_results.py")],
        check=True,
        cwd=ROOT,
    )
    src_csv = ROOT / "data" / "exports" / "results.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, csv_out)
        artifacts.append(csv_out)

    # Figures
    figures_dir = dist_dir / "figures"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_figures.py"), "--out", str(figures_dir)],
        check=True,
        cwd=ROOT,
    )
    artifacts.extend(sorted(figures_dir.glob("*.pdf")))

    # LaTeX tables
    tables_dir = dist_dir / "tables"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_latex_tables.py"),
            "--out",
            str(tables_dir),
        ],
        check=True,
        cwd=ROOT,
    )
    artifacts.extend(sorted(tables_dir.glob("*.tex")))

    # Publication large summary (static doc)
    summary = ROOT / "docs" / "publication_large_summary.md"
    if summary.exists():
        dest = dist_dir / "publication_large_summary.md"
        shutil.copy2(summary, dest)
        artifacts.append(dest)

    # Lock file for reproducibility
    lock = ROOT / "requirements.lock"
    if lock.exists():
        dest = dist_dir / "requirements.lock"
        shutil.copy2(lock, dest)
        artifacts.append(dest)

    manifest = dist_dir / "MANIFEST.txt"
    manifest.write_text(
        "\n".join(str(p.relative_to(dist_dir)) for p in sorted(artifacts)) + "\n",
        encoding="utf-8",
    )
    artifacts.append(manifest)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare v0.4.0 release bundle for Zenodo")
    parser.add_argument(
        "--dist",
        type=Path,
        default=DEFAULT_DIST,
        help="Output directory (default: dist/release)",
    )
    args = parser.parse_args()

    try:
        artifacts = prepare_release(dist_dir=args.dist)
    except subprocess.CalledProcessError as exc:
        print(f"Release preparation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Release bundle ready at {args.dist} ({len(artifacts)} artifacts)")
    print("Next steps:")
    print("  1. git tag v0.4.0 && git push origin v0.4.0")
    print("  2. Create GitHub release from tag (attach dist/release/*)")
    print("  3. Enable Zenodo-GitHub integration → copy DOI to CITATION.cff")
    print("  See docs/zenodo.md for full instructions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
