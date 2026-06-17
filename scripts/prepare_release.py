#!/usr/bin/env python3
"""Bundle release artifacts for GitHub release and Zenodo archival."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = ROOT / "dist" / "release"
RELEASE_VERSION = "0.9.7"
STATIC_ARTIFACTS = ("CITATION.cff", "RELEASE_NOTES.md", "CHANGELOG.md")


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_text(artifacts: list[Path], base_dir: Path) -> str:
    """Build MANIFEST.txt body with relative paths and SHA-256 checksums."""
    lines: list[str] = []
    for path in sorted(artifacts, key=lambda p: str(p.relative_to(base_dir))):
        rel = path.relative_to(base_dir)
        lines.append(f"{rel} sha256:{sha256_file(path)}")
    return "\n".join(lines) + "\n"


def verify_manifest(manifest_path: Path, base_dir: Path) -> bool:
    """Verify every manifest entry exists and matches its checksum."""
    if not manifest_path.is_file():
        return False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(" sha256:", maxsplit=1)
        if len(parts) != 2:
            return False
        rel_path, expected = parts
        target = base_dir / rel_path
        if not target.is_file() or sha256_file(target) != expected:
            return False
    return True


def _copy_static_artifacts(dist_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for name in STATIC_ARTIFACTS:
        src = ROOT / name
        if src.exists():
            dest = dist_dir / name
            shutil.copy2(src, dest)
            copied.append(dest)
    return copied


def prepare_release(dist_dir: Path = DEFAULT_DIST) -> list[Path]:
    """Export CSV, figures, LaTeX tables into dist/release/. Returns artifact paths."""
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True)

    artifacts: list[Path] = []

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

    figures_dir = dist_dir / "figures"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_figures.py"), "--out", str(figures_dir)],
        check=True,
        cwd=ROOT,
    )
    artifacts.extend(sorted(figures_dir.glob("*.pdf")))

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

    summary = ROOT / "docs" / "publication_large_summary.md"
    if summary.exists():
        dest = dist_dir / "publication_large_summary.md"
        shutil.copy2(summary, dest)
        artifacts.append(dest)

    lock = ROOT / "requirements.lock"
    if lock.exists():
        dest = dist_dir / "requirements.lock"
        shutil.copy2(lock, dest)
        artifacts.append(dest)

    api_doc = ROOT / "docs" / "api.md"
    if api_doc.exists():
        docs_dir = dist_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        dest = docs_dir / "api.md"
        shutil.copy2(api_doc, dest)
        artifacts.append(dest)

    artifacts.extend(_copy_static_artifacts(dist_dir))

    manifest = dist_dir / "MANIFEST.txt"
    manifest.write_text(build_manifest_text(artifacts, dist_dir), encoding="utf-8")
    artifacts.append(manifest)

    if not verify_manifest(manifest, dist_dir):
        raise RuntimeError("Release manifest verification failed")

    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Prepare v{RELEASE_VERSION} release bundle for Zenodo",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=DEFAULT_DIST,
        help="Output directory (default: dist/release)",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="Verify an existing bundle MANIFEST.txt instead of rebuilding",
    )
    args = parser.parse_args()

    if args.verify_only is not None:
        manifest = args.verify_only / "MANIFEST.txt"
        ok = verify_manifest(manifest, args.verify_only)
        print(f"Manifest verification: {'OK' if ok else 'FAIL'} ({manifest})")
        return 0 if ok else 1

    try:
        artifacts = prepare_release(dist_dir=args.dist)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"Release preparation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Release bundle ready at {args.dist} ({len(artifacts)} artifacts)")
    print("Next steps:")
    print(f"  1. git tag v{RELEASE_VERSION} && git push origin v{RELEASE_VERSION}")
    print("  2. GitHub Actions uploads dist/release/* to the release")
    print("  3. Enable Zenodo-GitHub integration → copy DOI to CITATION.cff")
    print("  See docs/zenodo.md for full instructions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
