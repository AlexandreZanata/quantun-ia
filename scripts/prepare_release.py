#!/usr/bin/env python3
"""Bundle release artifacts for GitHub release and Zenodo archival."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = ROOT / "dist" / "release"
RELEASE_VERSION = "0.9.22"
PUBLICATION_JSONL = ROOT / "tests" / "contracts" / "fixtures" / "publication_experiments.jsonl"
STATIC_ARTIFACTS = ("AUTHORS.md", "CITATION.cff", "RELEASE_NOTES.md", "CHANGELOG.md", "SECURITY.md")
RELEASE_DOCS = (
    "docs/api.md",
    "docs/arxiv.md",
    "docs/citation_loop.md",
    "docs/releases/v0.9.16.md",
    "docs/compute_environment.md",
    "docs/ethics.md",
    "docs/method_adaptive_lr.md",
    "docs/microqml_bench.md",
    "docs/paper_narrative.md",
    "docs/reviewer_guide.md",
    "docs/reproducibility.md",
    "docs/zenodo.md",
)
RELEASE_RESULTS = (
    "experiments/exp_021_qml_backend_parity/results.md",
    "experiments/exp_022_nano_quantum_parity/results.md",
    "experiments/exp_023_encoding_backend/results.md",
    "experiments/exp_024_quantum_nano_bc/results.md",
)
RELEASE_MODEL_CARDS = ("model_cards/quantum_nano_bc.md",)
RELEASE_LEADERBOARD = ("v1.json", "meta.json")


def _seed_publication_logs() -> None:
    """Use publication fixture so Zenodo bundle matches paper/leaderboard numbers."""
    if not PUBLICATION_JSONL.is_file():
        raise FileNotFoundError(f"missing publication fixture: {PUBLICATION_JSONL}")
    logs_path = ROOT / "logs" / "experiments.jsonl"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PUBLICATION_JSONL, logs_path)


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
    _seed_publication_logs()
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

    ref_datasets_dir = dist_dir / "reference_datasets"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_reference_datasets.py"), "--out-dir", str(ref_datasets_dir)],
        check=True,
        cwd=ROOT,
    )
    artifacts.extend(sorted(ref_datasets_dir.glob("*.csv")))
    artifacts.extend(sorted(ref_datasets_dir.glob("*.meta.json")))

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

    reviewer_script = ROOT / "scripts" / "reviewer_repro.sh"
    if reviewer_script.exists():
        scripts_dir = dist_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        dest = scripts_dir / "reviewer_repro.sh"
        shutil.copy2(reviewer_script, dest)
        dest.chmod(dest.stat().st_mode | 0o111)
        artifacts.append(dest)

    docs_dir = dist_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for rel in RELEASE_DOCS:
        src = ROOT / rel
        if src.exists():
            dest = docs_dir / src.name
            shutil.copy2(src, dest)
            artifacts.append(dest)

    results_dir = dist_dir / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    for rel in RELEASE_RESULTS:
        src = ROOT / rel
        if src.exists():
            dest = results_dir / src.parent.name / "results.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            artifacts.append(dest)

    model_cards_dir = dist_dir / "model_cards"
    model_cards_dir.mkdir(parents=True, exist_ok=True)
    for rel in RELEASE_MODEL_CARDS:
        src = ROOT / rel
        if src.exists():
            dest = model_cards_dir / src.name
            shutil.copy2(src, dest)
            artifacts.append(dest)

    leaderboard_dir = dist_dir / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    for name in RELEASE_LEADERBOARD:
        src = ROOT / "docs" / "leaderboard" / name
        if src.exists():
            dest = leaderboard_dir / name
            shutil.copy2(src, dest)
            artifacts.append(dest)

    microqml_script = ROOT / "scripts" / "export_microqml_bench.py"
    if microqml_script.exists():
        bench_dir = dist_dir / "microqml_bench"
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_json = bench_dir / "v1.json"
        subprocess.run(
            [
                sys.executable,
                str(microqml_script),
                "--output",
                str(bench_json),
            ],
            check=True,
            cwd=ROOT,
            env={**os.environ, "MLFLOW_DISABLE": "1"},
        )
        if bench_json.exists():
            artifacts.append(bench_json)

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
