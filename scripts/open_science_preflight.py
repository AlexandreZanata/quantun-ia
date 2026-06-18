#!/usr/bin/env python3
"""One-command open-science preflight before Zenodo tag and arXiv upload."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_JSONL = ROOT / "tests" / "contracts" / "fixtures" / "publication_experiments.jsonl"
LOGS_JSONL = ROOT / "logs" / "experiments.jsonl"
RELEASE_DIST = ROOT / "dist" / "release"
ARXIV_DIST = ROOT / "dist" / "arxiv"


def seed_publication_logs(*, force: bool = False) -> Path:
    """Copy publication fixture into logs/ for reproducible release exports."""
    if not PUBLICATION_JSONL.is_file():
        raise FileNotFoundError(f"missing publication fixture: {PUBLICATION_JSONL}")
    LOGS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if LOGS_JSONL.exists() and not force:
        return LOGS_JSONL
    shutil.copy2(PUBLICATION_JSONL, LOGS_JSONL)
    return LOGS_JSONL


def _run_step(name: str, command: list[str]) -> None:
    print(f"→ {name}")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {result.returncode})")


def open_science_preflight(
    *,
    skip_arxiv: bool = False,
    skip_release: bool = False,
    force_logs: bool = False,
) -> None:
    """Run release + citation + leaderboard checks for Zenodo/arXiv readiness."""
    python = sys.executable
    seed_publication_logs(force=force_logs)

    if not skip_release:
        _run_step("release bundle", [python, str(ROOT / "scripts" / "prepare_release.py")])
        _run_step(
            "release manifest",
            [python, str(ROOT / "scripts" / "prepare_release.py"), "--verify-only", str(RELEASE_DIST)],
        )

    _run_step("public leaderboard", [python, str(ROOT / "scripts" / "publish_leaderboard.py"), "--verify-only"])
    _run_step(
        "citation artifacts",
        [python, str(ROOT / "scripts" / "validate_citation_ready.py"), "--skip-release"],
    )

    if not skip_arxiv:
        _run_step(
            "arxiv sources",
            [
                python,
                str(ROOT / "scripts" / "prepare_arxiv_submission.py"),
                "--skip-pdf-check",
            ],
        )

    required_release_paths = [
        RELEASE_DIST / "microqml_bench" / "v1.json",
        RELEASE_DIST / "model_cards" / "quantum_nano_bc.md",
        RELEASE_DIST / "leaderboard" / "v1.json",
        RELEASE_DIST / "experiments" / "exp_024_quantum_nano_bc" / "results.md",
        ROOT / "AUTHORS.md",
    ]
    if not skip_release:
        missing = [str(p.relative_to(ROOT)) for p in required_release_paths if not p.is_file()]
        if missing:
            raise RuntimeError(f"release bundle missing artifacts: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Open-science preflight (Zenodo + arXiv readiness)")
    parser.add_argument("--skip-arxiv", action="store_true", help="Skip arXiv source bundle step")
    parser.add_argument("--skip-release", action="store_true", help="Skip release rebuild")
    parser.add_argument("--force-logs", action="store_true", help="Overwrite logs/experiments.jsonl with fixture")
    args = parser.parse_args()

    try:
        open_science_preflight(
            skip_arxiv=args.skip_arxiv,
            skip_release=args.skip_release,
            force_logs=args.force_logs,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Open-science preflight failed: {exc}", file=sys.stderr)
        return 1

    print("Open-science preflight complete.")
    print("Manual steps: git tag v* → Zenodo DOI → make finalize-citation DOI=...")
    print("See docs/citation_loop.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
