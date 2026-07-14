#!/usr/bin/env python3
"""Publish MicroQML Bench v1 JSON for GitHub Pages (public leaderboard)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema

from tests.contracts.microqml_bench_schema import MICROQML_BENCH_V1_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = ROOT / "tests" / "contracts" / "fixtures" / "publication_experiments.jsonl"
DEFAULT_OUT_DIR = ROOT / "docs" / "leaderboard"
PUBLIC_JSON_URL = "https://alexandrezanata.github.io/quantun-ia/leaderboard/v1.json"
PUBLIC_VIEWER_URL = "https://alexandrezanata.github.io/quantun-ia/leaderboard/"


def load_records_from_jsonl(jsonl_path: Path) -> list[dict]:
    """Load experiment records from a JSONL file."""
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_meta(*, row_count: int) -> dict[str, str | int]:
    """Sidecar metadata for external consumers and CI."""
    return {
        "bench_id": "microqml_bench",
        "schema_version": "1",
        "format": "microqml_bench_v1",
        "canonical_json_url": PUBLIC_JSON_URL,
        "viewer_url": PUBLIC_VIEWER_URL,
        "leaderboard_rows": row_count,
        "source_fixture": "tests/contracts/fixtures/publication_experiments.jsonl",
    }


def publish_leaderboard(
    *,
    jsonl_path: Path = DEFAULT_JSONL,
    output_dir: Path = DEFAULT_OUT_DIR,
) -> tuple[Path, Path]:
    """Build and write public leaderboard JSON + meta sidecar."""
    # Lazy import: verify-only must stay torch-free for the Pages workflow.
    from src.benchmark.microqml_bench import build_bench_export, write_bench_export

    records = load_records_from_jsonl(jsonl_path)
    export = build_bench_export(records=records)
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_bench_export(export, output_dir / "v1.json")
    meta_path = output_dir / "meta.json"
    meta_path.write_text(
        json.dumps(build_meta(row_count=len(export["leaderboard"])), indent=2) + "\n",
        encoding="utf-8",
    )
    return json_path, meta_path


def verify_published_leaderboard(output_dir: Path = DEFAULT_OUT_DIR) -> bool:
    """Validate committed public leaderboard artifacts."""
    json_path = output_dir / "v1.json"
    meta_path = output_dir / "meta.json"
    index_path = output_dir / "index.html"
    if not json_path.is_file() or not meta_path.is_file() or not index_path.is_file():
        return False
    export = json.loads(json_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("leaderboard_rows") != len(export["leaderboard"]):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish MicroQML Bench public leaderboard")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Source JSONL (default: publication fixture)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for GitHub Pages",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing docs/leaderboard artifacts",
    )
    args = parser.parse_args()

    if args.verify_only:
        ok = verify_published_leaderboard(args.output_dir)
        print(f"Public leaderboard verification: {'OK' if ok else 'FAIL'}")
        return 0 if ok else 1

    json_path, meta_path = publish_leaderboard(jsonl_path=args.jsonl, output_dir=args.output_dir)
    print(f"Published {json_path} ({json.loads(json_path.read_text())['leaderboard'].__len__()} rows)")
    print(f"Metadata → {meta_path}")
    print(f"Viewer: {PUBLIC_VIEWER_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
