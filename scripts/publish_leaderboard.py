#!/usr/bin/env python3
"""Publish MicroQML Bench + Cycle v2/v3 scorecards for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from tests.contracts.microqml_bench_schema import MICROQML_BENCH_V1_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = ROOT / "tests" / "contracts" / "fixtures" / "publication_experiments.jsonl"
DEFAULT_OUT_DIR = ROOT / "docs" / "leaderboard"
PUBLIC_JSON_URL = "https://alexandrezanata.github.io/quantun-ia/leaderboard/v1.json"
PUBLIC_VIEWER_URL = "https://alexandrezanata.github.io/quantun-ia/leaderboard/"
CYCLE2_SRC = ROOT / "dist" / "leaderboards" / "cycle2_grand_leaderboard.json"
CYCLE3_SRC = ROOT / "dist" / "leaderboards" / "cycle3_grand_leaderboard.json"
CYCLE2_REGISTRY = ROOT / "config" / "cycle2_grand_leaderboard_registry.yaml"
CYCLE3_REGISTRY = ROOT / "config" / "cycle3_grand_leaderboard_registry.yaml"


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


def _ensure_cycle_exports() -> tuple[Path, Path]:
    """Build cycle2/cycle3 JSON into dist/ if missing, return source paths."""
    if not CYCLE2_SRC.is_file():
        from src.training.cycle2_grand_leaderboard import (
            build_cycle2_grand_leaderboard,
            cycle2_leaderboard_to_dict,
            export_cycle2_grand_leaderboard_json,
            load_cycle2_grand_leaderboard_registry,
        )

        registry = load_cycle2_grand_leaderboard_registry(CYCLE2_REGISTRY)
        matrix = build_cycle2_grand_leaderboard(registry)
        export_cycle2_grand_leaderboard_json(
            cycle2_leaderboard_to_dict(matrix, registry), CYCLE2_SRC
        )
    if not CYCLE3_SRC.is_file():
        from src.training.cycle3_grand_leaderboard import (
            build_cycle3_grand_leaderboard,
            cycle3_leaderboard_to_dict,
            export_cycle3_grand_leaderboard_json,
            load_cycle3_grand_leaderboard_registry,
        )

        registry = load_cycle3_grand_leaderboard_registry(CYCLE3_REGISTRY)
        matrix = build_cycle3_grand_leaderboard(registry)
        export_cycle3_grand_leaderboard_json(
            cycle3_leaderboard_to_dict(matrix, registry), CYCLE3_SRC
        )
    return CYCLE2_SRC, CYCLE3_SRC


def build_meta(
    *,
    row_count: int,
    cycle2: dict,
    cycle3: dict,
) -> dict:
    """Sidecar metadata for external consumers and CI."""
    return {
        "bench_id": "microqml_bench",
        "schema_version": "1",
        "format": "microqml_bench_v1",
        "canonical_json_url": PUBLIC_JSON_URL,
        "viewer_url": PUBLIC_VIEWER_URL,
        "leaderboard_rows": row_count,
        "source_fixture": "tests/contracts/fixtures/publication_experiments.jsonl",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "cycles": {
            "v1_microqml": {"json": "v1.json", "rows": row_count},
            "v2_agro": {
                "json": "cycle2.json",
                "accepted": cycle2.get("observed_accepts", []),
                "n_rows": cycle2.get("n_rows"),
                "hypothesis_confirmed": cycle2.get("hypothesis_confirmed"),
            },
            "v3_image": {
                "json": "cycle3.json",
                "accepted": cycle3.get("observed_accepts", []),
                "n_rows": cycle3.get("n_rows"),
                "hypothesis_confirmed": cycle3.get("hypothesis_confirmed"),
            },
        },
    }


def publish_leaderboard(
    *,
    jsonl_path: Path = DEFAULT_JSONL,
    output_dir: Path = DEFAULT_OUT_DIR,
) -> tuple[Path, Path]:
    """Build and write public leaderboard JSON + meta sidecar + cycle scorecards."""
    # Lazy import: verify-only must stay torch-free for the Pages workflow.
    from src.benchmark.microqml_bench import build_bench_export, write_bench_export

    records = load_records_from_jsonl(jsonl_path)
    export = build_bench_export(records=records)
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_bench_export(export, output_dir / "v1.json")

    cycle2_src, cycle3_src = _ensure_cycle_exports()
    cycle2_out = output_dir / "cycle2.json"
    cycle3_out = output_dir / "cycle3.json"
    shutil.copyfile(cycle2_src, cycle2_out)
    shutil.copyfile(cycle3_src, cycle3_out)
    cycle2 = json.loads(cycle2_out.read_text(encoding="utf-8"))
    cycle3 = json.loads(cycle3_out.read_text(encoding="utf-8"))

    meta_path = output_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            build_meta(
                row_count=len(export["leaderboard"]),
                cycle2=cycle2,
                cycle3=cycle3,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path, meta_path


def verify_published_leaderboard(output_dir: Path = DEFAULT_OUT_DIR) -> bool:
    """Validate committed public leaderboard artifacts."""
    json_path = output_dir / "v1.json"
    meta_path = output_dir / "meta.json"
    index_path = output_dir / "index.html"
    cycle2_path = output_dir / "cycle2.json"
    cycle3_path = output_dir / "cycle3.json"
    required = [json_path, meta_path, index_path, cycle2_path, cycle3_path]
    if not all(p.is_file() for p in required):
        return False
    export = json.loads(json_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=export, schema=MICROQML_BENCH_V1_SCHEMA)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("leaderboard_rows") != len(export["leaderboard"]):
        return False
    cycle2 = json.loads(cycle2_path.read_text(encoding="utf-8"))
    cycle3 = json.loads(cycle3_path.read_text(encoding="utf-8"))
    if cycle2.get("bench_id") != "cycle2_grand_leaderboard":
        return False
    if cycle3.get("bench_id") != "cycle3_grand_leaderboard":
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
    print(f"Cycle scorecards → {(args.output_dir / 'cycle2.json')}, {(args.output_dir / 'cycle3.json')}")
    print(f"Viewer: {PUBLIC_VIEWER_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
