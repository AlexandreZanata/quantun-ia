#!/usr/bin/env python3
"""Build Synthea cardiovascular risk open dataset — FHIR or clinical simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.synthea_cv_risk import (
    build_synthea_processed,
    run_synthea_cli,
    update_synthea_manifest_ready,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "open" / "synthea_cv_risk" / "processed" / "v1"
DEFAULT_MANIFEST = ROOT / "data" / "open" / "manifest.json"
DEFAULT_RAW = ROOT / "data" / "open" / "synthea_cv_risk" / "raw"
DEFAULT_JAR = DEFAULT_RAW / "synthea-with-dependencies.jar"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Synthea CV risk dataset (Phase L3)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Processed output directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest.json to update when build completes",
    )
    parser.add_argument(
        "--fhir-dir",
        type=Path,
        default=None,
        help="Existing Synthea FHIR directory (default: clinical simulation)",
    )
    parser.add_argument(
        "--run-synthea",
        action="store_true",
        help="Run Synthea JAR before extraction (requires --synthea-jar)",
    )
    parser.add_argument(
        "--synthea-jar",
        type=Path,
        default=DEFAULT_JAR,
        help="Path to synthea-with-dependencies.jar",
    )
    parser.add_argument(
        "--synthea-output",
        type=Path,
        default=DEFAULT_RAW / "generated",
        help="Directory for Synthea CLI output",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=120_000,
        help="Synthea population when --run-synthea (subsampled to 1M rows)",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not update manifest.json",
    )
    args = parser.parse_args()

    fhir_dir = args.fhir_dir
    if args.run_synthea:
        print(f"Running Synthea JAR population={args.population} → {args.synthea_output}")
        fhir_dir = run_synthea_cli(
            args.synthea_jar,
            args.synthea_output,
            population=args.population,
        )

    mode = "FHIR extraction" if fhir_dir else "clinical simulation"
    print(f"Building Synthea CV risk ({mode}) → {args.out}")
    paths, source_mode = build_synthea_processed(args.out, fhir_dir=fhir_dir)
    for name, path in paths.items():
        print(f"  wrote {name}: {path} (source_mode={source_mode})")

    if not args.skip_manifest:
        update_synthea_manifest_ready(args.manifest, args.out)
        print(f"Updated {args.manifest} (synthea_cv_risk_v1 ready=true)")

    print("Synthea CV risk build complete.")
    print("Next: dvc add data/open/synthea_cv_risk/processed/v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
