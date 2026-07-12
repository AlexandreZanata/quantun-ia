"""CLI — export Research Cycle v2 LaTeX tables into paper/tables/."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.cycle_v2_tables import DEFAULT_OUT, DEFAULT_REGISTRY, export_cycle_v2_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    created = export_cycle_v2_tables(registry_path=args.registry, out_dir=args.out)
    print(f"Exported {len(created)} Cycle v2 LaTeX table(s):")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
