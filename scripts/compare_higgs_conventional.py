#!/usr/bin/env python3
"""Thin CLI wrapper around EXP 058 conventional baseline comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_058_conventional_higgs_baselines.run import run_exp_058


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare LargeNanoMLP vs conventional sklearn/XGBoost on HIGGS (EXP 058)",
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_exp_058(profile=args.profile, verbose=not args.json)
    if args.json:
        payload = {
            "profile": result.profile,
            "nano_auc": result.nano_auc,
            "best_conventional_auc": result.best_conventional_auc,
            "advantage_pp": result.advantage_vs_best_conventional_pp,
            "scores": [s.__dict__ for s in result.scores],
        }
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
