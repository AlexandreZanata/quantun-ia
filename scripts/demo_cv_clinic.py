#!/usr/bin/env python3
"""CLI demo — human-readable CV risk scoring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.human_cv_scorer import PatientProfile, compare_profiles, score_patient
from src.shared.result import Fail, Ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score human patient profiles with exp_034")
    parser.add_argument("--age", type=float, default=55)
    parser.add_argument("--male", action="store_true")
    parser.add_argument("--bmi", type=float, default=28)
    parser.add_argument("--sbp", type=float, default=128, help="Systolic BP mmHg")
    parser.add_argument("--smoker", action="store_true")
    parser.add_argument("--diabetes", action="store_true")
    parser.add_argument("--prior-mi", action="store_true")
    parser.add_argument("--game", action="store_true", help="Run easy A vs B challenge")
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    os.environ.setdefault("QML_DEVICE", "cuda")

    if args.game:
        a = PatientProfile(age_years=32, sex_male=True, bmi=24, systolic_bp=118)
        b = PatientProfile(
            age_years=78, sex_male=True, bmi=31, systolic_bp=165,
            smoker=True, diabetes=True, prior_mi=True,
        )
        outcome = compare_profiles(a, b)
        if isinstance(outcome, Fail):
            print(outcome.error.message, file=sys.stderr)
            return 1
        assert isinstance(outcome, Ok)
        ra, rb = outcome.value
        print(f"Patient A (young):  {ra.risk_percent:.1f}%")
        print(f"Patient B (elderly): {rb.risk_percent:.1f}%")
        winner = "A" if ra.probability > rb.probability else "B"
        print(f"Higher risk: Patient {winner}")
        return 0

    profile = PatientProfile(
        age_years=args.age,
        sex_male=args.male,
        bmi=args.bmi,
        systolic_bp=args.sbp,
        smoker=args.smoker,
        diabetes=args.diabetes,
        prior_mi=args.prior_mi,
    )
    outcome = score_patient(profile)
    if isinstance(outcome, Fail):
        print(outcome.error.message, file=sys.stderr)
        return 1
    assert isinstance(outcome, Ok)
    r = outcome.value
    print(f"Patient: {r.human_summary}")
    print(f"12-month CV risk: {r.risk_percent:.1f}% ({r.risk_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
