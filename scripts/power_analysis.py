#!/usr/bin/env python3
"""Minimum detectable effect (MDE) tables for multi-seed holdout experiments."""

from __future__ import annotations

import argparse

from src.training.effect_size import minimum_detectable_effect


def power_table(
    *,
    n_seeds: list[int] | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> list[tuple[int, float]]:
    """Return (n_pairs, mde_cohens_d) rows."""
    seeds = n_seeds or list(range(2, 21))
    return [(n, minimum_detectable_effect(n, alpha=alpha, power=power)) for n in seeds]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print minimum detectable Cohen's d for paired holdout comparisons",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Publication profile seed count (default: 10)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print MDE for n=2..20 instead of a single n",
    )
    args = parser.parse_args(argv)

    if args.table:
        print(f"Minimum detectable Cohen's d (paired, alpha={args.alpha}, power={args.power})")
        print("| Seeds | MDE (|d|) |")
        print("|-------|-----------|")
        for n, mde in power_table(alpha=args.alpha, power=args.power):
            mde_txt = "∞" if mde == float("inf") else f"{mde:.2f}"
            print(f"| {n} | {mde_txt} |")
        return 0

    mde = minimum_detectable_effect(args.n_seeds, alpha=args.alpha, power=args.power)
    print(f"n_pairs={args.n_seeds}  MDE (|Cohen's d|)={mde:.3f}")
    print(f"Comparisons with |d| below {mde:.2f} may be underpowered at power={args.power}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
