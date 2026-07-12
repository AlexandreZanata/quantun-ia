"""Cycle v2 grand leaderboard synthesis from curated publication metrics (exp_100)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "config" / "cycle2_grand_leaderboard_registry.yaml"
REQUIRED_EXPERIMENTS = tuple(f"exp_{i:03d}" for i in range(84, 100))


@dataclass(frozen=True)
class Cycle2LeaderboardRow:
    experiment: str
    arm: str
    family: str
    quantum: bool
    metric: str
    primary: float
    reference: float
    delta_pp: float
    verdict: str
    notes: str


@dataclass(frozen=True)
class Cycle2LeaderboardResult:
    rows: tuple[Cycle2LeaderboardRow, ...]
    claim_win_delta_pp: float
    expected_accepts: frozenset[str]
    observed_accepts: frozenset[str]
    quantum_claim_wins: tuple[str, ...]
    coverage_ok: bool
    quantum_honesty_ok: bool
    accepts_ok: bool
    hypothesis_confirmed: bool


def load_cycle2_grand_leaderboard_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "rows" not in payload:
        msg = f"invalid cycle2 grand leaderboard registry: {path}"
        raise ValueError(msg)
    return payload


def _parse_rows(registry: dict[str, Any]) -> tuple[Cycle2LeaderboardRow, ...]:
    rows: list[Cycle2LeaderboardRow] = []
    for raw in registry.get("rows", []):
        rows.append(
            Cycle2LeaderboardRow(
                experiment=str(raw["experiment"]),
                arm=str(raw["arm"]),
                family=str(raw.get("family", "unknown")),
                quantum=bool(raw.get("quantum", False)),
                metric=str(raw["metric"]),
                primary=float(raw["primary"]),
                reference=float(raw["reference"]),
                delta_pp=float(raw["delta_pp"]),
                verdict=str(raw["verdict"]).lower(),
                notes=str(raw.get("notes", "")),
            )
        )
    return tuple(rows)


def build_cycle2_grand_leaderboard(
    registry: dict[str, Any],
    *,
    claim_win_delta_pp: float | None = None,
) -> Cycle2LeaderboardResult:
    rows = _parse_rows(registry)
    win_delta = float(
        claim_win_delta_pp
        if claim_win_delta_pp is not None
        else registry.get("claim_win_delta_pp", 0.5)
    )
    expected = frozenset(str(x) for x in registry.get("expected_accepts", []))
    observed = frozenset(r.experiment for r in rows if r.verdict == "accepted")
    present = {r.experiment for r in rows}
    coverage_ok = all(exp_id in present for exp_id in REQUIRED_EXPERIMENTS)
    quantum_claim_wins = tuple(
        sorted(r.experiment for r in rows if r.quantum and r.delta_pp >= win_delta)
    )
    quantum_honesty_ok = len(quantum_claim_wins) == 0
    accepts_ok = observed == expected
    return Cycle2LeaderboardResult(
        rows=rows,
        claim_win_delta_pp=win_delta,
        expected_accepts=expected,
        observed_accepts=observed,
        quantum_claim_wins=quantum_claim_wins,
        coverage_ok=coverage_ok,
        quantum_honesty_ok=quantum_honesty_ok,
        accepts_ok=accepts_ok,
        hypothesis_confirmed=coverage_ok and quantum_honesty_ok and accepts_ok,
    )


def cycle2_leaderboard_to_dict(
    result: Cycle2LeaderboardResult,
    registry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bench_id": "cycle2_grand_leaderboard",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": registry.get("hardware"),
        "cycle": registry.get("cycle", "v2"),
        "claim_win_delta_pp": result.claim_win_delta_pp,
        "hypothesis": (
            "No Cycle v2 quantum recipe beats classical floor by >= +0.5 pp; "
            "accepts match curated set"
        ),
        "hypothesis_confirmed": result.hypothesis_confirmed,
        "coverage_ok": result.coverage_ok,
        "quantum_honesty_ok": result.quantum_honesty_ok,
        "accepts_ok": result.accepts_ok,
        "expected_accepts": sorted(result.expected_accepts),
        "observed_accepts": sorted(result.observed_accepts),
        "quantum_claim_wins": list(result.quantum_claim_wins),
        "n_rows": len(result.rows),
        "n_accepted": sum(1 for r in result.rows if r.verdict == "accepted"),
        "n_rejected": sum(1 for r in result.rows if r.verdict == "rejected"),
        "rows": [
            {
                "experiment": r.experiment,
                "arm": r.arm,
                "family": r.family,
                "quantum": r.quantum,
                "metric": r.metric,
                "primary": r.primary,
                "reference": r.reference,
                "delta_pp": r.delta_pp,
                "verdict": r.verdict,
                "notes": r.notes,
            }
            for r in result.rows
        ],
    }


def export_cycle2_grand_leaderboard_json(payload: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}"


def export_cycle2_grand_leaderboard_latex(
    result: Cycle2LeaderboardResult,
    out_path: Path,
) -> Path:
    lines = [
        "% Auto-generated Cycle v2 grand leaderboard (exp_100)",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Research Cycle v2 grand leaderboard ($\\Delta$ pp vs registered floor; RTX~4060).}",
        "\\label{tab:cycle2_grand_leaderboard}",
        "\\begin{tabular}{llrrl}",
        "\\toprule",
        "Exp & Arm & Primary & $\\Delta$ pp & Verdict \\\\",
        "\\midrule",
    ]
    for row in result.rows:
        arm = _latex_escape(row.arm)
        if len(arm) > 42:
            arm = arm[:39] + "..."
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.experiment),
                    arm,
                    f"{row.primary:.4f}",
                    _format_delta(row.delta_pp),
                    row.verdict,
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
