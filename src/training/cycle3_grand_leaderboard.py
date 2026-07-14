"""Cycle v3 image grand leaderboard synthesis from curated publication metrics (exp_112)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "config" / "cycle3_grand_leaderboard_registry.yaml"
REQUIRED_EXPERIMENTS = tuple(
    ["exp_101", "exp_102", "exp_103", "exp_104", "exp_105", "exp_105b"]
    + [f"exp_{i:03d}" for i in range(106, 112)]
)


@dataclass(frozen=True)
class Cycle3LeaderboardRow:
    experiment: str
    arm: str
    family: str
    quantum: bool
    metric: str
    primary: float
    reference: float
    delta: float
    verdict: str
    notes: str


@dataclass(frozen=True)
class Cycle3LeaderboardResult:
    rows: tuple[Cycle3LeaderboardRow, ...]
    claim_win_clip: float
    claim_win_fid: float
    expected_accepts: frozenset[str]
    observed_accepts: frozenset[str]
    quantum_false_claims: tuple[str, ...]
    coverage_ok: bool
    quantum_honesty_ok: bool
    accepts_ok: bool
    hypothesis_confirmed: bool


def load_cycle3_grand_leaderboard_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "rows" not in payload:
        msg = f"invalid cycle3 grand leaderboard registry: {path}"
        raise ValueError(msg)
    return payload


def _parse_rows(registry: dict[str, Any]) -> tuple[Cycle3LeaderboardRow, ...]:
    rows: list[Cycle3LeaderboardRow] = []
    for raw in registry.get("rows", []):
        rows.append(
            Cycle3LeaderboardRow(
                experiment=str(raw["experiment"]),
                arm=str(raw["arm"]),
                family=str(raw.get("family", "unknown")),
                quantum=bool(raw.get("quantum", False)),
                metric=str(raw["metric"]),
                primary=float(raw["primary"]),
                reference=float(raw["reference"]),
                delta=float(raw["delta"]),
                verdict=str(raw["verdict"]).lower(),
                notes=str(raw.get("notes", "")),
            )
        )
    return tuple(rows)


def _meets_quantum_claim(
    row: Cycle3LeaderboardRow,
    *,
    claim_win_clip: float,
    claim_win_fid: float,
) -> bool:
    if not row.quantum:
        return False
    if row.metric == "clip_score":
        return row.delta >= claim_win_clip
    if row.metric == "fid_r18":
        return row.delta <= claim_win_fid
    return False


def build_cycle3_grand_leaderboard(
    registry: dict[str, Any],
    *,
    claim_win_clip: float | None = None,
    claim_win_fid: float | None = None,
) -> Cycle3LeaderboardResult:
    rows = _parse_rows(registry)
    clip_thr = float(
        claim_win_clip if claim_win_clip is not None else registry.get("claim_win_clip", 0.5)
    )
    fid_thr = float(
        claim_win_fid if claim_win_fid is not None else registry.get("claim_win_fid", -2.0)
    )
    expected = frozenset(str(x) for x in registry.get("expected_accepts", []))
    observed = frozenset(r.experiment for r in rows if r.verdict == "accepted")
    present = {r.experiment for r in rows}
    coverage_ok = all(exp_id in present for exp_id in REQUIRED_EXPERIMENTS)
    # Honesty: rejected quantum arms must not secretly clear claim thresholds.
    quantum_false_claims = tuple(
        sorted(
            r.experiment
            for r in rows
            if r.verdict == "rejected"
            and _meets_quantum_claim(r, claim_win_clip=clip_thr, claim_win_fid=fid_thr)
        )
    )
    quantum_honesty_ok = len(quantum_false_claims) == 0
    accepts_ok = observed == expected
    return Cycle3LeaderboardResult(
        rows=rows,
        claim_win_clip=clip_thr,
        claim_win_fid=fid_thr,
        expected_accepts=expected,
        observed_accepts=observed,
        quantum_false_claims=quantum_false_claims,
        coverage_ok=coverage_ok,
        quantum_honesty_ok=quantum_honesty_ok,
        accepts_ok=accepts_ok,
        hypothesis_confirmed=coverage_ok and quantum_honesty_ok and accepts_ok,
    )


def cycle3_leaderboard_to_dict(
    result: Cycle3LeaderboardResult,
    registry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bench_id": "cycle3_grand_leaderboard",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": registry.get("hardware"),
        "cycle": registry.get("cycle", "v3"),
        "claim_win_clip": result.claim_win_clip,
        "claim_win_fid": result.claim_win_fid,
        "hypothesis": (
            "Cycle v3 scorecard: accepts match curated set; "
            "no rejected quantum arm clears CLIP/+FID claim thresholds"
        ),
        "hypothesis_confirmed": result.hypothesis_confirmed,
        "coverage_ok": result.coverage_ok,
        "quantum_honesty_ok": result.quantum_honesty_ok,
        "accepts_ok": result.accepts_ok,
        "expected_accepts": sorted(result.expected_accepts),
        "observed_accepts": sorted(result.observed_accepts),
        "quantum_false_claims": list(result.quantum_false_claims),
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
                "delta": r.delta,
                "verdict": r.verdict,
                "notes": r.notes,
            }
            for r in result.rows
        ],
    }


def export_cycle3_grand_leaderboard_json(payload: dict[str, Any], out_path: Path) -> Path:
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


def export_cycle3_grand_leaderboard_latex(
    result: Cycle3LeaderboardResult,
    out_path: Path,
) -> Path:
    lines = [
        "% Auto-generated Cycle v3 grand leaderboard (exp_112)",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Research Cycle v3 image grand leaderboard "
        "($\\Delta$ = primary $-$ reference; RTX~4060).}",
        "\\label{tab:cycle3_grand_leaderboard}",
        "\\begin{tabular}{llrrl}",
        "\\toprule",
        "Exp & Arm & Primary & $\\Delta$ & Verdict \\\\",
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
                    f"{row.primary:.2f}",
                    _format_delta(row.delta),
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
