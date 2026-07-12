"""Agro maize side-by-side comparison — HistGB / distill nano / quantum floors (E-T3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.application.agro_maize_predict import DEFAULT_MODEL_ID, predict_agro_maize
from src.application.human_agro_scorer import AgroMunicipalityProfile, HumanAgroScoreResult
from src.shared.result import Fail, Ok, Result, fail, ok

ROOT = Path(__file__).resolve().parents[2]
CYCLE_V2_REGISTRY = ROOT / "config" / "cycle_v2_paper.yaml"


@dataclass(frozen=True)
class PublishedFloorRow:
    name: str
    roc_auc: float
    notes: str
    source: str


@dataclass(frozen=True)
class AgroMaizeCompareBundle:
    live_nano: HumanAgroScoreResult
    model_id: str
    floors: tuple[PublishedFloorRow, ...]
    quantum_note: str


def load_maize_published_floors(path: Path = CYCLE_V2_REGISTRY) -> tuple[PublishedFloorRow, ...]:
    """Load HistGB / distill / quantum published AUCs for dashboard compare cards."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: list[PublishedFloorRow] = []
    for entry in payload.get("boosting_frontier", {}).get("rows", []):
        if "HistGradientBoosting" in str(entry.get("model", "")) or "distill" in str(
            entry.get("model", "")
        ).lower():
            rows.append(
                PublishedFloorRow(
                    name=str(entry["model"]),
                    roc_auc=float(entry["roc_auc"]),
                    notes=str(entry.get("notes", "")),
                    source=str(entry.get("source", "")),
                )
            )
    for entry in payload.get("quantum_v2", {}).get("rows", []):
        arm = str(entry.get("arm", ""))
        if "Residual-skip" in arm or "Plain 4q" in arm:
            rows.append(
                PublishedFloorRow(
                    name=f"{entry['experiment']} · {arm}",
                    roc_auc=float(entry["roc_auc"]),
                    notes=str(entry.get("verdict", "")),
                    source=str(entry.get("experiment", "")),
                )
            )
    return tuple(rows)


def compare_agro_maize(
    profile: AgroMunicipalityProfile,
    *,
    with_uncertainty: bool = True,
) -> Result[AgroMaizeCompareBundle, Any]:
    """Live distill nano score + published Cycle v2 floors for side-by-side UI."""
    outcome = predict_agro_maize(
        profile,
        log_prediction=False,
        with_uncertainty=with_uncertainty,
    )
    if isinstance(outcome, Fail):
        return fail(outcome.error)
    assert isinstance(outcome, Ok)
    floors = load_maize_published_floors()
    return ok(
        AgroMaizeCompareBundle(
            live_nano=outcome.value,
            model_id=DEFAULT_MODEL_ID,
            floors=floors,
            quantum_note=(
                "exp_086 residual QNN was parity-only (+0.07 pp vs plain); "
                "not a serve winner — floors shown from publication metrics."
            ),
        )
    )
