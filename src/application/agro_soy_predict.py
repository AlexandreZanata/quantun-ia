"""Agro soybean prediction use case — wraps C4 serve checkpoint for API and dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.application.human_agro_scorer import (
    DEFAULT_DATASET,
    DEFAULT_EXP_ID,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    AgroMunicipalityProfile,
    HumanAgroScoreError,
    HumanAgroScoreResult,
    score_municipality,
)
from src.shared.result import Fail, Ok, Result, fail, ok

ROOT = Path(__file__).resolve().parents[2]
MODEL_CARD_PATH = ROOT / "model_cards" / "large_nano_mlp_acyd_soy.md"


@dataclass(frozen=True)
class AgroSoyModelCard:
    model_id: str
    dataset_id: str
    exp_id: str
    seed: int
    markdown: str


def load_agro_soy_model_card() -> AgroSoyModelCard:
    if not MODEL_CARD_PATH.is_file():
        msg = f"model card missing: {MODEL_CARD_PATH}"
        raise FileNotFoundError(msg)
    return AgroSoyModelCard(
        model_id="large_nano_mlp_acyd_soy",
        dataset_id=DEFAULT_DATASET,
        exp_id=DEFAULT_EXP_ID,
        seed=DEFAULT_SEED,
        markdown=MODEL_CARD_PATH.read_text(encoding="utf-8"),
    )


def predict_agro_soy(
    profile: AgroMunicipalityProfile,
    *,
    log_prediction: bool = True,
) -> Result[HumanAgroScoreResult, HumanAgroScoreError]:
    """Score soybean low-yield probability for one municipality profile."""
    outcome = score_municipality(
        profile,
        exp_id=DEFAULT_EXP_ID,
        model_name=DEFAULT_MODEL,
        dataset=DEFAULT_DATASET,
        seed=DEFAULT_SEED,
        log_prediction=log_prediction,
        root=ROOT,
    )
    if isinstance(outcome, Fail):
        return fail(outcome.error)
    assert isinstance(outcome, Ok)
    return ok(outcome.value)
