"""Agro maize prediction use case — wraps C4b serve checkpoint for API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.application.human_agro_scorer import (
    AgroMunicipalityProfile,
    HumanAgroScoreError,
    HumanAgroScoreResult,
    score_municipality,
)
from src.shared.result import Fail, Ok, Result, fail, ok

ROOT = Path(__file__).resolve().parents[2]
MODEL_CARD_PATH = ROOT / "model_cards" / "large_nano_mlp_acyd_maize.md"

DEFAULT_EXP_ID = "exp_081"
DEFAULT_MODEL = "large_nano_mlp"
DEFAULT_DATASET = "acyd_maize_brazil_v1"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class AgroMaizeModelCard:
    model_id: str
    dataset_id: str
    exp_id: str
    seed: int
    markdown: str


def load_agro_maize_model_card() -> AgroMaizeModelCard:
    if not MODEL_CARD_PATH.is_file():
        msg = f"model card missing: {MODEL_CARD_PATH}"
        raise FileNotFoundError(msg)
    return AgroMaizeModelCard(
        model_id="large_nano_mlp_acyd_maize",
        dataset_id=DEFAULT_DATASET,
        exp_id=DEFAULT_EXP_ID,
        seed=DEFAULT_SEED,
        markdown=MODEL_CARD_PATH.read_text(encoding="utf-8"),
    )


def predict_agro_maize(
    profile: AgroMunicipalityProfile,
    *,
    log_prediction: bool = True,
) -> Result[HumanAgroScoreResult, HumanAgroScoreError]:
    """Score maize low-yield probability for one municipality profile."""
    outcome = score_municipality(
        profile,
        exp_id=DEFAULT_EXP_ID,
        model_name=DEFAULT_MODEL,
        dataset=DEFAULT_DATASET,
        seed=DEFAULT_SEED,
        log_prediction=log_prediction,
        root=ROOT,
        log_domain="agro_maize",
    )
    if isinstance(outcome, Fail):
        return fail(outcome.error)
    assert isinstance(outcome, Ok)
    return ok(outcome.value)
