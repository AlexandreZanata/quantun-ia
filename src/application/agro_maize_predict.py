"""Agro maize prediction use case — distill ResidualNano + MC-dropout uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from src.application.human_agro_scorer import (
    AgroMunicipalityProfile,
    HumanAgroScoreError,
    HumanAgroScoreResult,
    score_municipality,
)
from src.application.mc_dropout_uncertainty import mc_dropout_predict
from src.application.model_registry import build_model
from src.data.scaling import transform_with_scaler
from src.shared.result import Fail, Ok, Result, fail, ok
from src.training.checkpoints import load_checkpoint_bundle, load_scaler
from src.training.device import resolve_device

ROOT = Path(__file__).resolve().parents[2]
MODEL_CARD_PATH = ROOT / "model_cards" / "residual_nano_distill_acyd_maize.md"

DEFAULT_EXP_ID = "exp_092"
DEFAULT_MODEL = "residual_nano_distill"
DEFAULT_DATASET = "acyd_maize_brazil_v1"
DEFAULT_SEED = 42
DEFAULT_MODEL_ID = "residual_nano_distill_acyd_maize"
DEFAULT_MC_SAMPLES = 20


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
        model_id=DEFAULT_MODEL_ID,
        dataset_id=DEFAULT_DATASET,
        exp_id=DEFAULT_EXP_ID,
        seed=DEFAULT_SEED,
        markdown=MODEL_CARD_PATH.read_text(encoding="utf-8"),
    )


def _attach_mc_dropout(
    score: HumanAgroScoreResult,
    *,
    n_samples: int = DEFAULT_MC_SAMPLES,
) -> HumanAgroScoreResult:
    try:
        bundle = load_checkpoint_bundle(
            DEFAULT_EXP_ID,
            DEFAULT_MODEL,
            DEFAULT_DATASET,
            seed=DEFAULT_SEED,
        )
        scaler = load_scaler(bundle.directory)
        model, _ = build_model(DEFAULT_MODEL, input_dim=len(score.feature_vector))
        model.load_state_dict(bundle.state_dict)
        device = resolve_device(None, model=model)
        model = model.to(device)
        raw = np.asarray([score.feature_vector], dtype=np.float32)
        scaled = transform_with_scaler(scaler, raw)
        x = torch.tensor(scaled, dtype=torch.float32)
        unc = mc_dropout_predict(model, x, n_samples=n_samples, seed=DEFAULT_SEED)
    except FileNotFoundError:
        return score
    return replace(
        score,
        uncertainty_std=unc.std_probability,
        uncertainty_method=unc.method,
        mc_mean_probability=unc.mean_probability,
    )


def predict_agro_maize(
    profile: AgroMunicipalityProfile,
    *,
    log_prediction: bool = True,
    with_uncertainty: bool = True,
    mc_samples: int = DEFAULT_MC_SAMPLES,
) -> Result[HumanAgroScoreResult, HumanAgroScoreError]:
    """Score maize low-yield probability; optionally attach MC-dropout std."""
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
    score = outcome.value
    if with_uncertainty:
        score = _attach_mc_dropout(score, n_samples=mc_samples)
    return ok(score)
