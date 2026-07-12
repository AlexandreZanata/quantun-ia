"""Human-readable agro-climate scoring — maps municipality profiles to ACYD model features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.application.dto import PredictNanomodelDTO
from src.application.predict_nanomodel import execute as predict_execute
from src.data.open_acyd import N_FEATURES
from src.shared.result import Fail, Ok, Result, fail, ok

DEFAULT_EXP_ID = "exp_060"
DEFAULT_MODEL = "large_nano_mlp"
DEFAULT_DATASET = "acyd_soy_brazil_v1"
DEFAULT_SEED = 42

AGRO_RESEARCH_DISCLAIMER = (
    "Research benchmark on ACYD Brazil soybean — not official ZARC / not insurance advice."
)

WEATHER_STAT_NAMES = (
    "precipitation",
    "t2m_min",
    "t2m_max",
    "solar_radiation",
    "lai_high",
    "ndvi",
    "vapor_pressure_deficit",
)


@dataclass(frozen=True)
class WeatherBlockStats:
    """In-season aggregate stats for one weather variable (weeks 10–40)."""

    mean: float
    std: float
    min_v: float
    max_v: float


@dataclass(frozen=True)
class AgroMunicipalityProfile:
    """Fields an agronomist can reason about — mapped to 37 ACYD features."""

    municipality: str
    state: str
    crop_year: int
    latitude: float
    longitude: float
    area_harvested_ha: float
    organic_carbon: float = 1.5
    ph_h2o: float = 5.5
    clay: float = 30.0
    sand: float = 40.0
    cec: float = 10.0
    bulk_density: float = 1.3
    precipitation: WeatherBlockStats = WeatherBlockStats(3.5, 1.2, 1.0, 8.0)
    t2m_min: WeatherBlockStats = WeatherBlockStats(288.0, 2.0, 284.0, 292.0)
    t2m_max: WeatherBlockStats = WeatherBlockStats(301.0, 2.5, 296.0, 306.0)
    solar_radiation: WeatherBlockStats = WeatherBlockStats(15_500_000.0, 1_500_000.0, 12_000_000.0, 18_000_000.0)
    lai_high: WeatherBlockStats = WeatherBlockStats(2.2, 0.4, 1.2, 3.5)
    ndvi: WeatherBlockStats = WeatherBlockStats(3.0, 0.5, 2.0, 4.2)
    vapor_pressure_deficit: WeatherBlockStats = WeatherBlockStats(12.0, 3.0, 6.0, 20.0)


@dataclass(frozen=True)
class HumanAgroScoreError:
    code: str
    message: str


@dataclass(frozen=True)
class ClimateDriver:
    name: str
    direction: str
    detail: str


@dataclass(frozen=True)
class HumanAgroScoreResult:
    probability: float
    risk_percent: float
    risk_tier: str
    risk_label: str
    human_summary: str
    feature_vector: list[float]
    checkpoint_path: str
    profile: AgroMunicipalityProfile
    top_drivers: tuple[ClimateDriver, ...]


def yield_risk_tier(probability: float) -> tuple[str, str]:
    """Return (tier_code, human_label) for P(low yield)."""
    pct = probability * 100.0
    if pct < 35.0:
        return "low", "Low yield risk"
    if pct < 65.0:
        return "moderate", "Moderate yield risk"
    return "high", "High yield risk"


def profile_to_features(profile: AgroMunicipalityProfile) -> list[float]:
    """Convert human agro profile to raw 37-dim ACYD feature vector."""
    vector: list[float] = [
        float(profile.latitude),
        float(profile.longitude),
        float(math.log1p(profile.area_harvested_ha)),
        float(profile.organic_carbon),
        float(profile.ph_h2o),
        float(profile.clay),
        float(profile.sand),
        float(profile.cec),
        float(profile.bulk_density),
    ]
    for block in (
        profile.precipitation,
        profile.t2m_min,
        profile.t2m_max,
        profile.solar_radiation,
        profile.lai_high,
        profile.ndvi,
        profile.vapor_pressure_deficit,
    ):
        vector.extend([block.mean, block.std, block.min_v, block.max_v])
    if len(vector) != N_FEATURES:
        msg = f"expected {N_FEATURES} features, built {len(vector)}"
        raise ValueError(msg)
    return vector


def profile_summary(profile: AgroMunicipalityProfile) -> str:
    return (
        f"{profile.municipality}, {profile.state} ({profile.crop_year}) — "
        f"{profile.area_harvested_ha:,.0f} ha harvested"
    )


def top_climate_drivers(profile: AgroMunicipalityProfile) -> tuple[ClimateDriver, ...]:
    """Heuristic top-3 drivers vs a neutral soybean-season baseline."""
    baseline = AgroMunicipalityProfile(
        municipality="baseline",
        state="BR",
        crop_year=profile.crop_year,
        latitude=profile.latitude,
        longitude=profile.longitude,
        area_harvested_ha=profile.area_harvested_ha,
    )
    candidates: list[tuple[float, ClimateDriver]] = []

    def _add(name: str, delta: float, direction: str, detail: str) -> None:
        candidates.append((abs(delta), ClimateDriver(name=name, direction=direction, detail=detail)))

    _add(
        "Season precipitation",
        profile.precipitation.mean - baseline.precipitation.mean,
        "below normal" if profile.precipitation.mean < baseline.precipitation.mean else "above normal",
        f"mean {profile.precipitation.mean:.2f} mm/week vs typical {baseline.precipitation.mean:.2f}",
    )
    _add(
        "Max temperature",
        profile.t2m_max.max_v - baseline.t2m_max.max_v,
        "heat stress" if profile.t2m_max.max_v > baseline.t2m_max.max_v else "cooler peak",
        f"peak {profile.t2m_max.max_v - 273.15:.1f} °C season max",
    )
    _add(
        "NDVI",
        baseline.ndvi.mean - profile.ndvi.mean,
        "vegetation stress" if profile.ndvi.mean < baseline.ndvi.mean else "healthy canopy",
        f"mean NDVI {profile.ndvi.mean:.2f}",
    )
    _add(
        "VPD",
        profile.vapor_pressure_deficit.mean - baseline.vapor_pressure_deficit.mean,
        "dry air" if profile.vapor_pressure_deficit.mean > baseline.vapor_pressure_deficit.mean else "humid",
        f"mean VPD {profile.vapor_pressure_deficit.mean:.1f}",
    )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return tuple(driver for _, driver in candidates[:3])


def append_prediction_log(
    payload: dict,
    *,
    root: Path | None = None,
) -> None:
    """Append one JSON line to logs/predictions.jsonl (gitignored)."""
    log_root = root or Path(__file__).resolve().parents[2]
    log_path = log_root / "logs" / "predictions.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now(UTC).isoformat(),
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def score_municipality(
    profile: AgroMunicipalityProfile,
    *,
    exp_id: str = DEFAULT_EXP_ID,
    model_name: str = DEFAULT_MODEL,
    dataset: str = DEFAULT_DATASET,
    seed: int = DEFAULT_SEED,
    log_prediction: bool = False,
    root: Path | None = None,
    log_domain: str = "agro_soy",
) -> Result[HumanAgroScoreResult, HumanAgroScoreError]:
    """Score one municipality profile through a LargeNanoMLP serve checkpoint."""
    features = profile_to_features(profile)
    outcome = predict_execute(
        PredictNanomodelDTO(
            exp_id=exp_id,
            model_name=model_name,
            dataset=dataset,
            seed=seed,
            features=[features],
        )
    )
    if isinstance(outcome, Fail):
        return fail(HumanAgroScoreError(outcome.error.code, outcome.error.message))

    assert isinstance(outcome, Ok)
    pred = outcome.value
    prob = pred.probabilities[0]
    tier_code, tier_label = yield_risk_tier(prob)
    drivers = top_climate_drivers(profile)

    if log_prediction:
        append_prediction_log(
            {
                "domain": log_domain,
                "model_id": f"{exp_id}/{model_name}",
                "dataset": dataset,
                "municipality": profile.municipality,
                "state": profile.state,
                "crop_year": profile.crop_year,
                "probability": round(prob, 6),
                "risk_tier": tier_code,
            },
            root=root,
        )

    return ok(
        HumanAgroScoreResult(
            probability=prob,
            risk_percent=prob * 100.0,
            risk_tier=tier_code,
            risk_label=tier_label,
            human_summary=profile_summary(profile),
            feature_vector=features,
            checkpoint_path=pred.checkpoint_path,
            profile=profile,
            top_drivers=drivers,
        )
    )
