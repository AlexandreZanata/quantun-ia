"""Pydantic schemas for HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateTrainingJobRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    profile: str = "mini"
    epochs: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    exp_id: str = "nano_train"
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    async_mode: bool = False
    save_checkpoints: bool = False


class PredictRequest(BaseModel):
    exp_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    seed: int = Field(..., ge=0)
    features: list[list[float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    probabilities: list[float]
    labels: list[int]
    checkpoint_path: str


class AgroSoyPredictRequest(BaseModel):
    municipality: str = Field(..., min_length=1)
    state: str = Field(..., min_length=2, max_length=2)
    crop_year: int = Field(..., ge=2000, le=2035)
    latitude: float
    longitude: float
    area_harvested_ha: float = Field(..., gt=0)
    precip_mean: float = Field(default=3.5)
    tmax_peak_k: float = Field(default=304.0)
    ndvi_mean: float = Field(default=3.0)


class AgroSoyPredictResponse(BaseModel):
    probability: float
    risk_tier: str
    risk_label: str
    model_id: str
    crop_year: int
    municipality: str
    state: str
    top_drivers: list[dict[str, str]]


class AgroSoyModelCardResponse(BaseModel):
    model_id: str
    dataset_id: str
    exp_id: str
    seed: int
    markdown: str


class AgroMaizePredictRequest(BaseModel):
    municipality: str = Field(..., min_length=1)
    state: str = Field(..., min_length=2, max_length=2)
    crop_year: int = Field(..., ge=2000, le=2035)
    latitude: float
    longitude: float
    area_harvested_ha: float = Field(..., gt=0)
    precip_mean: float = Field(default=3.5)
    tmax_peak_k: float = Field(default=304.0)
    ndvi_mean: float = Field(default=3.0)


class AgroMaizePredictResponse(BaseModel):
    probability: float
    risk_tier: str
    risk_label: str
    model_id: str
    crop_year: int
    municipality: str
    state: str
    top_drivers: list[dict[str, str]]
    uncertainty_std: float | None = None
    uncertainty_method: str = "none"
    mc_mean_probability: float | None = None


class AgroMaizeModelCardResponse(BaseModel):
    model_id: str
    dataset_id: str
    exp_id: str
    seed: int
    markdown: str


class IssueTokenRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    user_id: str = "api-client"


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 900


class TrainingJobResponse(BaseModel):
    id: str
    tenant_id: str
    model_name: str
    dataset: str
    profile: str
    status: str
    exp_id: str
    seed: int | None
    epochs: int | None
    device: str
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    version: int


class LeaderboardRowResponse(BaseModel):
    exp_id: str
    model: str
    accuracy: float
    ci_low: float | None = None
    ci_high: float | None = None
    source: str | None = None
    elapsed_s: float | None = None
    n_epochs: int | None = None


class LeaderboardResponse(BaseModel):
    rows: list[LeaderboardRowResponse]
