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
