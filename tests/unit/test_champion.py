"""Unit tests for champion/challenger promotion gate."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.training.champion import (
    ChampionManifest,
    holdout_delta_pp,
    latest_holdout_record,
    load_champion_manifest,
    promote_champion,
    read_jsonl_records,
    should_promote,
    should_rollback,
)


def test_holdout_delta_pp():
    assert holdout_delta_pp(0.97, 0.974) == pytest.approx(0.4, abs=0.01)
    assert holdout_delta_pp(0.974, 0.974) == 0.0


def test_should_promote_within_tolerance():
    assert should_promote(0.972, 0.974, max_delta_pp=0.5) is True
    assert should_promote(0.968, 0.974, max_delta_pp=0.5) is False


def test_should_rollback_on_large_regression():
    assert should_rollback(0.960, 0.974, max_regression_pp=1.0) is True
    assert should_rollback(0.970, 0.974, max_regression_pp=1.0) is False


def test_read_jsonl_records_skips_blank_lines(tmp_path):
    path = tmp_path / "experiments.jsonl"
    path.write_text(
        '{"exp_id":"a","test_accuracy":0.9}\n\n{"exp_id":"b","test_accuracy":0.8}\n',
        encoding="utf-8",
    )
    records = read_jsonl_records(path)
    assert len(records) == 2


def test_latest_holdout_record_picks_newest(tmp_path):
    records = [
        {
            "exp_id": "quantum_nano_bc_app",
            "model_name": "hybrid_sandwich_breast_cancer",
            "test_accuracy": 0.90,
            "eval_set": "holdout_test",
            "started_at": "2026-01-01T00:00:00",
        },
        {
            "exp_id": "quantum_nano_bc_app",
            "model_name": "hybrid_sandwich_breast_cancer",
            "test_accuracy": 0.95,
            "eval_set": "holdout_test",
            "started_at": "2026-06-01T00:00:00",
        },
    ]
    latest = latest_holdout_record(records, "quantum_nano_bc_app", "hybrid_sandwich_breast_cancer")
    assert latest is not None
    assert latest["test_accuracy"] == 0.95


def test_promote_champion_writes_manifest_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path)
    source = tmp_path / "quantum_nano_bc_app" / "hybrid_sandwich_breast_cancer" / "seed_42"
    source.mkdir(parents=True)
    (source / "best.pt").write_text("weights", encoding="utf-8")
    (source / "config.json").write_text("{}", encoding="utf-8")

    manifest = ChampionManifest(
        exp_id="quantum_nano_bc_app",
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        seed=42,
        holdout_accuracy=0.974,
        checkpoint_path=str(source),
        promoted_at=datetime.now(timezone.utc).isoformat(),
    )
    champion_dir = promote_champion(manifest)
    assert (champion_dir / "manifest.json").is_file()
    link = champion_dir / "checkpoint"
    assert link.is_symlink()
    assert link.resolve() == source.resolve()

    loaded = load_champion_manifest()
    assert loaded is not None
    assert loaded.holdout_accuracy == 0.974
    assert loaded.seed == 42
