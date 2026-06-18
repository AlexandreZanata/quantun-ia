"""Unit tests for QuantumNano-BC model card generation."""

import json
from pathlib import Path

from scripts.generate_model_card import generate_quantum_nano_bc_card, write_quantum_nano_bc_card


def _summary_record() -> dict:
    return {
        "exp_id": "exp_024",
        "model_name": "exp_024_multi_seed_summary",
        "record_type": "multi_seed_summary",
        "started_at": "2026-06-18T00:00:00",
        "summary": {
            "hybrid_sandwich": {
                "mean": 0.95,
                "ci_low": 0.93,
                "ci_high": 0.97,
                "n_seeds": 2,
            },
            "logistic_regression": {"mean": 0.94, "ci_low": 0.92, "ci_high": 0.96, "n_seeds": 2},
            "xgboost_shallow": {"mean": 0.96, "ci_low": 0.94, "ci_high": 0.98, "n_seeds": 2},
        },
    }


def test_generate_model_card_includes_metrics(tmp_path: Path):
    jsonl = tmp_path / "experiments.jsonl"
    jsonl.write_text(json.dumps(_summary_record()) + "\n", encoding="utf-8")
    text = generate_quantum_nano_bc_card(jsonl_path=jsonl)
    assert "QuantumNano-BC" in text
    assert "hybrid_sandwich" in text
    assert "95.0%" in text
    assert "logistic_regression" in text


def test_write_model_card_creates_file(tmp_path: Path):
    jsonl = tmp_path / "experiments.jsonl"
    out = tmp_path / "cards" / "quantum_nano_bc.md"
    jsonl.write_text(json.dumps(_summary_record()) + "\n", encoding="utf-8")
    path = write_quantum_nano_bc_card(jsonl_path=jsonl, output_path=out)
    assert path.is_file()
    assert "Not for clinical deployment" in path.read_text(encoding="utf-8")
