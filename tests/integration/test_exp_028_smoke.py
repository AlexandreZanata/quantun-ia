"""Integration smoke test — exp_028 chatbot tool vs API parity."""

from __future__ import annotations

from pathlib import Path

from experiments.exp_028_chatbot_tool_parity.run import MAX_DELTA, run_exp_028

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chatbot_dialogues"


def test_exp_028_ci_chatbot_tool_parity_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    results = run_exp_028(
        profile="ci",
        fixtures_dir=FIXTURES,
        verbose=False,
        require_cuda=False,
        bootstrap_checkpoint=True,
    )

    assert len(results) == 10
    assert all(r.feature_count == 30 for r in results)
    assert all(r.has_disclaimer for r in results)
    assert max(r.max_delta for r in results) < MAX_DELTA
