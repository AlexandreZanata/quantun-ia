"""Integration smoke test — exp_035 Synthea serve parity."""

from __future__ import annotations

from experiments.exp_035_synthea_serve_parity.run import MAX_DELTA, run_exp_035


def test_exp_035_ci_serve_parity_smoke(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")

    result = run_exp_035(profile="ci", verbose=False, require_cuda=False)

    assert result.n_rows == 500
    assert result.max_delta_batch_api < MAX_DELTA
    assert result.max_delta_tool_api < MAX_DELTA
    assert result.max_delta_batch_tool < MAX_DELTA
