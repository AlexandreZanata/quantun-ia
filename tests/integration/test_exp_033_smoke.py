"""Integration smoke test — exp_033 batch vs API vs chatbot parity on HIGGS."""

from __future__ import annotations

import pytest

from experiments.exp_033_higgs_serve_parity.run import MAX_DELTA, run_exp_033

pytestmark = pytest.mark.real


def test_exp_033_ci_serve_parity_smoke(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")

    result = run_exp_033(profile="ci", verbose=False, require_cuda=False)

    assert result.n_rows == 500
    assert result.max_delta_batch_api < MAX_DELTA
    assert result.max_delta_tool_api < MAX_DELTA
    assert result.max_delta_batch_tool < MAX_DELTA
