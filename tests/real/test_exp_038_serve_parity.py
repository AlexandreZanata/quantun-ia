"""Real gate — exp_038 hybrid serve parity on RTX 4060 (10K HIGGS rows)."""

from __future__ import annotations

import pytest

from experiments.exp_038_hybrid_serve_parity.run import MAX_DELTA, run_exp_038

pytestmark = pytest.mark.real


def test_exp_038_hybrid_serve_parity_publication():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_038 real gate")

    result = run_exp_038(profile="publication", verbose=False)

    assert result.n_rows == 10000
    assert result.max_delta_batch_api < MAX_DELTA
    assert result.max_delta_tool_api < MAX_DELTA
    assert result.max_delta_batch_tool < MAX_DELTA
