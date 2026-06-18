"""Real gate — exp_035 Synthea serve parity on RTX 4060 (10K rows)."""

from __future__ import annotations

import pytest

from experiments.exp_035_synthea_serve_parity.run import MAX_DELTA, run_exp_035

pytestmark = pytest.mark.real


def test_exp_035_synthea_serve_parity_publication():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_035 real gate")

    result = run_exp_035(profile="publication", verbose=False)

    assert result.n_rows == 10000
    assert result.max_delta_batch_api < MAX_DELTA
    assert result.max_delta_tool_api < MAX_DELTA
    assert result.max_delta_batch_tool < MAX_DELTA
