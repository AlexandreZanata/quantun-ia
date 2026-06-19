"""Real gate — exp_041 clinical case validation on RTX 4060."""

from __future__ import annotations

import pytest

from experiments.exp_041_human_cv_clinical_cases.run import MIN_SPEARMAN, run_exp_041

pytestmark = pytest.mark.real


def test_exp_041_clinical_validation_gate():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for exp_041 real gate")

    result = run_exp_041(verbose=False)

    assert len(result.case_scores) == 8
    assert result.spearman_rho >= MIN_SPEARMAN
    assert result.separation_pp > 0
    assert result.spearman_rho >= MIN_SPEARMAN
    assert result.passed is True
