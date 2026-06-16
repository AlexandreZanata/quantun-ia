"""Unit tests for curriculum epoch budget."""

from src.training.curriculum import curriculum_total_epochs


def test_curriculum_total_epochs_matches_stages_plus_refine():
    cfg = {
        "curriculum_stages": 4,
        "epochs_per_stage": 12,
        "refine_epochs": 12,
    }
    assert curriculum_total_epochs(cfg) == 60
