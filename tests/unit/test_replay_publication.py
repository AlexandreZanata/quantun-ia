"""Unit tests for publication replay orchestration."""

from scripts.replay_publication import replay_publication, replay_steps


def test_replay_steps_full_includes_publication_large():
    names = [name for name, _ in replay_steps(artifacts_only=False)]
    assert names[0] == "publication_large"
    assert names[-1] == "latex_tables"


def test_replay_steps_artifacts_only_skips_runs():
    names = [name for name, _ in replay_steps(artifacts_only=True)]
    assert "publication_large" not in names
    assert names == ["export_results", "figures", "latex_tables"]


def test_replay_publication_dry_run_returns_zero(tmp_path):
    assert replay_publication(artifacts_only=True, dry_run=True, cwd=tmp_path) == 0
