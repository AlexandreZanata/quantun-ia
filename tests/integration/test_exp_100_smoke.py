"""Smoke test — exp_100 Cycle v2 grand leaderboard imports and ci run."""

from experiments.exp_100_cycle2_grand_leaderboard.run import gate_passed, run_exp_100


def test_exp_100_ci_synthesis():
    result = run_exp_100(profile="ci", verbose=False)
    assert gate_passed(result)
    assert result.n_rows == 16
    assert result.hypothesis_confirmed
    assert result.json_path.is_file()
    assert result.latex_path.is_file()
