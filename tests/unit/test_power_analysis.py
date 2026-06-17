"""CLI smoke tests for power analysis script."""

from scripts.power_analysis import main, power_table


def test_power_table_returns_rows():
    rows = power_table(n_seeds=[5, 10])
    assert len(rows) == 2
    assert rows[0][0] == 5
    assert rows[1][1] < rows[0][1]


def test_main_single_n(capsys):
    assert main(["--n-seeds", "10"]) == 0
    out = capsys.readouterr().out
    assert "MDE" in out


def test_main_table_mode(capsys):
    assert main(["--table"]) == 0
    out = capsys.readouterr().out
    assert "| Seeds |" in out
