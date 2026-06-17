"""Unit tests for CLI entry point."""

import sys
from pathlib import Path
from unittest.mock import patch

from scripts.cli import main

ROOT = Path(__file__).resolve().parents[2]
EXP_RUN = ROOT / "experiments" / "exp_001_quantum_vs_classical" / "run.py"


def test_cli_missing_script_returns_1():
    assert main(["/nonexistent/run.py"]) == 1


def test_cli_sets_profile_env(monkeypatch):
    monkeypatch.delenv("QML_PROFILE", raising=False)
    with patch("scripts.cli.runpy.run_path") as mock_run:
        rc = main([str(EXP_RUN), "--profile", "ci"])
    assert rc == 0
    assert mock_run.called
    import os

    assert os.environ.get("QML_PROFILE") == "ci"


def test_cli_inserts_root_on_path():
    original_path = sys.path.copy()
    try:
        with patch("scripts.cli.runpy.run_path"):
            main([str(EXP_RUN)])
        assert str(ROOT) in sys.path
    finally:
        sys.path[:] = original_path
