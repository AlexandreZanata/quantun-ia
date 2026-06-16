"""Smoke tests — experiment run.py imports."""

import importlib.util
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def _experiment_run_paths():
    return sorted(EXPERIMENTS_DIR.glob("exp_*/run.py"))


@pytest.mark.parametrize("run_py", _experiment_run_paths(), ids=lambda p: p.parent.name)
def test_experiment_run_imports(run_py: Path):
    spec = importlib.util.spec_from_file_location(run_py.parent.name, run_py)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
