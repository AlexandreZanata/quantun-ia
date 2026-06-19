"""Smoke test — EXP 043 calibration module imports."""

from __future__ import annotations

from pathlib import Path


def test_exp_043_run_module_imports():
    path = Path("experiments/exp_043_calibration_synthea/run.py")
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "run_exp_043" in source
    assert "CalibrationEvaluationDTO" in source


def test_exp_043_hypothesis_exists():
    hypothesis = Path("experiments/exp_043_calibration_synthea/hypothesis.md")
    assert hypothesis.is_file()
    text = hypothesis.read_text(encoding="utf-8")
    assert "isotonic" in text.lower()
    assert "ECE" in text
