"""Ensure experiment hypothesis files are filled before runs."""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
PLACEHOLDER = "[Write in clear English"


def test_hypothesis_files_have_real_content():
    paths = sorted(EXPERIMENTS_DIR.glob("exp_*/hypothesis.md"))
    assert paths, "expected at least one experiment hypothesis.md"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert PLACEHOLDER not in text, f"{path} still contains template placeholder"
