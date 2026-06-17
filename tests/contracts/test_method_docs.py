"""Contract tests for citable method documentation (Phase 24)."""

from __future__ import annotations

from pathlib import Path

METHOD_DOC = Path("docs/method_adaptive_lr.md")
MODULE_PATH = Path("src/training/adaptive_lr.py")
EXP_RESULTS = Path("experiments/exp_015_adaptive_qnn/results.md")

REQUIRED_SECTIONS = (
    "## Algorithm",
    "## Configuration",
    "## Experiment linkage",
    "## Empirical status",
    "## Limitations",
)


def test_method_adaptive_lr_doc_exists():
    assert METHOD_DOC.is_file(), "docs/method_adaptive_lr.md is required for Phase 24"


def test_method_doc_required_sections():
    text = METHOD_DOC.read_text(encoding="utf-8")
    for heading in REQUIRED_SECTIONS:
        assert heading in text, f"missing section: {heading}"


def test_method_doc_links_to_implementation():
    text = METHOD_DOC.read_text(encoding="utf-8")
    assert "adaptive_lr.py" in text
    assert "AdaptiveLRConfig" in text
    assert "compute_lr_scale" in text
    assert MODULE_PATH.is_file()


def test_method_doc_links_to_exp_015():
    text = METHOD_DOC.read_text(encoding="utf-8")
    assert "exp_015" in text
    assert EXP_RESULTS.is_file()


def test_method_doc_includes_pseudocode():
    text = METHOD_DOC.read_text(encoding="utf-8")
    assert "var_target" in text
    assert "warmup_epochs" in text
    assert "```" in text
