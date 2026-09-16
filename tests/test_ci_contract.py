from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"

FORBIDDEN_LEGACY_CI_TERMS = {
    "Experiment Smoke",
    "Paper Build",
    "pip-audit",
    "Docker Test Suite",
    "E2E (API + JWT + async jobs)",
    "Open Science Preflight",
    "Type Check (mypy)",
    "PennyLane",
    "qml-run",
}


def test_only_the_lean_research_workflow_is_active():
    assert sorted(path.name for path in WORKFLOWS.glob("*.y*ml")) == ["ci.yml"]


def test_ci_runs_the_formal_research_contract():
    text = CI.read_text(encoding="utf-8")
    assert "name: Lean Research CI" in text
    assert "name: Formal Research Gate" in text
    assert "leanprover/lean-action@v1" in text
    assert "lake-package-directory: lean" in text
    assert "make PYTHON=python check" in text
    assert "make PYTHON=python lean-gate" in text
    assert "permissions:\n  contents: read" in text
    assert not (FORBIDDEN_LEGACY_CI_TERMS & set(term for term in FORBIDDEN_LEGACY_CI_TERMS if term in text))


def test_ci_has_no_schedule_or_legacy_dependency_install():
    text = CI.read_text(encoding="utf-8")
    assert "schedule:" not in text
    assert "requirements.lock" not in text
    assert "requirements-dev.txt" not in text
    assert "torch" not in text.lower()
    assert "docker" not in text.lower()

