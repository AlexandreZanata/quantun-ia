import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_only_audited_experiments_are_active() -> None:
    registry = json.loads((ROOT / "research/evidence_registry.json").read_text())
    expected = {"exp_011", "exp_012", "exp_024", "exp_025"}
    assert {item["exp_id"] for item in registry["active_experiments"]} == expected

    active_dirs = {
        "_".join(path.name.split("_")[:2])
        for path in (ROOT / "experiments").glob("exp_*")
        if path.is_dir()
    }
    assert active_dirs == expected


def test_dashboard_is_read_only_and_has_no_lab_pages() -> None:
    pages = ROOT / "dashboard/pages"
    assert not pages.exists() or not any(pages.glob("*.py"))
    dashboard = (ROOT / "dashboard/app.py").read_text(encoding="utf-8").lower()
    assert "st.button(" not in dashboard
    assert "model lab" not in dashboard
    assert "agro risk lab" not in dashboard


def test_active_claims_do_not_claim_quantum_advantage() -> None:
    registry = json.loads((ROOT / "research/evidence_registry.json").read_text())
    assert all(
        item["delta_pp"] <= 0 or item["exp_id"] == "exp_012"
        for item in registry["active_experiments"]
    )
    assert registry["claim_policy"] == "internal_reproducible_evidence_not_external_confirmation"

