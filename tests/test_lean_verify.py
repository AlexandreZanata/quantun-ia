from pathlib import Path

from scripts.lean_verify import (
    DEFAULT_ADVERSARIAL,
    DEFAULT_SMOKE,
    LeanContext,
    build_source,
    evaluate_results,
    load_suites,
    materialize_cases,
    normalize_statement,
    parse_axioms,
    run_lean,
    run_pass,
    scan_forbidden,
    statement_sha256,
    validate_adversarial,
    validate_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = load_suites(DEFAULT_SMOKE, DEFAULT_ADVERSARIAL)[0]["policy"]["forbidden_tokens"]


def test_normalize_statement_collapses_whitespace_and_comments():
    raw = "  ∀ n : Nat,   n + 0 = n -- comentário\n"
    assert normalize_statement(raw) == "∀ n : Nat, n + 0 = n"
    assert normalize_statement("/- sorry admit -/ n = n") == "n = n"


def test_statement_hash_is_deterministic_and_comment_insensitive():
    assert statement_sha256("n + 0 = n") == statement_sha256("n + 0 = n")
    assert statement_sha256("n + 0 = n") == statement_sha256("n + 0 = n -- nota")
    assert statement_sha256("n + 0 = n") != statement_sha256("n + 1 = n")


def test_scanner_ignores_comments_and_strings_but_flags_real_tokens():
    assert scan_forbidden("-- sorry\nnorm_num", FORBIDDEN) == []
    assert scan_forbidden("/- admit -/\nring", FORBIDDEN) == []
    assert scan_forbidden('have h : String := "axiom sorry"', FORBIDDEN) == []
    assert scan_forbidden("run_tac foo", FORBIDDEN) == ["run_tac"]
    assert scan_forbidden("exact sorryAx _ true", FORBIDDEN) == ["sorryAx"]
    assert scan_forbidden("set_option maxHeartbeats 0", FORBIDDEN) == ["set_option"]


def test_scanner_does_not_flag_partial_words():
    assert scan_forbidden("exact Nat.sorry_free", FORBIDDEN) == []
    assert scan_forbidden("exact assumption", FORBIDDEN) == []


def test_build_source_contains_sealed_statement_and_audit():
    smoke, adversarial = load_suites(DEFAULT_SMOKE, DEFAULT_ADVERSARIAL)
    case = materialize_cases(smoke, adversarial)[0]
    source = build_source(case, smoke)
    assert "import Mathlib.Tactic" in source
    assert "set_option autoImplicit false" in source
    assert case.statement in source
    assert "#print axioms ltp_target" in source


def test_parse_axioms_variants():
    assert parse_axioms("'ltp_target' does not depend on any axioms") == []
    assert parse_axioms("'ltp_target' depends on axioms: [propext, Classical.choice]") == [
        "propext",
        "Classical.choice",
    ]
    assert parse_axioms("nada") is None


def test_suites_are_valid_and_sealed():
    smoke, adversarial = load_suites(DEFAULT_SMOKE, DEFAULT_ADVERSARIAL)
    assert validate_smoke(smoke) == []
    assert validate_adversarial(adversarial) == []
    assert all(item["statement_sha256"] for item in smoke["theorems"])


def test_case_materialization_covers_valid_mutations_and_specials():
    smoke, adversarial = load_suites(DEFAULT_SMOKE, DEFAULT_ADVERSARIAL)
    cases = materialize_cases(smoke, adversarial)
    valid = [case for case in cases if case.expectation == "accept"]
    reject = [case for case in cases if case.expectation == "reject"]
    ids = {case.id for case in cases}
    assert len([case for case in cases if case.kind == "sealed_valid"]) == 32
    assert len(valid) == 36
    assert len(reject) == 71
    assert len(ids) == len(cases)
    assert any(case.tampered_statement for case in cases)
    assert any(case.preamble for case in cases)


def test_run_pass_preserves_run_metadata(tmp_path):
    smoke = {
        "imports": ["Mathlib.Tactic"],
        "policy": {"allowed_axioms": [], "forbidden_tokens": [], "max_heartbeats": 200000},
    }
    sentinel = tmp_path / "run.json"
    sentinel.write_text("{}\n", encoding="utf-8")
    run_pass([], smoke, tmp_path, LeanContext("", "", ""), 1)
    assert sentinel.exists()


def test_run_lean_enforces_wall_timeout(tmp_path):
    fake_lean = tmp_path / "fake_lean"
    fake_lean.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    fake_lean.chmod(0o755)
    context = LeanContext(lean_bin=str(fake_lean), lean_path="", toolchain="fake")
    run = run_lean("-- nada", tmp_path / "scratch", context, timeout_seconds=1.0)
    assert run.status == "timeout"
    assert run.wall_seconds < 15


def test_evaluate_results_detects_expectation_and_reason_mismatch():
    good = {"case_id": "A", "expectation": "reject", "verdict": "reject", "reason": "timeout", "expected_reason": "timeout"}
    assert evaluate_results([good])["passed"] is True
    wrong_reason = dict(good, reason="lean_error")
    assert evaluate_results([wrong_reason])["passed"] is False
    wrong_verdict = dict(good, verdict="accept")
    assert evaluate_results([wrong_verdict])["passed"] is False
