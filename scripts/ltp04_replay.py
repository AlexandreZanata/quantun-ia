#!/usr/bin/env python3
"""LH-069 — reexecução em ambiente limpo.

Repete a verificação de provas aceitas, mutações de afirmação e provas
inválidas em dois passes limpos independentes e exige vereditos, axiomas e
hash semântico idênticos. A prova nunca é aceita com `sorry` ou axioma extra.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import (  # noqa: E402
    canonical_type_sha256,
    parse_canonical_type,
    rename_declaration,
    resolve_lean_context,
    run_lean,
    statement_sha256,
)

NATIVE_PROJECT = ROOT / "data" / "raw" / "mathlib4_src"
MEMORY_LIMIT = 8 * 1024**3
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
TARGET = "ltp_target"
AXIOM_FREE = re.compile(r"'([\w.']*\.)?ltp_target' does not depend on any axioms")
AXIOM_DEPENDS = re.compile(r"'([\w.']*\.)?ltp_target' depends on axioms: \[(.*?)\]")


def parse_axioms_any(stdout: str, name: str = TARGET) -> list[str] | None:
    escaped = re.escape(name)
    if re.search(rf"'([\w.']*\.)?{escaped}' does not depend on any axioms", stdout):
        return []
    match = re.search(rf"'([\w.']*\.)?{escaped}' depends on axioms: \[(.*?)\]", stdout, re.DOTALL)
    if not match:
        return None
    inner = match.group(2).strip()
    if not inner:
        return []
    return [item.strip() for item in inner.split(",") if item.strip()]


def mutate_declaration(declaration: str) -> tuple[str, str] | None:
    for pattern, replacement, label in (
        (r"(?<![\d.])0(?![\d.])", "1", "numeral_0_to_1"),
        (r"≤", "<", "le_to_lt"),
        (r" \+ ", " * ", "add_to_mul"),
    ):
        mutated, count = re.subn(pattern, replacement, declaration, count=1)
        if count:
            return mutated, label
    return None


def build_replay_source(
    record: dict,
    declaration: str,
    proof: str,
    rename: bool = True,
) -> str:
    from scripts.lean_verify import strip_to_additive, with_options

    options = ["set_option autoImplicit false"] if record["file_path"].startswith("Mathlib/") else []
    header = strip_to_additive(record["header"])
    if not header.endswith("\n"):
        header += "\n"
    if rename:
        declaration_text = rename_declaration(declaration, TARGET)
        check_name = TARGET
        body = "\n" + "\n".join("  " + line for line in proof.splitlines()) + "\n"
    else:
        declaration_text = declaration
        check_name = record["full_name"]
        body = ""
    return (
        with_options(header, options)
        + declaration_text
        + body
        + f"#print axioms {check_name}\n"
        + "set_option pp.all true\n"
        + f"#check @{check_name}\n"
    )


def collect_valid_cases(runs_root: Path) -> list[dict]:
    cases: list[dict] = []
    for name in ("symbolic_single_random.json", "symbolic_single_novel_premises.json"):
        path = runs_root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["results"]:
            for tactic, payload in item["per_tactic"].items():
                if payload["verdict"] == "closed":
                    cases.append(
                        {
                            "id": f"{item['split']}:{item['full_name']}:symbolic",
                            "split": item["split"],
                            "full_name": item["full_name"],
                            "proof": tactic,
                            "origin": name,
                        }
                    )
                    break
    recheck = runs_root / "reprover_recheck.json"
    tac_path = runs_root / "reprover_tacgen.json"
    if recheck.exists() and tac_path.exists():
        tactics = {
            (item["split"], item["full_name"]): item["generated_tactic"]
            for item in json.loads(tac_path.read_text(encoding="utf-8"))["results"]
        }
        data = json.loads(recheck.read_text(encoding="utf-8"))
        for item in data["cases"]:
            proof = tactics.get((item["split"], item["full_name"]))
            if item["corrected_verdict"] == "closed" and proof:
                cases.append(
                    {
                        "id": f"{item['split']}:{item['full_name']}:reprover",
                        "split": item["split"],
                        "full_name": item["full_name"],
                        "proof": proof,
                        "origin": "reprover_recheck.json",
                    }
                )
    return cases


def collect_original_cases(
    records: dict[tuple[str, str], dict],
    mathlib_root: Path,
    per_split: int = 2,
) -> list[dict]:
    cases: list[dict] = []
    for split in ("random", "novel_premises"):
        selected = [item for (item_split, _), item in records.items() if item_split == split][:per_split]
        for record in selected:
            source_path = mathlib_root / record["file_path"]
            if not source_path.exists():
                continue
            lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
            declaration = "".join(lines[record["start"][0] - 1 : record["end"][0]])
            cases.append(
                {
                    "id": f"{split}:{record['full_name']}:original_proof",
                    "split": split,
                    "full_name": record["full_name"],
                    "kind": "original_proof",
                    "expect": "accept",
                    "declaration": declaration,
                    "origin": "mathlib",
                }
            )
    return cases


def materialize_cases(
    valid_cases: list[dict],
    original_cases: list[dict],
    records: dict[tuple[str, str], dict],
    sealed: dict[str, dict],
) -> list[dict]:
    cases: list[dict] = []
    for case in valid_cases:
        record = records.get((case["split"], case["full_name"]))
        if record is None or case["full_name"] not in sealed:
            continue
        cases.append({**case, "kind": "valid", "expect": "accept"})
    for case in original_cases:
        if case["full_name"] in sealed:
            cases.append(case)
    for case in cases[:2]:
        record = records[(case["split"], case["full_name"])]
        mutation = mutate_declaration(record["decl_prefix"])
        if mutation is None:
            continue
        _, label = mutation
        cases.append(
            {
                "id": f"{case['split']}:{case['full_name']}:mutant:{label}",
                "split": case["split"],
                "full_name": case["full_name"],
                "kind": "mutant_semantic",
                "expect": "reject",
                "mutation": label,
            }
        )
    for case in cases[:2]:
        cases.append(
            {
                "id": f"{case['split']}:{case['full_name']}:invalid_proof",
                "split": case["split"],
                "full_name": case["full_name"],
                "kind": "invalid_proof",
                "expect": "reject",
                "proof": "exact (0 : Nat)",
            }
        )
    return cases


def verify(case: dict, records: dict, sealed: dict, context, root: Path, index: int) -> dict:
    record = records[(case["split"], case["full_name"])]
    declaration = record["decl_prefix"]
    proof = case.get("proof", "")
    rename = True
    check_name = TARGET
    if case["kind"] == "original_proof":
        declaration = case["declaration"]
        proof = ""
        rename = False
        check_name = record["full_name"]
    elif case["kind"] == "mutant_semantic":
        mutation = mutate_declaration(declaration)
        if mutation is None:
            return {
                "id": case["id"],
                "kind": case["kind"],
                "expect": case["expect"],
                "status": "skipped",
                "exit_code": None,
                "wall_seconds": 0.0,
                "verdict": "reject",
                "reason": "mutation_unavailable",
                "axioms": None,
                "semantic_sha256": None,
                "sealed_semantic_sha256": None,
            }
        declaration, label = mutation
        case = {**case, "mutation": label}
        proof = case.get("original_proof") or "sorry"
    source = build_replay_source(record, declaration, proof, rename=rename)
    run = run_lean(source, root / f"case_{index:03d}", context, 60.0, MEMORY_LIMIT)
    result = {
        "id": case["id"],
        "kind": case["kind"],
        "expect": case["expect"],
        "status": run.status,
        "exit_code": run.exit_code,
        "wall_seconds": round(run.wall_seconds, 3),
        "verdict": "reject",
        "reason": None,
        "axioms": None,
        "semantic_sha256": None,
        "sealed_semantic_sha256": sealed.get(case["full_name"], {}).get("semantic_sha256"),
    }
    if run.status != "ok":
        result["reason"] = run.status
        return result
    axioms = parse_axioms_any(run.stdout, check_name)
    result["axioms"] = axioms
    if axioms is None:
        result["reason"] = "missing_axiom_report"
        return result
    extra = [axiom for axiom in axioms if axiom not in ALLOWED_AXIOMS]
    if extra:
        result["reason"] = "extra_axiom"
        return result
    canonical = parse_canonical_type(run.stdout, check_name)
    if canonical is None:
        result["reason"] = "missing_semantic_type"
        return result
    semantic = canonical_type_sha256(canonical)
    result["semantic_sha256"] = semantic
    if case["kind"] in ("valid", "original_proof"):
        if semantic != result["sealed_semantic_sha256"]:
            result["reason"] = "semantic_hash_mismatch"
            return result
    if case["kind"] == "mutant_semantic":
        if semantic == result["sealed_semantic_sha256"]:
            result["reason"] = "mutation_passed"
            result["verdict"] = "accept"
            return result
        result["reason"] = "semantic_hash_mismatch"
        return result
    result["verdict"] = "accept"
    return result


def compare_passes(first: list[dict], second: list[dict]) -> dict:
    mismatches = []
    for left, right in zip(first, second):
        fields = ("verdict", "reason", "axioms", "semantic_sha256")
        if any(left[field] != right[field] for field in fields):
            mismatches.append(
                {
                    "id": left["id"],
                    "first": {field: left[field] for field in fields},
                    "second": {field: right[field] for field in fields},
                }
            )
    return {"identical": not mismatches, "mismatches": mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description="LH-069 replay limpo")
    parser.add_argument("--statements", required=True)
    parser.add_argument("--hashes", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = {
        (item["split"], item["full_name"]): item
        for item in (
            json.loads(line)
            for line in Path(args.statements).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    sealed = json.loads(Path(args.hashes).read_text(encoding="utf-8"))["hashes"]
    valid_cases = collect_valid_cases(Path(args.runs_root))
    original_cases = collect_original_cases(records, NATIVE_PROJECT, per_split=2)
    cases = materialize_cases(valid_cases, original_cases, records, sealed)
    for case in cases:
        if case["kind"] == "mutant_semantic":
            original = next(
                (item for item in valid_cases if item["full_name"] == case["full_name"]), None
            )
            case["original_proof"] = original["proof"] if original else "sorry"
    context = resolve_lean_context(NATIVE_PROJECT)
    root = Path(args.out).with_suffix("")
    root.mkdir(parents=True, exist_ok=True)
    first = [verify(case, records, sealed, context, root / "pass_a", index) for index, case in enumerate(cases)]
    second = [verify(case, records, sealed, context, root / "pass_b", index) for index, case in enumerate(cases)]
    comparison = compare_passes(first, second)
    failures = []
    for item in first:
        if item["verdict"] != item["expect"]:
            failures.append({"id": item["id"], "expected": item["expect"], "observed": item["verdict"], "reason": item["reason"]})
    if not comparison["identical"]:
        failures.append({"replay": "divergência entre passes", "detail": comparison["mismatches"]})
    summary = {
        "cases": len(cases),
        "valid": sum(1 for item in first if item["kind"] == "valid"),
        "valid_accepted": sum(1 for item in first if item["kind"] == "valid" and item["verdict"] == "accept"),
        "mutants": sum(1 for item in first if item["kind"] == "mutant_semantic"),
        "mutants_rejected": sum(1 for item in first if item["kind"] == "mutant_semantic" and item["verdict"] == "reject"),
        "invalid_proofs": sum(1 for item in first if item["kind"] == "invalid_proof"),
        "invalid_rejected": sum(1 for item in first if item["kind"] == "invalid_proof" and item["verdict"] == "reject"),
        "replay_identical": comparison["identical"],
        "failures": failures,
        "wall_seconds_total": round(sum(item["wall_seconds"] for item in first) + sum(item["wall_seconds"] for item in second), 3),
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "pass_a": first, "pass_b": second, "comparison": comparison}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    passed = not failures
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
