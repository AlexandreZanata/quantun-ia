#!/usr/bin/env python3
"""LH-070 — hash imutável da afirmação.

Para cada afirmação do LeanDojo Benchmark 4 (val), calcula o hash semântico do
tipo elaborado com `pp.all` no ambiente nativo, repete o cálculo em ambiente
limpo e mede mutações textuais que devem ser rejeitadas por hash divergente ou
por falha de elaboração.
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
    build_semantic_source,
    canonical_type_sha256,
    parse_canonical_type,
    resolve_lean_context,
    run_lean,
)

NATIVE_PROJECT = ROOT / "data" / "raw" / "mathlib4_src"
MEMORY_LIMIT = 8 * 1024**3


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


def semantic_hash(record: dict, context, root: Path, label: str) -> dict:
    source = build_semantic_source(
        record["header"],
        record["decl_prefix"],
        auto_implicit_false=record["file_path"].startswith("Mathlib/"),
    )
    run = run_lean(source, root / label, context, 60.0, MEMORY_LIMIT)
    canonical = parse_canonical_type(run.stdout)
    error_count = run.stdout.count(": error:")
    return {
        "status": run.status,
        "exit_code": run.exit_code,
        "semantic_sha256": canonical_type_sha256(canonical) if canonical else None,
        "canonical_length": len(canonical) if canonical else 0,
        "error_count": error_count,
        "wall_seconds": round(run.wall_seconds, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LH-070 hash semântico")
    parser.add_argument("--statements", required=True)
    parser.add_argument("--per-split", type=int, default=16)
    parser.add_argument("--mutants-per-split", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hashes-out", required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.statements).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    for split in ("random", "novel_premises"):
        selected.extend([item for item in records if item["split"] == split][: args.per_split])
    context = resolve_lean_context(NATIVE_PROJECT)
    root = Path(args.out).with_suffix("")
    root.mkdir(parents=True, exist_ok=True)

    case_results = []
    sealed_hashes: dict[str, dict] = {}
    for index, record in enumerate(selected):
        first = semantic_hash(record, context, root, f"case_{index:03d}_run1")
        second = semantic_hash(record, context, root, f"case_{index:03d}_run2")
        deterministic = (
            first["semantic_sha256"] is not None
            and first["semantic_sha256"] == second["semantic_sha256"]
        )
        entry = {
            "split": record["split"],
            "full_name": record["full_name"],
            "statement_sha256": record["statement_sha256"],
            "semantic_sha256": first["semantic_sha256"],
            "run1_status": first["status"],
            "run2_status": second["status"],
            "run1_errors": first["error_count"],
            "deterministic": deterministic,
            "wall_seconds": round(first["wall_seconds"] + second["wall_seconds"], 3),
        }
        case_results.append(entry)
        if first["semantic_sha256"]:
            sealed_hashes[record["full_name"]] = {
                "split": record["split"],
                "statement_sha256": record["statement_sha256"],
                "semantic_sha256": first["semantic_sha256"],
            }
        print(
            f"[semantic {index + 1}/{len(selected)}] {record['full_name'][:45]} "
            f"det={deterministic} {first['status']}",
            flush=True,
        )

    mutants = []
    for split in ("random", "novel_premises"):
        subset = [item for item in records if item["split"] == split][: args.mutants_per_split]
        for index, record in enumerate(subset):
            mutation = mutate_declaration(record["decl_prefix"])
            if mutation is None:
                continue
            mutated_decl, label = mutation
            mutated_record = dict(record)
            mutated_record["decl_prefix"] = mutated_decl
            run = semantic_hash(mutated_record, context, root, f"mut_{split}_{index:03d}")
            sealed = sealed_hashes.get(record["full_name"], {}).get("semantic_sha256")
            passed = run["semantic_sha256"] is not None and run["semantic_sha256"] == sealed
            mutants.append(
                {
                    "split": split,
                    "full_name": record["full_name"],
                    "mutation": label,
                    "status": run["status"],
                    "mutated_semantic_sha256": run["semantic_sha256"],
                    "passed_gate": passed,
                }
            )
            print(
                f"[mutant {split} {index + 1}] {record['full_name'][:40]} {label} -> "
                f"{'PASSOU (falha)' if passed else 'rejeitada'}",
                flush=True,
            )

    deterministic_count = sum(1 for item in case_results if item["deterministic"])
    mutants_passed = sum(1 for item in mutants if item["passed_gate"])
    summary = {
        "cases": len(case_results),
        "hash_deterministic": deterministic_count,
        "hash_deterministic_rate": round(deterministic_count / len(case_results), 4) if case_results else 0.0,
        "mutants": len(mutants),
        "mutants_rejected": sum(1 for item in mutants if not item["passed_gate"]),
        "mutants_passed_gate": mutants_passed,
        "wall_seconds_total": round(sum(item["wall_seconds"] for item in case_results), 3),
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "cases": case_results, "mutants": mutants}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    Path(args.hashes_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.hashes_out).write_text(
        json.dumps({"schema_version": 1, "source": str(args.statements), "hashes": sealed_hashes}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if mutants_passed == 0 and deterministic_count == len(case_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
