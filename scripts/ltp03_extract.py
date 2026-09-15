#!/usr/bin/env python3
"""Extrai declarações do LeanDojo Benchmark 4 a partir do código-fonte do mathlib4.

Para cada teorema de val com traço de táticas, produz o cabeçalho do arquivo
original e o prefixo da declaração (afirmação sem o corpo de prova), mantendo a
afirmação intacta. A taxa de extração e os casos falhos são registrados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4"
MATHLIB_ROOT = ROOT / "data" / "raw" / "mathlib4_src"
PROOF_BY = re.compile(r":=\s*by\b")
DECL_KEYWORD = re.compile(r"\b(theorem|lemma|example|instance|def)\b")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_declaration(declaration: str) -> tuple[str, str] | None:
    match = PROOF_BY.search(declaration)
    if match:
        return declaration[: match.end()], declaration[match.end() :]
    depth = 0
    index = 0
    while index < len(declaration) - 1:
        char = declaration[index]
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif char == ":" and declaration[index + 1] == "=" and depth == 0:
            return declaration[: index + 2], declaration[index + 2 :]
        index += 1
    return None


def rename_declaration(declaration: str, new_name: str) -> str:
    keyword = DECL_KEYWORD.search(declaration)
    if keyword is None:
        return declaration
    start = keyword.end()
    while start < len(declaration) and declaration[start].isspace():
        start += 1
    end = start
    while end < len(declaration) and (declaration[end].isalnum() or declaration[end] in "._'!"):
        end += 1
    return declaration[:start] + new_name + declaration[end:]


def extract_split(split: str, limit: int, mathlib_root: Path) -> dict[str, Any]:
    entries = json.loads((BENCHMARK_ROOT / split / "val.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    attempts = 0
    failures: list[dict[str, str]] = []
    for entry in entries:
        if not entry.get("traced_tactics"):
            continue
        attempts += 1
        if len(records) >= limit:
            break
        source_path = mathlib_root / entry["file_path"]
        if not source_path.exists():
            failures.append({"full_name": entry["full_name"], "reason": "arquivo ausente"})
            continue
        lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        start_line = entry["start"][0] - 1
        end_line = entry["end"][0]
        header = "".join(lines[:start_line])
        declaration = "".join(lines[start_line:end_line])
        pieces = split_declaration(declaration)
        if pieces is None:
            failures.append({"full_name": entry["full_name"], "reason": "marcador de prova ausente"})
            continue
        decl_prefix, _ = pieces
        if len(decl_prefix) < 10:
            failures.append({"full_name": entry["full_name"], "reason": "prefixo suspeito"})
            continue
        records.append(
            {
                "split": split,
                "full_name": entry["full_name"],
                "file_path": entry["file_path"],
                "start": entry["start"],
                "end": entry["end"],
                "header": header,
                "decl_prefix": decl_prefix,
                "statement_sha256": hashlib.sha256(
                    normalize(decl_prefix).encode("utf-8")
                ).hexdigest(),
                "first_state": entry["traced_tactics"][0]["state_before"],
                "first_tactic": entry["traced_tactics"][0]["tactic"],
            }
        )
    return {
        "split": split,
        "attempts": attempts,
        "extracted": len(records),
        "failures": failures,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai declarações do LeanDojo Benchmark 4")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=["random", "novel_premises"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"limit": args.limit, "splits": []}
    for split in args.splits:
        result = extract_split(split, args.limit, MATHLIB_ROOT)
        payload["splits"].append(
            {
                "split": result["split"],
                "attempts": result["attempts"],
                "extracted": result["extracted"],
                "failures": result["failures"],
            }
        )
        for record in result["records"]:
            print(json.dumps(record, ensure_ascii=False))
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": payload["splits"]}, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
