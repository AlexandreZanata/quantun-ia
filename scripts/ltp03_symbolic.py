#!/usr/bin/env python3
"""Baseline simbólico LTP-03 e portabilidade para o ambiente moderno.

Modo `portfolio`: uma chamada por meta com as cinco táticas renomeadas no mesmo
arquivo, cada declaração limitada por `maxHeartbeats`. Modo `portability`:
verifica se a afirmação de 2024 ainda elabora no mathlib pinado atual, com os
imports originais substituídos por `import Mathlib` e prova `by sorry`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import resolve_lean_context, run_lean  # noqa: E402

ENVIRONMENTS = {
    "modern": ROOT / "lean",
    "native": ROOT / "data" / "raw" / "mathlib4_src",
}
TACTICS = ["simp", "aesop", "omega", "linarith", "nlinarith"]
HEARTBEATS = 200000
IMPORT_LINE = re.compile(r"^\s*(public\s+)?import\s")
DECL_KEYWORD = re.compile(r"\b(theorem|lemma|example|instance|def)\b")
MEMORY_LIMIT = 8 * 1024**3


def rename_declaration(declaration: str, new_name: str) -> str:
    keyword = DECL_KEYWORD.search(declaration)
    if keyword is None or keyword.group(1) == "example":
        return declaration
    start = keyword.end()
    while start < len(declaration) and declaration[start].isspace():
        start += 1
    end = start
    while end < len(declaration) and (declaration[end].isalnum() or declaration[end] in "._'!"):
        end += 1
    return declaration[:start] + new_name + declaration[end:]


def replace_imports(header: str) -> str:
    lines = [line for line in header.splitlines(keepends=True) if not IMPORT_LINE.match(line)]
    return "import Mathlib\n" + "".join(lines)


def portfolio_source(record: dict) -> tuple[str, list[tuple[str, int, int]]]:
    parts = [record["header"]]
    if not parts[0].endswith("\n"):
        parts[0] += "\n"
    parts.append("set_option autoImplicit false\n")
    parts.append(f"set_option maxHeartbeats {HEARTBEATS}\n")
    ranges: list[tuple[str, int, int]] = []
    for tactic in TACTICS:
        declaration = rename_declaration(record["decl_prefix"], f"ltp_try_{tactic}")
        block = declaration + f"\n  {tactic}\n"
        start_line = sum(part.count("\n") for part in parts) + 1
        parts.append(block)
        end_line = start_line + block.count("\n") - 1
        ranges.append((tactic, start_line, end_line))
    return "".join(parts), ranges


CONTEXT_LINE = re.compile(
    r"^\s*(open|namespace|section|variable|local|set_option|attribute|notation|scoped|include|omit)\b"
)


def context_only_header(header: str) -> str:
    kept = [line for line in header.splitlines() if CONTEXT_LINE.match(line)]
    return "import Mathlib\n" + "\n".join(kept) + "\n"


def portability_source(record: dict) -> str:
    declaration = re.sub(r"\s*:=\s*by\s*$", "", record["decl_prefix"]).strip()
    keyword = DECL_KEYWORD.search(declaration)
    body = declaration[keyword.end():] if keyword else declaration
    name_match = re.match(r"\s*([\w.'!]+)", body)
    rest = body[name_match.end():] if name_match else body
    return (
        context_only_header(record["header"])
        + "set_option autoImplicit false\n"
        + f"example{rest} := by\n  sorry\n"
    )


def parse_error_lines(text: str) -> set[int]:
    lines: set[int] = set()
    for match in re.finditer(r"Candidate\.lean:(\d+):\d+: error:", text):
        lines.add(int(match.group(1)))
    return lines


def run_mode(args) -> int:
    records = [
        json.loads(line)
        for line in Path(args.statements).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [item for item in records if item["split"] == args.split][: args.goals]
    context = resolve_lean_context(ENVIRONMENTS[args.env])
    run_dir = Path(args.out).with_suffix("")
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, record in enumerate(selected):
        if args.mode == "portfolio":
            source, ranges = portfolio_source(record)
            timeout = args.timeout * len(TACTICS)
        else:
            source = portability_source(record)
            timeout = args.timeout
        workdir = run_dir / "scratch" / f"{index:02d}"
        run = run_lean(source, workdir, context, timeout, MEMORY_LIMIT)
        error_lines = parse_error_lines(run.stdout)
        entry = {
            "split": args.split,
            "env": args.env,
            "mode": args.mode,
            "full_name": record["full_name"],
            "statement_sha256": record["statement_sha256"],
            "status": run.status,
            "exit_code": run.exit_code,
            "wall_seconds": round(run.wall_seconds, 3),
            "peak_rss_bytes": run.peak_rss_bytes,
            "per_tactic": {},
        }
        if args.mode == "portfolio":
            for tactic, start_line, end_line in ranges:
                failed = any(start_line <= line <= end_line for line in error_lines)
                if run.status == "timeout":
                    verdict = "timeout"
                elif run.status != "ok" and not error_lines:
                    verdict = "file_error"
                else:
                    verdict = "error" if failed else "closed"
                entry["per_tactic"][tactic] = {"start_line": start_line, "verdict": verdict}
        else:
            entry["portable"] = run.status == "ok"
            first_error = next(
                (line for line in run.stdout.splitlines() if ": error:" in line),
                None,
            )
            entry["first_error"] = first_error
        results.append(entry)
        print(
            f"[{args.mode} {args.env}/{args.split}] {index + 1}/{len(selected)} "
            f"{record['full_name'][:45]} {entry['status']} {run.wall_seconds:.1f}s",
            flush=True,
        )
    if args.mode == "portfolio":
        by_tactic = {
            tactic: {
                "attempts": len(results),
                "closed": sum(1 for item in results if item["per_tactic"][tactic]["verdict"] == "closed"),
            }
            for tactic in TACTICS
        }
        goals_closed = sum(
            1
            for item in results
            if any(value["verdict"] == "closed" for value in item["per_tactic"].values())
        )
        summary = {
            "mode": args.mode,
            "env": args.env,
            "split": args.split,
            "goals": len(selected),
            "goals_closed_by_portfolio": goals_closed,
            "pass_at_1_by_tactic": {
                tactic: round(by_tactic[tactic]["closed"] / len(results), 4) if results else 0.0
                for tactic in TACTICS
            },
            "portfolio_pass_at_5": round(goals_closed / len(results), 4) if results else 0.0,
            "wall_total_seconds": round(sum(item["wall_seconds"] for item in results), 3),
            "peak_rss_max_bytes": max((item["peak_rss_bytes"] for item in results), default=0),
        }
    else:
        portable = sum(1 for item in results if item["portable"])
        summary = {
            "mode": args.mode,
            "env": args.env,
            "split": args.split,
            "goals": len(selected),
            "portable": portable,
            "portability_rate": round(portable / len(results), 4) if results else 0.0,
            "wall_total_seconds": round(sum(item["wall_seconds"] for item in results), 3),
        }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline simbólico/portabilidade LTP-03")
    parser.add_argument("--statements", required=True)
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", choices=["portfolio", "portability"], required=True)
    parser.add_argument("--goals", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    return run_mode(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
