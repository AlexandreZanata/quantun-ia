#!/usr/bin/env python3
"""Rechecagem do ReProver tacgen após a correção da extração (errata do LTP-03).

Reavalia as táticas geradas contra as declarações corrigidas (somente estilo
`:= by`) no ambiente nativo e reporta quantos casos do LTP-03 mudaram.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import (  # noqa: E402
    rename_declaration,
    resolve_lean_context,
    run_lean,
    scan_forbidden,
    strip_to_additive,
    with_options,
)

NATIVE_PROJECT = ROOT / "data" / "raw" / "mathlib4_src"
MEMORY_LIMIT = 8 * 1024**3
FORBIDDEN = ["sorry", "admit", "sorryAx"]
TARGET = "ltp_target"


def classify(run) -> str:
    if run.status == "timeout":
        return "timeout"
    if run.status == "memory_limit":
        return "memory_limit"
    if run.status == "ok":
        return "closed"
    if "unsolved goals" in run.stdout:
        return "valid_incomplete"
    return "invalid"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rechecagem ReProver LTP-04")
    parser.add_argument("--results", required=True)
    parser.add_argument("--statements", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))["results"]
    records = {
        (item["split"], item["full_name"]): item
        for item in (
            json.loads(line)
            for line in Path(args.statements).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    context = resolve_lean_context(NATIVE_PROJECT)
    root = Path(args.out).with_suffix("")
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    dropped = []
    for index, item in enumerate(results):
        record = records.get((item["split"], item["full_name"]))
        if record is None:
            dropped.append({"split": item["split"], "full_name": item["full_name"], "reason": "declaração term-mode excluída"})
            continue
        tactic = item["generated_tactic"]
        forbidden = scan_forbidden(tactic, FORBIDDEN)
        if forbidden:
            verdict = "spurious_sorry"
            run = None
        else:
            header = strip_to_additive(record["header"])
            if not header.endswith("\n"):
                header += "\n"
            options = ["set_option autoImplicit false"] if record["file_path"].startswith("Mathlib/") else []
            source = (
                with_options(header, options)
                + rename_declaration(record["decl_prefix"], TARGET)
                + "\n"
                + "\n".join("  " + line for line in tactic.splitlines())
                + "\n"
            )
            run = run_lean(source, root / "scratch" / f"{index:03d}", context, 60.0, MEMORY_LIMIT)
            verdict = classify(run)
        cases.append(
            {
                "split": item["split"],
                "full_name": item["full_name"],
                "original_verdict": item["verdict"],
                "corrected_verdict": verdict,
                "changed": item["verdict"] != verdict,
                "wall_seconds": round(run.wall_seconds, 3) if run else 0.0,
            }
        )
        print(f"[recheck {index + 1}/{len(results)}] {item['full_name'][:40]} {item['verdict']} -> {verdict}", flush=True)
    summary = {
        "cases": len(cases),
        "dropped_term_mode": len(dropped),
        "changed": sum(1 for item in cases if item["changed"]),
        "by_split": {
            split: {
                "goals": sum(1 for item in cases if item["split"] == split),
                "closed": sum(1 for item in cases if item["split"] == split and item["corrected_verdict"] == "closed"),
                "valid_incomplete": sum(
                    1 for item in cases if item["split"] == split and item["corrected_verdict"] == "valid_incomplete"
                ),
                "invalid": sum(1 for item in cases if item["split"] == split and item["corrected_verdict"] == "invalid"),
            }
            for split in ("random", "novel_premises")
        },
        "wall_seconds_total": round(sum(item["wall_seconds"] for item in cases), 3),
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "cases": cases, "dropped": dropped}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
