#!/usr/bin/env python3
"""Tentativa de reprodução do ReProver tacgen sem recuperação (LTP-03).

Geração greedy com o modelo pinado e verificação da tática gerada no ambiente
nativo do LeanDojo Benchmark 4. A prova nunca é aceita com `sorry`; a métrica
principal é a validade formal no Lean.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import resolve_lean_context, run_lean, scan_forbidden  # noqa: E402

NATIVE_PROJECT = ROOT / "data" / "raw" / "mathlib4_src"
MEMORY_LIMIT = 8 * 1024**3
FORBIDDEN = ["sorry", "admit", "sorryAx"]


def remove_marks(text: str) -> str:
    return text.replace("<a>", "").replace("</a>", "")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
    parser = argparse.ArgumentParser(description="ReProver tacgen LTP-03")
    parser.add_argument("--statements", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--goals", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["random", "novel_premises"])
    parser.add_argument("--max-input", type=int, default=2048)
    parser.add_argument("--max-output", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    context = resolve_lean_context(NATIVE_PROJECT)
    records = [
        json.loads(line)
        for line in Path(args.statements).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir = Path(args.out).with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, record in enumerate(records):
        if record["split"] not in args.splits:
            continue
        split_index = sum(1 for item in results if item["split"] == record["split"])
        if split_index >= args.goals:
            continue
        inputs = tokenizer(record["first_state"], max_length=args.max_input, truncation=True, return_tensors="pt")
        started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=args.max_output,
                num_beams=1,
                do_sample=False,
                num_return_sequences=1,
                early_stopping=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        generation_seconds = time.monotonic() - started
        tactic = remove_marks(tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0])
        forbidden = scan_forbidden(tactic, FORBIDDEN)
        source = (
            record["header"]
            + ("\n" if not record["header"].endswith("\n") else "")
            + "set_option autoImplicit false\n"
            + record["decl_prefix"]
            + "\n"
            + "\n".join("  " + line for line in tactic.splitlines())
            + "\n"
        )
        verdict = "spurious_sorry" if forbidden else None
        run = None
        if verdict is None:
            run = run_lean(source, output_dir / "scratch" / f"{index:02d}", context, args.timeout, MEMORY_LIMIT)
            verdict = classify(run)
        entry = {
            "split": record["split"],
            "full_name": record["full_name"],
            "statement_sha256": record["statement_sha256"],
            "human_tactic": record["first_tactic"],
            "generated_tactic": tactic,
            "generated_tokens": int(generated.sequences.shape[1]),
            "exact_match": normalize_text(tactic) == normalize_text(record["first_tactic"]),
            "prefix_match": normalize_text(tactic).startswith(normalize_text(record["first_tactic"])),
            "forbidden_tokens": forbidden,
            "verdict": verdict,
            "generation_seconds": round(generation_seconds, 3),
            "verification_seconds": round(run.wall_seconds, 3) if run else None,
            "exit_code": run.exit_code if run else None,
        }
        results.append(entry)
        print(
            f"[reprover {record['split']}] {split_index + 1}/{args.goals} "
            f"{record['full_name'][:40]} gen={generation_seconds:.0f}s -> {verdict}",
            flush=True,
        )
    summary = {
        "model": args.model,
        "parameters": parameters,
        "goals_per_split": args.goals,
        "decoding": {
            "num_beams": 1,
            "max_input": args.max_input,
            "max_output": args.max_output,
        },
        "by_split": {
            split: {
                "goals": sum(1 for item in results if item["split"] == split),
                "exact_match": sum(1 for item in results if item["split"] == split and item["exact_match"]),
                "prefix_match": sum(1 for item in results if item["split"] == split and item["prefix_match"]),
                "closed": sum(1 for item in results if item["split"] == split and item["verdict"] == "closed"),
                "valid_incomplete": sum(
                    1 for item in results if item["split"] == split and item["verdict"] == "valid_incomplete"
                ),
                "invalid": sum(1 for item in results if item["split"] == split and item["verdict"] == "invalid"),
                "timeout": sum(1 for item in results if item["split"] == split and item["verdict"] == "timeout"),
                "spurious_sorry": sum(
                    1 for item in results if item["split"] == split and item["verdict"] == "spurious_sorry"
                ),
            }
            for split in args.splits
        },
        "generation_seconds_total": round(sum(item["generation_seconds"] for item in results), 3),
        "verification_seconds_total": round(
            sum(item["verification_seconds"] or 0.0 for item in results), 3
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
