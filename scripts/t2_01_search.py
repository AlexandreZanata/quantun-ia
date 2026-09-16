#!/usr/bin/env python3
"""T2-01 — endpoint B: busca best-first pareada nano vs ReProver.

Ambos os sistemas recebem o mesmo orçamento de chamadas ao verificador por item
e a mesma geração k=4. Um nó é o arquivo com a sequência de táticas; "unsolved
goals" fornece o estado para expandir (primeira meta), profundidade <= 2.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import resolve_lean_context, run_lean, scan_forbidden, strip_to_additive, with_options  # noqa: E402

NATIVE = ROOT / "data" / "raw" / "mathlib4_src"
REPROVER = ROOT / "data" / "raw" / "reprover" / "leandojo-lean4-tacgen-byt5-small" / "67a2c53cc36186fe8539d0a342fc42c50edc68fd"
TIMEOUT = 60.0
MEMORY = 8 * 1024**3
FORBIDDEN = ["sorry", "admit", "sorryAx"]
DIAGNOSTIC = re.compile(r"^/.*:\d+:\d+: (error|warning):")
STATE_START = re.compile(r"^[^\s].*:.*$|^⊢")


def run_system(entry: dict, sequence: list[str], context, label: str) -> dict:
    header = strip_to_additive(entry["header"])
    if not header.endswith("\n"):
        header += "\n"
    options = ["set_option autoImplicit false"] if entry["file_path"].startswith("Mathlib/") else []
    body = "\n".join("  " + line for line in sequence for line in line.splitlines())
    source = with_options(header, options) + entry["decl_prefix"] + "\n" + body + "\n"
    run = run_lean(source, ROOT / ".local" / "runs" / "lean" / "T2-01" / label, context, TIMEOUT, MEMORY)
    return {"status": run.status, "stdout": run.stdout, "wall": run.wall_seconds}


def extract_state(stdout: str) -> str | None:
    lines = stdout.splitlines()
    start = None
    for index, line in enumerate(lines):
        if "error: unsolved goals" in line:
            start = index + 1
            break
    if start is None:
        return None
    block = []
    for line in lines[start:]:
        if DIAGNOSTIC.match(line):
            break
        block.append(line)
    text = "\n".join(block).strip()
    return text or None


def generate(model, tokenizer, state: str, device: str, k: int) -> list[str]:
    inputs = tokenizer(state, max_length=1024, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=48, num_beams=k, num_return_sequences=k, do_sample=False
        )
    tactics = [tokenizer.batch_decode(row, skip_special_tokens=True) for row in generated]
    tactics = [tactic.replace("<a>", "").replace("</a>", "").strip() for tactic in tactics]
    unique = []
    for tactic in tactics:
        if tactic and tactic not in unique:
            unique.append(tactic)
    return unique


def search(entry: dict, model, tokenizer, device: str, context, budget: int, k: int, label: str) -> dict:
    calls = 0
    wall = 0.0
    initial = entry["first_state"]
    queue: deque[tuple[list[str], int, str]] = deque()
    for tactic in generate(model, tokenizer, initial, device, k)[:k]:
        queue.append(([tactic], 1, tactic))
    while queue and calls < budget:
        sequence, depth, _ = queue.popleft()
        result = run_system(entry, sequence, context, f"{label}_c{calls:03d}")
        calls += 1
        wall += result["wall"]
        if result["status"] == "ok":
            return {"closed": True, "calls": calls, "wall": round(wall, 3), "sequence": sequence}
        if any(scan_forbidden(tactic, FORBIDDEN) for tactic in sequence):
            continue
        state = extract_state(result["stdout"])
        if state and depth < 2 and calls < budget:
            for tactic in generate(model, tokenizer, state, device, k)[:k]:
                queue.append((sequence + [tactic], depth + 1, tactic))
    return {"closed": False, "calls": calls, "wall": round(wall, 3), "sequence": None}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> int:
    parser = argparse.ArgumentParser(description="T2-01 busca best-first")
    parser.add_argument("--nano-dir", required=True)
    parser.add_argument("--items", default=str(ROOT / "research" / "lean" / "runs" / "LTP-04" / "statements_v2.jsonl"))
    parser.add_argument("--per-split", type=int, default=16)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nano_tokenizer = AutoTokenizer.from_pretrained(str(REPROVER))
    nano = AutoModelForSeq2SeqLM.from_pretrained(args.nano_dir).to(device).eval()
    nano_params = sum(parameter.numel() for parameter in nano.parameters())
    nano_bytes = sum(parameter.numel() * parameter.element_size() for parameter in nano.parameters())
    reprover_tokenizer = AutoTokenizer.from_pretrained(str(REPROVER))
    reprover = AutoModelForSeq2SeqLM.from_pretrained(str(REPROVER)).to(device).eval()
    reprover_params = sum(parameter.numel() for parameter in reprover.parameters())
    reprover_bytes = sum(parameter.numel() * parameter.element_size() for parameter in reprover.parameters())
    records = [json.loads(line) for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for split in ("random", "novel_premises"):
        selected.extend([item for item in records if item["split"] == split][: args.per_split])
    if args.max_items:
        selected = selected[: args.max_items]
    context = resolve_lean_context(NATIVE)
    results = []
    for index, entry in enumerate(selected):
        nano_result = search(entry, nano, nano_tokenizer, device, context, args.budget, args.k, f"nano_{index:03d}")
        reprover_result = search(entry, reprover, reprover_tokenizer, device, context, args.budget, args.k, f"reprover_{index:03d}")
        results.append(
            {
                "split": entry["split"],
                "full_name": entry["full_name"],
                "nano": nano_result,
                "reprover": reprover_result,
            }
        )
        print(
            f"[{index + 1}/{len(selected)}] {entry['full_name'][:40]} "
            f"nano={nano_result['closed']}({nano_result['calls']}) reprover={reprover_result['closed']}({reprover_result['calls']})",
            flush=True,
        )
    nano_closed = [item["nano"]["closed"] for item in results]
    reprover_closed = [item["reprover"]["closed"] for item in results]
    nano_n = sum(nano_closed)
    reprover_n = sum(reprover_closed)
    b = sum(1 for left, right in zip(nano_closed, reprover_closed) if left and not right)
    c = sum(1 for left, right in zip(nano_closed, reprover_closed) if right and not left)
    difference = np.array([1.0 if left else 0.0 for left in nano_closed]) - np.array(
        [1.0 if right else 0.0 for right in reprover_closed]
    )
    rng = np.random.default_rng(0)
    draws = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
    ratio = nano_n / reprover_n if reprover_n else None
    summary = {
        "phase": "T2-01",
        "endpoint": "B_best_first",
        "items": len(results),
        "budget_calls_per_item": args.budget,
        "k": args.k,
        "device": device,
        "nano": {"parameters": nano_params, "weights_bytes": nano_bytes, "closed": nano_n},
        "reprover": {"parameters": reprover_params, "weights_bytes": reprover_bytes, "closed": reprover_n},
        "paired": {
            "nano_only": b,
            "reprover_only": c,
            "mcnemar_p": mcnemar(b, c),
            "difference_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        },
        "ratio_nano_over_reprover": ratio,
        "calls": {
            "nano": sum(item["nano"]["calls"] for item in results),
            "reprover": sum(item["reprover"]["calls"] for item in results),
        },
        "wall_seconds": {
            "nano": round(sum(item["nano"]["wall"] for item in results), 3),
            "reprover": round(sum(item["reprover"]["wall"] for item in results), 3),
        },
        "gate": {
            "ratio_ok": ratio is not None and ratio >= 0.85,
            "memory_ok": nano_bytes <= reprover_bytes / 3,
            "passed": bool(ratio is not None and ratio >= 0.85 and nano_bytes <= reprover_bytes / 3),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "items": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
