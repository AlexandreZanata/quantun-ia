#!/usr/bin/env python3
"""T2-00 — avaliação pareada nano vs ReProver no val (mesmos itens e protocolo).

Gera a primeira tática com o nano treinado e com o ReProver tacgen, verifica as
duas no Lean nativo, aplica McNemar exato, bootstrap pareado e compara memória
de pesos. O test selado não é tocado; só seria aberto se o gate passasse.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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


def classify(run) -> str:
    if run.status == "timeout":
        return "timeout"
    if run.status == "ok":
        return "closed"
    if "unsolved goals" in run.stdout:
        return "valid_incomplete"
    return "invalid"


def verify(entry: dict, tactic: str, context, label: str) -> dict:
    if scan_forbidden(tactic, FORBIDDEN):
        return {"verdict": "spurious_sorry", "wall": 0.0}
    header = strip_to_additive(entry["header"])
    if not header.endswith("\n"):
        header += "\n"
    options = ["set_option autoImplicit false"] if entry["file_path"].startswith("Mathlib/") else []
    source = (
        with_options(header, options)
        + entry["decl_prefix"]
        + "\n"
        + "\n".join("  " + line for line in tactic.splitlines())
        + "\n"
    )
    run = run_lean(source, ROOT / ".local" / "runs" / "lean" / "T2-00" / label, context, TIMEOUT, MEMORY)
    return {"verdict": classify(run), "wall": round(run.wall_seconds, 3)}


def generate(model, tokenizer, state: str, device: str, max_new: int) -> tuple[str, float]:
    inputs = tokenizer(state, max_length=1024, truncation=True, return_tensors="pt").to(device)
    started = time.monotonic()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new, num_beams=1, do_sample=False)
    tactic = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].replace("<a>", "").replace("</a>", "")
    return tactic, time.monotonic() - started


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_diff(values: np.ndarray, rng: np.random.Generator, resamples: int = 10000) -> tuple[float, float]:
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def main() -> int:
    parser = argparse.ArgumentParser(description="T2-00 avaliação")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--items", default=str(ROOT / "research" / "lean" / "runs" / "LTP-04" / "statements_v2.jsonl"))
    parser.add_argument("--per-split", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nano_tokenizer = AutoTokenizer.from_pretrained(str(REPROVER))
    nano = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device).eval()
    nano_parameters = sum(parameter.numel() for parameter in nano.parameters())
    nano_bytes = sum(parameter.numel() * parameter.element_size() for parameter in nano.parameters())
    reprover_tokenizer = AutoTokenizer.from_pretrained(str(REPROVER))
    reprover = AutoModelForSeq2SeqLM.from_pretrained(str(REPROVER)).to(device).eval()
    reprover_parameters = sum(parameter.numel() for parameter in reprover.parameters())
    reprover_bytes = sum(parameter.numel() * parameter.element_size() for parameter in reprover.parameters())
    records = [json.loads(line) for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for split in ("random", "novel_premises"):
        selected.extend([item for item in records if item["split"] == split][: args.per_split])
    context = resolve_lean_context(NATIVE)
    items = []
    calls = 0
    walls = {"nano": 0.0, "reprover": 0.0}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, entry in enumerate(selected):
        nano_tactic, nano_gen = generate(nano, nano_tokenizer, entry["first_state"], device, args.max_new)
        reprover_tactic, reprover_gen = generate(reprover, reprover_tokenizer, entry["first_state"], device, args.max_new)
        nano_result = verify(entry, nano_tactic, context, f"nano_{index:03d}")
        reprover_result = verify(entry, reprover_tactic, context, f"reprover_{index:03d}")
        calls += int(nano_result["verdict"] != "spurious_sorry") + int(reprover_result["verdict"] != "spurious_sorry")
        walls["nano"] += nano_gen + nano_result["wall"]
        walls["reprover"] += reprover_gen + reprover_result["wall"]
        items.append(
            {
                "split": entry["split"],
                "full_name": entry["full_name"],
                "nano_tactic": nano_tactic,
                "reprover_tactic": reprover_tactic,
                "nano": nano_result["verdict"],
                "reprover": reprover_result["verdict"],
            }
        )
        print(
            f"[{index + 1}/{len(selected)}] {entry['full_name'][:40]} nano={nano_result['verdict']} "
            f"reprover={reprover_result['verdict']}",
            flush=True,
        )
    nano_closed = [item["nano"] == "closed" for item in items]
    reprover_closed = [item["reprover"] == "closed" for item in items]
    nano_n = sum(nano_closed)
    reprover_n = sum(reprover_closed)
    b = sum(1 for nano_ok, reprover_ok in zip(nano_closed, reprover_closed) if nano_ok and not reprover_ok)
    c = sum(1 for nano_ok, reprover_ok in zip(nano_closed, reprover_closed) if reprover_ok and not nano_ok)
    ratio = nano_n / reprover_n if reprover_n else None
    rng = np.random.default_rng(0)
    difference = np.array([1.0 if nano_ok else 0.0 for nano_ok in nano_closed]) - np.array(
        [1.0 if reprover_ok else 0.0 for reprover_ok in reprover_closed]
    )
    gate_ratio = ratio is not None and ratio >= 0.85
    gate_memory = nano_bytes <= reprover_bytes / 3
    summary = {
        "phase": "T2-00",
        "device": device,
        "items": len(items),
        "nano": {
            "parameters": nano_parameters,
            "weights_bytes": nano_bytes,
            "closed": nano_n,
            "valid_incomplete": sum(1 for item in items if item["nano"] == "valid_incomplete"),
            "invalid": sum(1 for item in items if item["nano"] == "invalid"),
            "spurious_sorry": sum(1 for item in items if item["nano"] == "spurious_sorry"),
        },
        "reprover": {
            "parameters": reprover_parameters,
            "weights_bytes": reprover_bytes,
            "closed": reprover_n,
            "valid_incomplete": sum(1 for item in items if item["reprover"] == "valid_incomplete"),
            "invalid": sum(1 for item in items if item["reprover"] == "invalid"),
        },
        "paired": {
            "nano_only": b,
            "reprover_only": c,
            "mcnemar_p": mcnemar(b, c),
            "difference_ci": bootstrap_diff(difference, rng),
        },
        "ratio_nano_over_reprover": ratio,
        "gate": {
            "ratio_ok": gate_ratio,
            "memory_ok": gate_memory,
            "passed": bool(gate_ratio and gate_memory),
        },
        "verifier_calls": calls,
        "wall_seconds": {"nano": round(walls["nano"], 3), "reprover": round(walls["reprover"], 3)},
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "items": items}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
