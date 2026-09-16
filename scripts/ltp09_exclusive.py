#!/usr/bin/env python3
"""LTP-09 — exclusividade por meta no subconjunto selado (sem reabrir o test).

Reexecuta as linhas de referência (simbólica e ReProver tacgen) no subconjunto
já extraído em `.local/sealed/LTP-09/` e registra por meta os resolvedores, sem
tocar `random/test.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import resolve_lean_context, run_lean, scan_forbidden, strip_to_additive, with_options  # noqa: E402

SEALED = ROOT / ".local" / "sealed" / "LTP-09"
NATIVE = ROOT / "data" / "raw" / "mathlib4_src"
MODEL = ROOT / "data" / "raw" / "reprover" / "leandojo-lean4-tacgen-byt5-small" / "67a2c53cc36186fe8539d0a342fc42c50edc68fd"
TACTICS = ["simp", "aesop", "omega", "linarith", "nlinarith"]
TIMEOUT = 60.0
MEMORY = 8 * 1024**3


def prepare(entry: dict) -> str:
    header = strip_to_additive(entry["header"])
    if not header.endswith("\n"):
        header += "\n"
    options = ["set_option autoImplicit false"] if entry["file_path"].startswith("Mathlib/") else []
    return with_options(header, options) + entry["decl_prefix"]


def main() -> int:
    parser = argparse.ArgumentParser(description="LTP-09 exclusividade")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    subset = json.loads((SEALED / "test_subset.json").read_text(encoding="utf-8"))
    context = resolve_lean_context(NATIVE)
    symbolic = []
    for index, entry in enumerate(subset["entries"]):
        base = prepare(entry)
        solved = None
        for tactic in TACTICS:
            run = run_lean(base + f"\n  {tactic}\n", SEALED / "exclusive_symbolic" / f"{index:02d}_{tactic}", context, TIMEOUT, MEMORY)
            if run.status == "ok":
                solved = tactic
                break
        symbolic.append(solved)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL)).eval()
    tacgen = []
    for index, entry in enumerate(subset["entries"]):
        inputs = tokenizer(entry["first_state"], max_length=2048, truncation=True, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=512, num_beams=1, do_sample=False, num_return_sequences=1)
        tactic = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].replace("<a>", "").replace("</a>", "")
        if scan_forbidden(tactic, ["sorry", "admit", "sorryAx"]):
            tacgen.append(False)
            continue
        run = run_lean(prepare(entry) + "\n" + "\n".join("  " + line for line in tactic.splitlines()) + "\n", SEALED / "exclusive_tacgen" / f"{index:02d}", context, TIMEOUT, MEMORY)
        tacgen.append(run.status == "ok")
    ids = [entry["full_name"] for entry in subset["entries"]]
    symbolic_solved = {ids[i] for i, value in enumerate(symbolic) if value}
    tacgen_solved = {ids[i] for i, value in enumerate(tacgen) if value}
    payload = {
        "goals": len(ids),
        "symbolic": {"solved": len(symbolic_solved), "per_goal": dict(zip(ids, [bool(value) for value in symbolic]))},
        "tacgen": {"solved": len(tacgen_solved), "per_goal": dict(zip(ids, tacgen))},
        "exclusive_symbolic": sorted(symbolic_solved - tacgen_solved),
        "exclusive_tacgen": sorted(tacgen_solved - symbolic_solved),
        "solved_by_both": sorted(symbolic_solved & tacgen_solved),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("goals", "exclusive_symbolic", "exclusive_tacgen", "solved_by_both")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
