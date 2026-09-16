#!/usr/bin/env python3
"""LTP-10 — generalização em ProofNet Lean 4 e miniCTX v2 (mathlib).

Abre cada jsonl selado uma única vez, congela subconjuntos determinísticos em
`.local/sealed/LTP-10/`, mede portabilidade no ambiente nativo e roda as linhas
de referência (simbólica e ReProver tacgen) no subconjunto portável.
PutnamBench não é adquirido (sem licença); projetos não-mathlib do miniCTX ficam
bloqueados por exigirem ambientes próprios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import resolve_lean_context, run_lean, scan_forbidden  # noqa: E402
from scripts.ltp05_canonical_bytes import normalize_state  # noqa: E402

PROOFNET = ROOT / "data" / "raw" / "proofnet4" / "ProofNet-lean4-6deae98b3895" / "proofnet_lean4.jsonl"
MINICTX = ROOT / "data" / "raw" / "minictx2" / "minictx-test" / "mathlib.jsonl"
SEALED = ROOT / ".local" / "sealed" / "LTP-10"
NATIVE = ROOT / "data" / "raw" / "mathlib4_src"
MODEL = ROOT / "data" / "raw" / "reprover" / "leandojo-lean4-tacgen-byt5-small" / "67a2c53cc36186fe8539d0a342fc42c50edc68fd"
PROOFNET_SAMPLE = 24
PROOFNET_CAP = 12
MINICTX_SAMPLE = 12
MINICTX_CAP = 6
TIMEOUT = 60.0
MEMORY = 8 * 1024**3
TACTICS = ["simp", "aesop", "omega", "linarith", "nlinarith"]
MARKER = "LTP10_STATE_"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def devision(declaration: str) -> str | None:
    index = declaration.rfind(":=")
    if index == -1 or "sorry" not in declaration[index:]:
        return None
    return declaration[:index]


def open_once() -> dict:
    SEALED.mkdir(parents=True, exist_ok=True)
    opening_path = SEALED / "opening.json"
    if opening_path.exists():
        return json.loads(opening_path.read_text(encoding="utf-8"))
    proofnet_rows = [json.loads(line) for line in PROOFNET.read_text(encoding="utf-8").splitlines() if line.strip()]
    proofnet_entries = []
    for row in proofnet_rows:
        prefix = devision(row["formal_statement"])
        if prefix is None:
            continue
        proofnet_entries.append(
            {
                "name": row["name"],
                "header": row["header"],
                "decl_prefix": prefix,
                "statement_sha256": hashlib.sha256(normalize_state(prefix).encode("utf-8")).hexdigest(),
            }
        )
        if len(proofnet_entries) >= PROOFNET_SAMPLE:
            break
    minictx_rows = [json.loads(line) for line in MINICTX.read_text(encoding="utf-8").splitlines() if line.strip()]
    minictx_entries = [
        {
            "name": row["theoremName"],
            "header": row["srcContext"],
            "statement": row["theoremStatement"],
            "statement_sha256": hashlib.sha256(normalize_state(row["theoremStatement"]).encode("utf-8")).hexdigest(),
        }
        for row in minictx_rows[:MINICTX_SAMPLE]
    ]
    (SEALED / "proofnet_subset.json").write_text(json.dumps(proofnet_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (SEALED / "minictx_mathlib_subset.json").write_text(json.dumps(minictx_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    opening = {
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "proofnet_file_sha256": sha256_file(PROOFNET),
        "proofnet_entries": len(proofnet_entries),
        "minictx_file_sha256": sha256_file(MINICTX),
        "minictx_entries": len(minictx_entries),
        "proofnet_subset_sha256": sha256_file(SEALED / "proofnet_subset.json"),
        "minictx_subset_sha256": sha256_file(SEALED / "minictx_mathlib_subset.json"),
        "putnambench": "não adquirido: sem licença (stress test reservado)",
        "minictx_other_projects": "bloqueado: ambientes por projeto necessários",
    }
    opening_path.write_text(json.dumps(opening, indent=2) + "\n", encoding="utf-8")
    return opening


def portable(entry: dict, mode: str, context, label: str) -> bool:
    if mode == "proofnet":
        source = entry["header"] + ("\n" if not entry["header"].endswith("\n") else "") + entry["decl_prefix"] + " := by\n  sorry\n"
    else:
        source = entry["header"] + ("\n" if not entry["header"].endswith("\n") else "") + entry["statement"] + " := by\n  sorry\n"
    run = run_lean(source, SEALED / label, context, TIMEOUT, MEMORY)
    return run.status == "ok"


def symbolic(entry: dict, mode: str, context, label: str) -> tuple[bool, int, float]:
    if mode == "proofnet":
        base = entry["header"] + ("\n" if not entry["header"].endswith("\n") else "") + entry["decl_prefix"] + " := by"
    else:
        base = entry["header"] + ("\n" if not entry["header"].endswith("\n") else "") + entry["statement"] + " := by"
    calls = 0
    wall = 0.0
    for tactic in TACTICS:
        calls += 1
        run = run_lean(base + f"\n  {tactic}\n", SEALED / f"{label}_{tactic}", context, TIMEOUT, MEMORY)
        wall += run.wall_seconds
        if run.status == "ok":
            return True, calls, wall
    return False, calls, wall


def generate_states(entries: list[dict], mode: str, context) -> dict[int, str]:
    parts = ["import Mathlib", "set_option linter.unusedTactic false"]
    for index, entry in enumerate(entries):
        if mode == "proofnet":
            statement = entry["decl_prefix"]
            match = re.match(r"\s*(theorem|lemma|instance|def)\s+[\w.'!?«»]+\s*", statement)
            body = statement[match.end():] if match else statement
            parts.append(f"example{body if body.startswith((' ', '(')) else ' ' + body} := by")
        else:
            statement = entry["statement"]
            colon = statement.find(":")
            parts.append(f"example{statement[colon:]} := by")
        parts.append("  trace_state")
        parts.append(f'  dbg_trace "{MARKER}{index}"')
        parts.append("  sorry")
    run = run_lean("\n".join(parts) + "\n", SEALED / "state_gen", context, 300.0, MEMORY)
    markers = [
        int(match.group(1))
        for line in run.stderr.splitlines()
        if (match := re.search(rf"{MARKER}(\d+)", line))
    ]
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in run.stdout.splitlines():
        if line.startswith("/") or line.startswith("warning:") or line.startswith("Note:"):
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return {
        markers[index]: "\n".join(block).strip()
        for index, block in enumerate(blocks)
        if index < len(markers)
    }


def tacgen(entries: list[dict], states: dict[int, str], mode: str, context) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL)).eval()
    solved = 0
    calls = 0
    wall = 0.0
    per_goal = []
    for index, entry in enumerate(entries):
        state = states.get(index)
        if not state:
            per_goal.append(False)
            continue
        inputs = tokenizer(state, max_length=2048, truncation=True, return_tensors="pt")
        started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=512, num_beams=1, do_sample=False, num_return_sequences=1)
        tactic = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].replace("<a>", "").replace("</a>", "")
        if scan_forbidden(tactic, ["sorry", "admit", "sorryAx"]):
            per_goal.append(False)
            continue
        if mode == "proofnet":
            base = entry["header"] + ("\n" if not entry["header"].endswith("\n") else "") + entry["decl_prefix"] + " := by\n"
        else:
            base = entry["header"] + ("\n" if not entry["header"].endswith("\n") else "") + entry["statement"] + " := by\n"
        run = run_lean(base + "\n".join("  " + line for line in tactic.splitlines()) + "\n", SEALED / f"tacgen_{index:03d}", context, TIMEOUT, MEMORY)
        wall += time.monotonic() - started
        calls += 1
        ok = run.status == "ok"
        solved += int(ok)
        per_goal.append(ok)
    return {"pass_at_1": solved, "verifier_calls": calls, "wall_seconds": round(wall, 3), "per_goal": per_goal}


def evaluate(subset_name: str, mode: str, cap: int, context) -> dict:
    entries = json.loads((SEALED / subset_name).read_text(encoding="utf-8"))
    flags = [portable(entry, mode, context, f"port_{mode}_{index:02d}") for index, entry in enumerate(entries)]
    portable_entries = [entry for entry, ok in zip(entries, flags) if ok][:cap]
    result = {
        "sampled": len(entries),
        "portable": sum(1 for ok in flags if ok),
        "portability_rate": round(sum(1 for ok in flags if ok) / len(entries), 4) if entries else 0.0,
        "reference_goals": len(portable_entries),
    }
    if not portable_entries:
        result["symbolic"] = None
        result["tacgen"] = None
        return result
    solved = []
    calls = 0
    wall = 0.0
    for index, entry in enumerate(portable_entries):
        ok, goal_calls, goal_wall = symbolic(entry, mode, context, f"{mode}_sym_{index:02d}")
        solved.append(ok)
        calls += goal_calls
        wall += goal_wall
    result["symbolic"] = {
        "portfolio_pass_at_5": sum(1 for ok in solved if ok),
        "verifier_calls": calls,
        "wall_seconds": round(wall, 3),
    }
    states = generate_states(portable_entries, mode, context)
    result["tacgen"] = tacgen(portable_entries, states, mode, context)
    exclusive_symbolic = [
        portable_entries[i]["name"] for i, ok in enumerate(solved) if ok and not result["tacgen"]["per_goal"][i]
    ]
    exclusive_tacgen = [
        portable_entries[i]["name"] for i, ok in enumerate(result["tacgen"]["per_goal"]) if ok and not solved[i]
    ]
    result["exclusive_symbolic"] = exclusive_symbolic
    result["exclusive_tacgen"] = exclusive_tacgen
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="LTP-10 generalização")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    opening = open_once()
    context = resolve_lean_context(NATIVE)
    started = time.monotonic()
    results = {
        "opening": opening,
        "proofnet": evaluate("proofnet_subset.json", "proofnet", PROOFNET_CAP, context),
        "minictx_mathlib": evaluate("minictx_mathlib_subset.json", "minictx", MINICTX_CAP, context),
    }
    results["wall_seconds"] = round(time.monotonic() - started, 3)
    results["peak_rss_bytes"] = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024
    results["program_final"] = {
        "funnel": "100 hipóteses → 8 confirmadas (Holm) → 3 finalistas no teste selado",
        "sealed_benchmark4": "LH-070 16/16, LH-069 16/16, LH-026 12,5%",
        "generalization": "ProofNet e miniCTX(mathlib) avaliados com portabilidade medida; projetos externos e PutnamBench bloqueados",
        "no_trained_prover": True,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
