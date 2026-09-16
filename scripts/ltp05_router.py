#!/usr/bin/env python3
"""LH-093 — Roteador simbólico-neural.

Política: tentar táticas simbólicas primeiro e escalar para o gerador neural
(ReProver tacgen) apenas nas metas não resolvidas. Mede custo e eficiência
contra o modelo direto nos conjuntos: smoke32 (triviais), val (difíceis) e união.

Subcomandos:
  smoke-states  gera estados iniciais do smoke32 via Lean (trace_state)
  smoke-model   gera e verifica táticas do ReProver para o smoke32
  evaluate      compara políticas com custos medidos nos artefatos
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import resolve_lean_context, run_lean, scan_forbidden  # noqa: E402

SMOKE = ROOT / "research" / "lean" / "smoke" / "smoke32.json"
MODEL = ROOT / "data" / "raw" / "reprover" / "leandojo-lean4-tacgen-byt5-small" / "67a2c53cc36186fe8539d0a342fc42c50edc68fd"
MODERN = ROOT / "lean"
NATIVE = ROOT / "data" / "raw" / "mathlib4_src"
RUNS = ROOT / "research" / "lean" / "runs"
MEMORY_LIMIT = 8 * 1024**3
FORBIDDEN = ["sorry", "admit", "sorryAx"]
MARKER = "LTP_STATE_"


def smoke_states(args: argparse.Namespace) -> int:
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    parts = [
        "import Mathlib.Tactic",
        "set_option autoImplicit false",
        "set_option linter.unusedTactic false",
    ]
    for index, theorem in enumerate(smoke["theorems"]):
        parts.append(f"example : {theorem['statement']} := by")
        parts.append("  trace_state")
        parts.append(f'  dbg_trace "{MARKER}{index}"')
        parts.append("  sorry")
    context = resolve_lean_context(MODERN)
    run = run_lean("\n".join(parts) + "\n", Path(args.workdir), context, 300.0, MEMORY_LIMIT)
    stdout_markers: list[str] = []
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in run.stdout.splitlines():
        match = re.search(rf"{MARKER}(\d+)", line)
        if match:
            if current:
                blocks.append(current)
                current = []
            if line.strip() == f"{MARKER}{match.group(1)}":
                stdout_markers.append(match.group(1))
            continue
        if line.startswith("/") or line.startswith("warning:") or line.startswith("Note:"):
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    markers = stdout_markers or [
        match.group(1)
        for line in run.stderr.splitlines()
        if (match := re.search(rf"{MARKER}(\d+)", line))
    ]
    states = {
        markers[index]: "\n".join(block).strip()
        for index, block in enumerate(blocks)
        if index < len(markers)
    }
    payload = {
        "status": run.status,
        "states": states,
        "wall_seconds": round(run.wall_seconds, 3),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"estados extraídos: {len(states)}/{len(smoke['theorems'])} (status {run.status})")
    return 0


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


def smoke_model(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    states = json.loads(Path(args.states).read_text(encoding="utf-8"))["states"]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL)).eval()
    context = resolve_lean_context(MODERN)
    results = []
    for index, theorem in enumerate(smoke["theorems"]):
        state = states.get(str(index))
        if state is None:
            continue
        inputs = tokenizer(state, max_length=2048, truncation=True, return_tensors="pt")
        started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=512,
                num_beams=1,
                do_sample=False,
                num_return_sequences=1,
                early_stopping=False,
            )
        generation_seconds = time.monotonic() - started
        tactic = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        tactic = tactic.replace("<a>", "").replace("</a>", "")
        forbidden = scan_forbidden(tactic, FORBIDDEN)
        if forbidden:
            verdict = "spurious_sorry"
            wall = None
        else:
            source = (
                "import Mathlib.Tactic\n"
                "set_option autoImplicit false\n"
                f"theorem ltp_smoke_{index} : {theorem['statement']} := by\n"
                + "\n".join("  " + line for line in tactic.splitlines())
                + "\n"
            )
            run = run_lean(source, Path(args.workdir) / f"case_{index:03d}", context, 60.0, MEMORY_LIMIT)
            verdict = classify(run)
            wall = round(run.wall_seconds, 3)
        results.append(
            {
                "id": theorem["id"],
                "statement_sha256": theorem["statement_sha256"],
                "generated_tactic": tactic,
                "verdict": verdict,
                "generation_seconds": round(generation_seconds, 3),
                "verification_seconds": wall,
            }
        )
        print(f"[LH-093 smoke-model] {index + 1}/{len(smoke['theorems'])} {theorem['id']} -> {verdict}", flush=True)
    summary = {
        "goals": len(results),
        "closed": sum(1 for item in results if item["verdict"] == "closed"),
        "valid_incomplete": sum(1 for item in results if item["verdict"] == "valid_incomplete"),
        "invalid": sum(1 for item in results if item["verdict"] == "invalid"),
        "generation_seconds_total": round(sum(item["generation_seconds"] for item in results), 3),
        "verification_seconds_total": round(sum(item["verification_seconds"] or 0.0 for item in results), 3),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    ltp00 = json.loads((RUNS / "LTP-00" / "results.json").read_text(encoding="utf-8"))
    unique_accept: dict[str, dict] = {}
    for item in ltp00:
        if item["expectation"] == "accept" and item["wall_seconds"]:
            unique_accept.setdefault(item["statement_sha256"], item)
    trivial_walls = [item["wall_seconds"] for item in unique_accept.values()]
    smoke_model_data = json.loads(Path(args.smoke_model).read_text(encoding="utf-8"))
    hard_symbolic = []
    for name in ("symbolic_single_random.json", "symbolic_single_novel_premises.json"):
        data = json.loads((RUNS / "LTP-04" / name).read_text(encoding="utf-8"))
        for item in data["results"]:
            hard_symbolic.append(
                {
                    "full_name": item["full_name"],
                    "solved": any(value["verdict"] == "closed" for value in item["per_tactic"].values()),
                    "cost": item["wall_seconds"],
                }
            )
    tacgen = json.loads((RUNS / "LTP-03" / "reprover_tacgen.json").read_text(encoding="utf-8"))["results"]
    recheck = json.loads((RUNS / "LTP-04" / "reprover_recheck.json").read_text(encoding="utf-8"))["cases"]
    recheck_map = {(item["split"], item["full_name"]): item["corrected_verdict"] for item in recheck}
    tacgen_map = {(item["split"], item["full_name"]): item for item in tacgen}
    hard_model = []
    for key, verdict in recheck_map.items():
        source = tacgen_map.get(key)
        if source is None:
            continue
        hard_model.append(
            {
                "full_name": key[1],
                "solved": verdict == "closed",
                "cost": source["generation_seconds"] + (source["verification_seconds"] or 0.0),
            }
        )
    trivial_symbolic_cost = sum(trivial_walls) / len(trivial_walls) if trivial_walls else 0.0
    trivial_model = smoke_model_data["results"]
    trivial_model_cost = (
        sum(item["generation_seconds"] + (item["verification_seconds"] or 0.0) for item in trivial_model) / len(trivial_model)
        if trivial_model
        else 0.0
    )
    trivial = {
        "goals": len(unique_accept),
        "symbolic_solved": len(unique_accept),
        "symbolic_cost_per_goal": round(trivial_symbolic_cost, 3),
        "model_solved": sum(1 for item in trivial_model if item["verdict"] == "closed"),
        "model_cost_per_goal": round(trivial_model_cost, 3),
    }
    hard_symbolic_solved = sum(1 for item in hard_symbolic if item["solved"])
    hard_symbolic_cost = sum(item["cost"] for item in hard_symbolic)
    hard_model_solved = sum(1 for item in hard_model if item["solved"])
    hard_model_cost = sum(item["cost"] for item in hard_model)
    hard = {
        "goals": len(hard_symbolic),
        "symbolic_solved": hard_symbolic_solved,
        "symbolic_cost_total": round(hard_symbolic_cost, 3),
        "model_solved": hard_model_solved,
        "model_cost_total": round(hard_model_cost, 3),
    }

    def policy(trivial_goals: int, trivial_sym_cost: float, trivial_sym_solved: int, trivial_model_solved: int, trivial_model_cost: float,
               hard_goals: int, hard_sym_cost: float, hard_sym_solved: int, hard_model_solved: int, hard_model_cost: float) -> dict:
        direct_cost = trivial_goals * trivial_model_cost + hard_model_cost
        direct_solved = trivial_model_solved + hard_model_solved
        router_cost = trivial_goals * trivial_sym_cost + hard_sym_cost + (hard_goals - hard_sym_solved) * (hard_model_cost / hard_goals if hard_goals else 0.0) + (trivial_goals - trivial_sym_solved) * trivial_model_cost
        router_solved = trivial_sym_solved + hard_sym_solved + hard_model_solved
        return {
            "direct_model": {"solved": direct_solved, "cost": round(direct_cost, 3), "efficiency": round(direct_solved / direct_cost * 100, 4) if direct_cost else None},
            "router": {"solved": router_solved, "cost": round(router_cost, 3), "efficiency": round(router_solved / router_cost * 100, 4) if router_cost else None},
            "overhead_seconds": round(router_cost - direct_cost, 3),
        }

    per_goal_hard_model = hard_model_cost / len(hard_model) if hard_model else 0.0
    sets = {
        "trivial_smoke32": policy(
            trivial["goals"], trivial["symbolic_cost_per_goal"], trivial["symbolic_solved"],
            trivial["model_solved"], trivial["model_cost_per_goal"],
            0, 0.0, 0, 0, 0.0,
        ),
        "hard_val": policy(
            0, 0.0, 0, 0, 0.0,
            hard["goals"], hard["symbolic_cost_total"], hard["symbolic_solved"], hard["model_solved"], hard["model_cost_total"],
        ),
    }
    union_cost_symbolic = trivial["goals"] * trivial["symbolic_cost_per_goal"] + hard["symbolic_cost_total"]
    union_model_cost = trivial["goals"] * trivial["model_cost_per_goal"] + hard["model_cost_total"]
    union_direct_solved = trivial["model_solved"] + hard["model_solved"]
    union_router_solved = trivial["symbolic_solved"] + hard["symbolic_solved"] + hard["model_solved"]
    union_router_cost = union_cost_symbolic + (hard["goals"] - hard["symbolic_solved"]) * per_goal_hard_model + (trivial["goals"] - trivial["symbolic_solved"]) * trivial["model_cost_per_goal"]
    sets["union"] = {
        "direct_model": {"solved": union_direct_solved, "cost": round(union_model_cost, 3), "efficiency": round(union_direct_solved / union_model_cost * 100, 4) if union_model_cost else None},
        "router": {"solved": union_router_solved, "cost": round(union_router_cost, 3), "efficiency": round(union_router_solved / union_router_cost * 100, 4) if union_router_cost else None},
        "overhead_seconds": round(union_router_cost - union_model_cost, 3),
    }
    verdicts = {
        name: ("não refutada" if (data["router"]["efficiency"] or 0) >= (data["direct_model"]["efficiency"] or 0) else "refutada neste subconjunto")
        for name, data in sets.items()
    }
    payload = {
        "hypothesis": "LH-093",
        "gate": "falha se overhead ou erros de roteamento reduzirem eficiência",
        "cost_model": "custos medidos: simbólicas no LTP-00/LTP-04, modelo no LTP-03/LTP-04 e smoke-model",
        "sets": sets,
        "verdicts": verdicts,
        "inputs": {
            "trivial_symbolic_source": "research/lean/runs/LTP-00/results.json (32 provas válidas)",
            "hard_symbolic_source": "research/lean/runs/LTP-04/symbolic_single_*.json",
            "hard_model_source": "research/lean/runs/LTP-03/reprover_tacgen.json + LTP-04/reprover_recheck.json",
            "trivial_model_source": args.smoke_model,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sets": sets, "verdicts": verdicts}, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LH-093 roteador simbólico-neural")
    sub = parser.add_subparsers(dest="command", required=True)
    states = sub.add_parser("smoke-states")
    states.add_argument("--out", required=True)
    states.add_argument("--workdir", default=str(ROOT / ".local" / "runs" / "lean" / "LTP-05" / "smoke_states"))
    model = sub.add_parser("smoke-model")
    model.add_argument("--states", required=True)
    model.add_argument("--out", required=True)
    model.add_argument("--workdir", default=str(ROOT / ".local" / "runs" / "lean" / "LTP-05" / "smoke_model"))
    ev = sub.add_parser("evaluate")
    ev.add_argument("--smoke-model", required=True)
    ev.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "smoke-states":
        return smoke_states(args)
    if args.command == "smoke-model":
        return smoke_model(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
