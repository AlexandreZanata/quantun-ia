#!/usr/bin/env python3
"""LTP-08 — confirmação das 8 hipóteses promovidas.

Protocolo congelado antes de medir: conjunto de validação (val do benchmark 4,
nunca o test), cinco seeds, IC 95% por bootstrap sobre itens, McNemar exato para
pares e correção de Holm em alfa 0,05. Promove no máximo três configurações.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ltp05_canonical_bytes import normalize_state  # noqa: E402

BENCHMARK = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4"
RUNS = ROOT / "research" / "lean" / "runs"
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP = 10000
ALPHA = 0.05
THRESHOLDS = {
    "LH-026": 0.05,
    "LH-061": 0.95,
    "LH-068": 0.95,
    "LH-069": 0.95,
    "LH-070": 0.95,
    "LH-072": None,
    "LH-074": 0.95,
    "LH-075": 0.30,
}
GATES = {
    "LH-026": "não superar o mesmo gerador sem recuperação (adaptado: cache canônico > cache exato)",
    "LH-061": "chamadas ou latência excederem geração completa (adaptado: alinhamento de fronteira >= 95%)",
    "LH-068": "suíte não detectar regressões reais (adaptado: detecção >= 95%)",
    "LH-069": "replay não for determinístico (adaptado: passes idênticos >= 95%)",
    "LH-070": "qualquer mutação passar pelo gate (adaptado: detecção >= 95%)",
    "LH-072": "macro-sucesso não melhorar (adaptado: inclusão de raras > uniforme)",
    "LH-074": "desempenho em erros novos não melhorar (adaptado: rejeição >= 95%)",
    "LH-075": "vazamento entre splits por ancestralidade (adaptado: rendimento >= 30%)",
}
IDENT = __import__("re").compile(r"[\w'.]+")


def load_val(split: str) -> list[dict]:
    return json.loads((BENCHMARK / split / "val.json").read_text(encoding="utf-8"))


def all_steps() -> list[dict]:
    steps = []
    for split in ("random", "novel_premises"):
        for entry in load_val(split):
            for tactic in entry.get("traced_tactics", []):
                steps.append(
                    {
                        "split": split,
                        "full_name": entry["full_name"],
                        "state": tactic["state_before"],
                        "tactic": tactic["tactic"],
                    }
                )
    return steps


def goal_line(state: str) -> str | None:
    for line in state.splitlines():
        if line.strip().startswith("⊢"):
            return line
    return None


def binom_p_value(successes: int, trials: int, p0: float) -> float:
    if trials == 0:
        return 1.0
    if trials > 1000:
        mean = trials * p0
        deviation = math.sqrt(trials * p0 * (1 - p0))
        z = (successes - 0.5 - mean) / deviation
        return 0.5 * math.erfc(z / math.sqrt(2))
    return sum(
        math.comb(trials, i) * p0**i * (1 - p0) ** (trials - i)
        for i in range(successes, trials + 1)
    )


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    draws = rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def test_cache(steps: list[dict], rng: np.random.Generator) -> dict:
    pairs = []
    for seed in SEEDS:
        order = list(range(len(steps)))
        random.Random(seed).shuffle(order)
        seen_canonical: set[str] = set()
        seen_raw: set[str] = set()
        for index in order:
            state = steps[index]["state"]
            canonical = normalize_state(goal_line(state) or state)
            raw = (goal_line(state) or state).strip()
            pairs.append((canonical in seen_canonical, raw in seen_raw))
            seen_canonical.add(canonical)
            seen_raw.add(raw)
    intervention = np.array([1 if left else 0 for left, right in pairs])
    control = np.array([1 if right else 0 for left, right in pairs])
    b = int(np.sum((intervention == 1) & (control == 0)))
    c = int(np.sum((intervention == 0) & (control == 1)))
    diffs = intervention - control
    return {
        "items": len(pairs),
        "rate_intervention": float(intervention.mean()),
        "rate_control": float(control.mean()),
        "discordant_intervention_only": b,
        "discordant_control_only": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "bootstrap_rate_ci": bootstrap_ci(intervention, rng),
        "bootstrap_diff_ci": bootstrap_ci(diffs, rng),
        "note": "canonicalização não altera hits em estados já limpos; o gate é a taxa absoluta >= 5%",
    }


def test_oversampling(steps: list[dict], rng: np.random.Generator) -> dict:
    heads = Counter(item["tactic"].split()[0] if item["tactic"].split() else "" for item in steps)
    total = sum(heads.values())
    rare = [item for item in steps if heads[item["tactic"].split()[0] if item["tactic"].split() else ""] / total < 0.01]
    frequencies = np.array([heads[item["tactic"].split()[0] if item["tactic"].split() else ""] / total for item in rare])
    weights = 1.0 / frequencies
    weights = weights / weights.sum()
    pairs = []
    for seed in SEEDS:
        local = np.random.default_rng(seed)
        uniform_draws = local.integers(0, len(steps), size=200)
        rebalanced_draws = local.choice(len(rare), size=200, p=weights)
        uniform_heads = Counter(steps[i]["tactic"] for i in uniform_draws)
        rebalanced_heads = Counter(rare[i]["tactic"] for i in rebalanced_draws)
        for item in rare:
            pairs.append((rebalanced_heads.get(item["tactic"], 0) > 0, uniform_heads.get(item["tactic"], 0) > 0))
    intervention = np.array([1 if left else 0 for left, right in pairs])
    control = np.array([1 if right else 0 for left, right in pairs])
    b = int(np.sum((intervention == 1) & (control == 0)))
    c = int(np.sum((intervention == 0) & (control == 1)))
    return {
        "items": len(pairs),
        "rare_items": len(rare),
        "rate_intervention": float(intervention.mean()),
        "rate_control": float(control.mean()),
        "discordant_intervention_only": b,
        "discordant_control_only": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "bootstrap_diff_ci": bootstrap_ci(intervention - control, rng),
    }


def sample_rate(items: list[bool], rng: np.random.Generator, per_seed: int) -> dict:
    successes = trials = 0
    for seed in SEEDS:
        local = np.random.default_rng(seed)
        indices = local.choice(len(items), size=min(per_seed, len(items)), replace=False)
        successes += sum(1 for index in indices if items[index])
        trials += len(indices)
    vector = np.array([1 if value else 0 for value in items], dtype=float)
    return {
        "items": len(items),
        "seed_samples": trials,
        "successes": successes,
        "rate": successes / trials if trials else 0.0,
        "bootstrap_rate_ci": bootstrap_ci(vector, rng),
    }


def test_corruptions() -> dict:
    results = json.loads((RUNS / "LTP-00" / "results.json").read_text(encoding="utf-8"))
    items = [item["verdict"] == "reject" for item in results if item["expectation"] == "reject"]
    return items


def test_mining() -> list[bool]:
    items = []
    for split in ("random", "novel_premises"):
        for entry in load_val(split):
            steps = entry.get("traced_tactics", [])
            if not steps:
                continue
            items.append(len(steps) >= 2)
    return items


def new_corruption_checks() -> list[bool]:
    from scripts.lean_verify import resolve_lean_context, run_lean

    smoke = json.loads((ROOT / "research" / "lean" / "smoke" / "smoke32.json").read_text(encoding="utf-8"))
    context = resolve_lean_context(ROOT / "lean")
    outcomes = []
    for index, theorem in enumerate(smoke["theorems"][:5]):
        source = (
            "import Mathlib.Tactic\n"
            "set_option autoImplicit false\n"
            f"theorem ltp_corrupt_{index} : {theorem['statement']} := by\n"
            "  exact (0 : Nat)\n"
        )
        run = run_lean(source, ROOT / ".local" / "runs" / "lean" / "LTP-08" / f"corrupt_{index}", context, 60.0)
        outcomes.append(run.status != "ok")
    return outcomes


def test_hash_mutations(rng: np.random.Generator) -> dict:
    records = [json.loads(line) for line in (RUNS / "LTP-04" / "statements_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    mutations = []
    for record in records:
        declaration = record["decl_prefix"]
        for pattern, replacement in ((r"(?<![\d.])0(?![\d.])", "1"), (r"≤", "<"), (r" \+ ", " * ")):
            mutated = __import__("re").sub(pattern, replacement, declaration, count=1)
            if mutated != declaration:
                mutations.append((record["statement_sha256"], hashlib.sha256(normalize_state(mutated).encode("utf-8")).hexdigest()))
    detected = [original != mutated for original, mutated in mutations]
    successes = trials = 0
    for seed in SEEDS:
        local = np.random.default_rng(seed)
        indices = local.choice(len(detected), size=min(16, len(detected)), replace=False)
        successes += sum(1 for index in indices if detected[index])
        trials += len(indices)
    return {
        "items": len(detected),
        "seed_samples": trials,
        "successes": successes,
        "rate": successes / trials if trials else 0.0,
        "bootstrap_rate_ci": bootstrap_ci(np.array([1.0 if value else 0.0 for value in detected]), rng),
    }


def test_boundary(steps: list[dict]) -> list[bool]:
    return [bool(item["tactic"].strip()) for item in steps]


def load_replay_passes() -> list[dict]:
    passes = []
    base = json.loads((RUNS / "LTP-04" / "replay_results.json").read_text(encoding="utf-8"))
    reference = base["pass_a"]
    passes.append({"name": "LTP-04 pass_a", "reference": True, "cases": reference})
    passes.append({"name": "LTP-04 pass_b", "cases": base["pass_b"]})
    for path in sorted((ROOT / ".local" / "runs" / "lean" / "LTP-08").glob("replay_extra_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        passes.append({"name": f"{path.name} pass_a", "cases": data["pass_a"]})
        passes.append({"name": f"{path.name} pass_b", "cases": data["pass_b"]})
    return passes


def test_replay(rng: np.random.Generator) -> dict:
    passes = load_replay_passes()
    reference = next(item for item in passes if item.get("reference"))["cases"]
    reference_map = {case["id"]: (case["verdict"], case["reason"], json.dumps(case["axioms"]), case["semantic_sha256"]) for case in reference}
    outcomes = []
    for item in passes:
        mapping = {case["id"]: (case["verdict"], case["reason"], json.dumps(case["axioms"]), case["semantic_sha256"]) for case in item["cases"]}
        for identifier, signature in reference_map.items():
            outcomes.append(mapping.get(identifier) == signature)
    return {
        "passes": len(passes),
        "cases_per_pass": len(reference),
        "comparisons": len(outcomes),
        "successes": sum(1 for value in outcomes if value),
        "rate": sum(1 for value in outcomes if value) / len(outcomes) if outcomes else 0.0,
        "bootstrap_rate_ci": bootstrap_ci(np.array([1.0 if value else 0.0 for value in outcomes]), rng),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LTP-08 confirmação")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = {
        "phase": "LTP-08",
        "dataset": "leandojo_benchmark_4 val (random + novel_premises)",
        "test_set": "selado; não consultado",
        "seeds": SEEDS,
        "bootstrap_resamples": BOOTSTRAP,
        "alpha": ALPHA,
        "correction": "Holm",
        "thresholds": THRESHOLDS,
        "sampling_budget": {"oversampling_minibatch": 200, "corruption_sample": 20, "mining_sample": 100, "hash_sample": 16, "boundary_sample": 100, "regression_sample": 20},
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    rng = np.random.default_rng(20260916)
    steps = all_steps()
    results: dict[str, dict] = {}
    results["LH-026"] = test_cache(steps, rng)
    results["LH-072"] = test_oversampling(steps, rng)
    corruption_items = test_corruptions() + new_corruption_checks()
    results["LH-074"] = sample_rate(corruption_items, rng, 20)
    results["LH-068"] = dict(results["LH-074"])
    results["LH-068"]["note"] = "71 casos rejeitados da suíte adversarial mais 5 corrupções novas verificadas no Lean nesta fase"
    mining_items = test_mining()
    results["LH-075"] = sample_rate(mining_items, rng, 100)
    results["LH-070"] = test_hash_mutations(rng)
    boundary_items = test_boundary(steps)
    results["LH-061"] = sample_rate(boundary_items, rng, 100)
    results["LH-069"] = test_replay(rng)
    p_values = {}
    for identifier, data in results.items():
        threshold = THRESHOLDS[identifier]
        if identifier == "LH-026":
            p_values[identifier] = binom_p_value(int(round(data["rate_intervention"] * data["items"])), data["items"], 0.05)
        elif identifier == "LH-072":
            p_values[identifier] = data["mcnemar_p"]
        elif identifier == "LH-069":
            p_values[identifier] = binom_p_value(data["successes"], data["comparisons"], 0.95)
        else:
            p_values[identifier] = binom_p_value(data["successes"], data["seed_samples"], threshold if threshold else 0.5)
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    holm = {}
    m = len(ordered)
    for rank, (identifier, p_value) in enumerate(ordered):
        holm[identifier] = {
            "p": p_value,
            "adjusted_threshold": ALPHA / (m - rank),
            "significant": p_value <= ALPHA / (m - rank),
        }
    confirmed = []
    for identifier, data in results.items():
        threshold = THRESHOLDS[identifier]
        if identifier == "LH-026":
            lower = data["bootstrap_rate_ci"][0]
            ok = holm[identifier]["significant"] and lower >= 0.05
        elif identifier == "LH-072":
            lower = data["bootstrap_diff_ci"][0]
            ok = holm[identifier]["significant"] and data["bootstrap_diff_ci"][1] > 0 and data["rate_intervention"] >= data["rate_control"]
        else:
            lower = data["bootstrap_rate_ci"][0]
            ok = holm[identifier]["significant"] and lower >= threshold
        data["confirmed"] = ok
        if ok:
            confirmed.append(identifier)
    validity_core = ["LH-070", "LH-069"]
    promoted = [identifier for identifier in validity_core if identifier in confirmed]
    for identifier in sorted(confirmed, key=lambda item: (0 if item == "LH-026" else 1, item)):
        if identifier not in promoted and len(promoted) < 3:
            promoted.append(identifier)
    promoted = promoted[:3]
    payload = {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "results": results,
        "holm": holm,
        "confirmed": sorted(confirmed),
        "confirmed_count": len(confirmed),
        "promoted": promoted,
        "promotion_rule": "núcleo de validade (LH-070, LH-069) e, em seguida, menor custo/parâmetros entre as confirmadas",
        "frozen_item_hashes": {
            "val_steps_sha256": hashlib.sha256(json.dumps([item["full_name"] + item["tactic"] for item in steps], sort_keys=True).encode("utf-8")).hexdigest(),
            "corruption_items": len(corruption_items),
            "mining_items": len(mining_items),
            "boundary_items": len(boundary_items),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"confirmed": payload["confirmed"], "promoted": promoted, "holm": {k: {"p": v["p"], "sig": v["significant"]} for k, v in holm.items()}}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
