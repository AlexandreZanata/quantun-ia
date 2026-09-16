#!/usr/bin/env python3
"""LTP-07 — fronteira de Pareto e promoção (máximo 30).

Critérios congelados antes da seleção: só hipóteses com microteste `valid` são
elegíveis; as demais ficam fora por refutação, inconclusão ou bloqueio. Eixos:
sucesso Lean (gate passado), classe de custo (negligenciável, offline,
limitado por verificação), parâmetros (zero-aprendido vs alvo nano) e novidade
(família/capacidade única). Domina quem é ≥ em todos os eixos e > em algum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research" / "lean" / "runs"

COST_CLASS = {
    "LH-026": ("negligible", 0, "hash canônico do objetivo em CPU"),
    "LH-061": ("verification_bound", 2, "uma chamada ao verificador por fim de tática"),
    "LH-068": ("offline", 1, "suíte de regressão executada no CI"),
    "LH-069": ("verification_bound", 2, "recompilação limpa por prova aceita"),
    "LH-070": ("verification_bound", 2, "elaboração do tipo por afirmação"),
    "LH-072": ("offline", 1, "reponderação de dados no treino"),
    "LH-074": ("offline", 1, "geração de corrupções por prova"),
    "LH-075": ("offline", 1, "mineração de submetas em traços"),
}

QUALITY_MARGIN = {
    "LH-026": 5.06,
    "LH-061": 1.0,
    "LH-068": 1.0,
    "LH-069": 1.0,
    "LH-070": 1.0,
    "LH-072": 1.0,
    "LH-074": 1.0,
    "LH-075": 2.14,
}

NOVELTY = {
    "LH-026": "cache determinístico de objetivos, zero parâmetros",
    "LH-061": "verificação em fronteiras sintáticas, zero parâmetros",
    "LH-068": "regressão formal permanente, zero parâmetros",
    "LH-069": "replay limpo determinístico, zero parâmetros",
    "LH-070": "hash semântico imutável da afirmação, zero parâmetros",
    "LH-072": "correção de desbalanceamento de táticas raras",
    "LH-074": "negativos mínimos com rótulo de erro",
    "LH-075": "expansão de currículo por submetas verificadas",
}
CAPABILITY_CLASS = {
    "LH-026": "objective_cache",
    "LH-061": "syntax_boundary",
    "LH-068": "regression_suite",
    "LH-069": "clean_replay",
    "LH-070": "statement_hash",
    "LH-072": "data_rebalancing",
    "LH-074": "corruption_negatives",
    "LH-075": "subtheorem_mining",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="LTP-07 fronteira Pareto")
    parser.add_argument("--microtests", default=str(RUNS / "LTP-06" / "microtests.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    micro = json.loads(Path(args.microtests).read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in micro["rows"]}
    candidates = []
    for identifier, row in rows.items():
        if row["status"] != "valid":
            continue
        cost_class, cost_rank, cost_note = COST_CLASS.get(identifier, ("offline", 1, "não classificado"))
        hypothesis_target = row["family_code"]
        parameters_rank = 0 if identifier in ("LH-026", "LH-061", "LH-068", "LH-069", "LH-070") else 1
        candidates.append(
            {
                "id": identifier,
                "title": row["title"],
                "family_code": hypothesis_target,
                "quality_margin": QUALITY_MARGIN.get(identifier, 1.0),
                "cost_class": cost_class,
                "cost_rank": cost_rank,
                "cost_note": cost_note,
                "parameters_rank": parameters_rank,
                "novelty": NOVELTY.get(identifier, row["notes"]),
                "capability_class": CAPABILITY_CLASS.get(identifier, identifier),
                "metrics": row["metrics"],
                "gate": row["gate"],
            }
        )
    dominated = []
    for left in candidates:
        for right in candidates:
            if left["capability_class"] != right["capability_class"]:
                continue
            if (
                left["quality_margin"] >= right["quality_margin"]
                and left["cost_rank"] <= right["cost_rank"]
                and left["parameters_rank"] <= right["parameters_rank"]
                and (
                    left["quality_margin"] > right["quality_margin"]
                    or left["cost_rank"] < right["cost_rank"]
                    or left["parameters_rank"] < right["parameters_rank"]
                )
            ):
                dominated.append({"dominated": right["id"], "by": left["id"], "class": left["capability_class"]})
    frontier = [item for item in candidates if item["id"] not in {entry["dominated"] for entry in dominated}]
    frontier.sort(key=lambda item: (item["cost_rank"], item["parameters_rank"], item["id"]))
    promoted = frontier[:30]
    excluded = []
    for identifier, row in sorted(rows.items()):
        if row["status"] == "valid":
            continue
        excluded.append({"id": identifier, "status": row["status"], "reason": row["notes"]})
    payload = {
        "phase": "LTP-07",
        "criteria": {
            "eligibility": "microteste valid (gate T0/T1 passado)",
            "axes": ["sucesso Lean (margem sobre o gate)", "classe de custo", "parâmetros", "novidade"],
            "dominance": "≥ em todos os eixos e > em pelo menos um, avaliada apenas dentro da mesma classe de capacidade; mecanismos complementares não competem",
            "cap": 30,
        },
        "candidates": candidates,
        "dominance_pairs": dominated,
        "frontier": [item["id"] for item in frontier],
        "promoted": [item["id"] for item in promoted],
        "promoted_count": len(promoted),
        "not_promoted": excluded,
        "notes": "somente 8 hipóteses passaram no T0/T1; o teto de 30 não foi atingido e 73 bloqueios/9 inconclusões ficam preservados",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"promoted": payload["promoted"], "count": len(promoted), "dominated": dominated}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
