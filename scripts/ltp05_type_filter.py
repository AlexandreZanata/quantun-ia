#!/usr/bin/env python3
"""LH-024 — Filtro por tipo antes da busca.

Constrói um filtro aproximado de compatibilidade entre o objetivo e as
premissas do corpus (interseção de identificadores) e mede o recall das
premissas realmente usadas nas provas de val. Gate: falha se o recall cair
abaixo de 99%.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4"
IDENT = re.compile(r"[\w'.]+")


def load_corpus() -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[tuple[str, int, int], int], dict[str, set[int]]]:
    premises: list[dict[str, Any]] = []
    by_path: dict[str, list[int]] = {}
    by_position: dict[tuple[str, int, int], int] = {}
    inverted: dict[str, set[int]] = {}
    with (BENCHMARK_ROOT / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            path = record["path"]
            for item in record["premises"]:
                index = len(premises)
                tokens = set(IDENT.findall(item["code"])) | set(IDENT.findall(item["full_name"]))
                premises.append(
                    {
                        "path": path,
                        "full_name": item["full_name"],
                        "start": tuple(item["start"]),
                        "end": tuple(item["end"]),
                        "tokens": tokens,
                    }
                )
                by_path.setdefault(path, []).append(index)
                by_position[(path, item["start"][0], item["start"][1])] = index
                for token in tokens:
                    inverted.setdefault(token, set()).add(index)
    return premises, by_path, by_position, inverted


def locate_premise(premises: list[dict[str, Any]], by_path: dict[str, list[int]], path: str, pos: tuple[int, int]) -> int | None:
    for index in by_path.get(path, []):
        item = premises[index]
        if item["start"] <= pos <= item["end"]:
            return index
    return None


def goal_tokens(state_before: str) -> set[str]:
    goal_lines = [line for line in state_before.splitlines() if line.strip().startswith("⊢")]
    if not goal_lines:
        return set()
    return set(IDENT.findall(goal_lines[-1]))


def filter_candidates(goal: set[str], inverted: dict[str, set[int]]) -> set[int]:
    candidates: set[int] = set()
    for token in goal:
        candidates |= inverted.get(token, set())
    return candidates


def evaluate(split: str, proof_limit: int, premises: list[dict[str, Any]], by_path, by_position, inverted) -> dict[str, Any]:
    val = json.loads((BENCHMARK_ROOT / split / "val.json").read_text(encoding="utf-8"))
    selected = []
    for entry in val:
        if not entry.get("traced_tactics"):
            continue
        has_positives = any(
            len(tactic.get("annotated_tactic") or [None, []]) > 1 and tactic["annotated_tactic"][1]
            for tactic in entry["traced_tactics"]
        )
        if has_positives:
            selected.append(entry)
        if len(selected) >= proof_limit:
            break
    total = len(premises)
    recall_sum = 0.0
    full_recall_steps = 0
    steps = 0
    kept_sum = 0
    for entry in selected:
        for tactic in entry["traced_tactics"]:
            annotation = tactic.get("annotated_tactic") or []
            positives = set()
            if len(annotation) > 1:
                for provenance in annotation[1]:
                    position = (provenance["def_pos"][0], provenance["def_pos"][1])
                    index = locate_premise(premises, by_path, provenance["def_path"], position)
                    if index is not None:
                        positives.add(index)
            if not positives:
                continue
            goal = goal_tokens(tactic["state_before"])
            candidates = filter_candidates(goal, inverted) if goal else set()
            kept = positives & candidates
            recall_sum += len(kept) / len(positives)
            if len(kept) == len(positives):
                full_recall_steps += 1
            kept_sum += len(candidates)
            steps += 1
    return {
        "split": split,
        "proofs": len(selected),
        "steps": steps,
        "premise_recall": round(recall_sum / steps, 4) if steps else None,
        "steps_with_full_recall": full_recall_steps,
        "mean_candidates": round(kept_sum / steps, 1) if steps else None,
        "corpus_size": total,
        "mean_reduction": round(1 - (kept_sum / steps) / total, 4) if steps else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LH-024 filtro por tipo")
    parser.add_argument("--proofs", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=["random", "novel_premises"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    premises, by_path, by_position, inverted = load_corpus()
    results = [evaluate(split, args.proofs, premises, by_path, by_position, inverted) for split in args.splits]
    payload = {
        "hypothesis": "LH-024",
        "gate": "falha se recall de premissas usadas cair abaixo de 99%",
        "method": "filtro aproximado por interseção de identificadores entre objetivo e código da premissa",
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
