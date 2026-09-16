#!/usr/bin/env python3
"""Baseline BM25 do ReProver sobre o LeanDojo Benchmark 4.

Reproduz o pipeline oficial: tokenizador BPE Whitespace, BM25Okapi, consulta no
estado de prova, premissas acessíveis pelo grafo de imports e métricas oficiais
R@1, R@10 e MRR (com R@32 adicional).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4"
MARK_START = "<a>"
MARK_END = "</a>"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus() -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, list[str]]]:
    premises: list[dict[str, Any]] = []
    by_path: dict[str, list[int]] = {}
    imports: dict[str, list[str]] = {}
    with (BENCHMARK_ROOT / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            path = record["path"]
            imports[path] = record.get("imports", [])
            for item in record["premises"]:
                index = len(premises)
                premises.append(
                    {
                        "path": path,
                        "full_name": item["full_name"],
                        "code": item["code"],
                        "start": tuple(item["start"]),
                        "end": tuple(item["end"]),
                        "kind": item.get("kind"),
                    }
                )
                by_path.setdefault(path, []).append(index)
    return premises, by_path, imports


def locate_premise(premises: list[dict[str, Any]], by_path: dict[str, list[int]], path: str, pos: tuple[int, int]) -> int | None:
    for index in by_path.get(path, []):
        item = premises[index]
        if item["start"] <= pos <= item["end"]:
            return index
    return None


def transitive_closure(imports: dict[str, list[str]], known_paths: set[str]) -> dict[str, set[str]]:
    module_to_path = {path[: -len(".lean")].replace("/", "."): path for path in known_paths}
    graph: dict[str, list[str]] = {}
    for path, import_list in imports.items():
        targets = []
        for name in import_list:
            if name in known_paths:
                targets.append(name)
                continue
            mapped = module_to_path.get(name)
            if mapped is not None:
                targets.append(mapped)
        graph[path] = targets
    memo: dict[str, set[str]] = {}

    def visit(node: str, stack: set[str]) -> set[str]:
        if node in memo:
            return memo[node]
        if node in stack:
            return set()
        stack.add(node)
        result: set[str] = set()
        for child in graph.get(node, []):
            result.add(child)
            result.update(visit(child, stack))
        stack.discard(node)
        memo[node] = result
        return result

    return {path: visit(path, set()) for path in graph}


def serialize_premise(item: dict[str, Any]) -> str:
    annotated = f"{MARK_START}{item['full_name']}{MARK_END}"
    code = item["code"].replace(f"_root_.{item['full_name']}", annotated)
    fields = item["full_name"].split(".")
    for i in range(len(fields)):
        prefix = ".".join(fields[i:])
        new_code = re.sub(rf"(?<=\s)«?{re.escape(prefix)}»?", annotated, code)
        if new_code != code:
            return new_code
    return code


def train_tokenizer(premises: list[dict[str, Any]], out_path: Path, limit: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    premise_texts = [serialize_premise(item) for item in premises[:limit]]
    train = read_json(BENCHMARK_ROOT / "random" / "train.json")
    state_texts: list[str] = []
    for entry in train:
        for tactic in entry.get("traced_tactics", []):
            state_texts.append(tactic["state_before"])
            if len(state_texts) >= limit:
                break
        if len(state_texts) >= limit:
            break
    tokenizer.train_from_iterator(premise_texts + state_texts, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def accessible_indexes(
    premises: list[dict[str, Any]],
    by_path: dict[str, list[int]],
    closure: dict[str, set[str]],
    path: str,
    theorem_start: tuple[int, int],
) -> list[int]:
    indexes = [index for index in by_path.get(path, []) if premises[index]["end"] <= theorem_start]
    for imported in closure.get(path, set()):
        indexes.extend(by_path.get(imported, []))
    return indexes


def evaluate(
    split: str,
    proof_limit: int,
    tokenizer: Tokenizer,
    premises: list[dict[str, Any]],
    by_path: dict[str, list[int]],
    closure: dict[str, set[str]],
    bm25: BM25Okapi,
) -> dict[str, Any]:
    val = read_json(BENCHMARK_ROOT / split / "val.json")
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
    r1 = r10 = r32 = mrr = 0.0
    queries = 0
    missing_positives = 0
    for entry in selected:
        theorem_start = tuple(entry["start"])
        candidates = accessible_indexes(premises, by_path, closure, entry["file_path"], theorem_start)
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
                missing_positives += 1
                continue
            query = tokenizer.encode(tactic["state_before"]).tokens
            scores = bm25.get_batch_scores(query, candidates)
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            ranked = [candidates[i] for i in order]
            if ranked and ranked[0] in positives:
                r1 += 1.0 / len(positives)
            hits10 = len(positives.intersection(ranked[:10]))
            r10 += hits10 / len(positives)
            hits32 = len(positives.intersection(ranked[:32]))
            r32 += hits32 / len(positives)
            for rank, index in enumerate(ranked, start=1):
                if index in positives:
                    mrr += 1.0 / rank
                    break
            queries += 1
    return {
        "split": split,
        "proofs": len(selected),
        "queries": queries,
        "steps_without_positive": missing_positives,
        "R@1": round(100 * r1 / queries, 4) if queries else 0.0,
        "R@10": round(100 * r10 / queries, 4) if queries else 0.0,
        "R@32": round(100 * r32 / queries, 4) if queries else 0.0,
        "MRR": round(mrr / queries, 4) if queries else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BM25 baseline LTP-03")
    parser.add_argument("--proofs", type=int, default=50)
    parser.add_argument("--splits", nargs="+", default=["random", "novel_premises"])
    parser.add_argument("--tokenizer-limit", type=int, default=10000)
    parser.add_argument("--tokenizer-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    premises, by_path, imports = load_corpus()
    closure = transitive_closure(imports, set(by_path))
    tokenizer = train_tokenizer(premises, Path(args.tokenizer_out), args.tokenizer_limit)
    corpus_tokens = [tokenizer.encode(serialize_premise(item)).tokens for item in premises]
    bm25 = BM25Okapi(corpus_tokens)
    summaries = [
        evaluate(split, args.proofs, tokenizer, premises, by_path, closure, bm25)
        for split in args.splits
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summaries": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
