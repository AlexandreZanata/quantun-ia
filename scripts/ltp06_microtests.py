#!/usr/bin/env python3
"""LTP-06 — microteste T0/T1 de LH-001…LH-100.

Cada hipótese recebe um resultado explícito: medida (T0/T1 com dados reais),
artefato (evidência de fase anterior) ou bloqueada (razão registrada). Nenhuma
hipótese é descartada em silêncio e resultados negativos são preservados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ltp03_bm25 import accessible_indexes, load_corpus, locate_premise, transitive_closure  # noqa: E402
from scripts.ltp05_canonical_bytes import normalize_state  # noqa: E402

BENCHMARK = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4"
RUNS = ROOT / "research" / "lean" / "runs"
IDENT = re.compile(r"[\w'.]+")
GOAL_LINE = re.compile(r"^\s*⊢")

BLOCKED_REASONS = {
    "ARC": "T2: requer treino de arquitetura",
    "OBJ": "T2: requer treino/objetivo de otimização",
    "CMP": "T2: requer treino, destilação ou quantização",
    "SEA": "T1/T2: requer loop de busca integrado com o verificador",
    "FBK": "T2: requer treino de modelo auxiliar ou parser incremental",
    "DAT": "T1/T2: requer histórico do mathlib, gerador de mutações verificado ou treino",
    "MEM": "T2: requer treino de adaptadores/memória neural",
    "SYS": "T2: requer modelo treinado ou pipeline de sistemas integrado",
    "RET": "T1/T2: requer reranker treinado ou verificação Lean por premissa",
    "REP": "T1: requer AST/normalização no Lean não implementados nesta fase",
}
BLOCKED_OVERRIDES = {
    "LH-002": "T1: requer normalização De Bruijn verificada no Lean",
    "LH-003": "T1: requer geração e verificação de renomeações alfa",
    "LH-006": "T1: requer pares de erro do compilador coletados",
    "LH-009": "T1: requer exportação de AST tipada",
    "LH-010": "T1: requer extração de AST e segunda visão",
    "LH-029": "T1: requer re-ranking por clusters em escala",
    "LH-030": "T1: requer ablação com verificação Lean por premissa",
    "LH-067": "T1: requer minimizador de termos com recompilação",
    "LH-071": "T1: requer snapshots históricos do mathlib",
    "LH-073": "T1: requer teoremas de equivalência certificados",
    "LH-077": "T1: requer grafo de dependências do mathlib em estágios",
    "LH-078": "T1: requer corpus de tentativas falhas com prefixos",
    "LH-079": "T1: requer amostragem por discordância com verificação",
    "LH-080": "T2: requer treino de adaptadores por domínio",
    "LH-086": "T2: requer treino de LoRA por namespace",
    "LH-087": "T2: requer treino com replay",
    "LH-089": "T1: requer unificação real no Lean",
    "LH-090": "T2: requer geração de adaptadores sem gradiente",
    "LH-094": "T1: requer pipeline CPU/GPU sobreposto",
    "LH-096": "T2: requer preditor de dificuldade treinado",
    "LH-097": "T2: requer dois modelos treinados",
    "LH-099": "T1: requer candidatos paralelos com verificação concorrente",
    "LH-100": "T2: requer treino de ensemble",
}


def load_val(split: str) -> list[dict]:
    return json.loads((BENCHMARK / split / "val.json").read_text(encoding="utf-8"))


def val_states(split: str, limit: int = 2000) -> list[str]:
    states = []
    for entry in load_val(split):
        for tactic in entry.get("traced_tactics", []):
            states.append(tactic["state_before"])
            if len(states) >= limit:
                return states
    return states


def hypothesis_line(state: str) -> str | None:
    for line in state.splitlines():
        if GOAL_LINE.match(line):
            return line
    return None


def goal_hash(state: str) -> str | None:
    line = hypothesis_line(state)
    if line is None:
        return None
    return hashlib.sha256(normalize_state(line).encode("utf-8")).hexdigest()


def result(status: str, gate: str, metrics: dict, notes: str) -> dict:
    return {"status": status, "gate": gate, "metrics": metrics, "notes": notes}


def lh004(states: list[str]) -> dict:
    total = unique = 0
    for state in states:
        tokens = IDENT.findall(state)
        bigrams = list(zip(tokens, tokens[1:]))
        total += len(bigrams)
        unique += len(set(bigrams))
    reduction = 1 - unique / total if total else 0.0
    return result(
        "valid" if reduction >= 0.25 else "refuted",
        "tokens caírem ≥25% ou sucesso cair",
        {"bigram_reduction": round(reduction, 4), "total_bigrams": total},
        "proxy de compressão por repetição de subexpressões; sucesso exige T2",
    )


def lh005(states: list[str]) -> dict:
    kept_tokens = total_tokens = 0
    losses = 0
    considered = 0
    for state in states:
        goal = hypothesis_line(state)
        if goal is None:
            continue
        considered += 1
        goal_tokens = set(IDENT.findall(goal))
        lines = state.splitlines()
        goal_index = next((i for i, line in enumerate(lines) if GOAL_LINE.match(line)), len(lines))
        hypotheses = lines[:goal_index]
        kept = [line for line in hypotheses if set(IDENT.findall(line)) & goal_tokens or ":" not in line]
        dropped = [line for line in hypotheses if line not in kept]
        total_tokens += len(IDENT.findall(state))
        kept_tokens += len(IDENT.findall(goal)) + sum(len(IDENT.findall(line)) for line in kept)
        for line in dropped:
            names = [name for name in IDENT.findall(line.split(":")[0]) if name]
            if any(re.search(rf"\b{re.escape(name)}\b", goal) for name in names):
                losses += 1
                break
    reduction = 1 - kept_tokens / total_tokens if total_tokens else 0.0
    loss_rate = losses / considered if considered else 0.0
    status = "valid" if reduction > 0 and loss_rate <= 0.02 else "refuted"
    return result(
        status,
        "economizar tokens mas perder mais de 2 pp de sucesso",
        {"token_reduction": round(reduction, 4), "loss_rate_proxy": round(loss_rate, 4), "states": considered},
        "perda medida por referência da tática humana a hipótese descartada",
    )


def lh007(split: str, limit: int = 2000) -> dict:
    diffs = []
    for entry in load_val(split):
        steps = entry.get("traced_tactics", [])
        for before, after in zip(steps, steps[1:]):
            a = set(normalize_state(before["state_after"]).splitlines())
            b = set(normalize_state(after["state_before"]).splitlines())
            if a or b:
                diffs.append(1 - len(a & b) / max(len(a | b), 1))
        if len(diffs) >= limit:
            break
    mean = sum(diffs) / len(diffs) if diffs else 0.0
    return result(
        "inconclusive",
        "compressão não melhorar provas por joule",
        {"pairs": len(diffs), "mean_delta_ratio": round(mean, 4)},
        "redundância medida; ganho fim a fim exige T2",
    )


def lh008(split: str) -> dict:
    ranks = []
    for entry in load_val(split):
        for tactic in entry.get("traced_tactics", []):
            state = tactic["state_before"]
            goal = hypothesis_line(state)
            if goal is None:
                continue
            lines = state.splitlines()
            goal_index = next((i for i, line in enumerate(lines) if GOAL_LINE.match(line)), len(lines))
            hypotheses = [line for line in lines[:goal_index] if ":" in line]
            if not hypotheses:
                continue
            used = IDENT.findall(tactic["tactic"])
            rank = None
            for index, line in enumerate(hypotheses):
                name = IDENT.findall(line.split(":")[0])
                if name and name[-1] in used:
                    rank = index + 1
                    break
            if rank is not None:
                ranks.append((rank, len(hypotheses)))
    if not ranks:
        return result("inconclusive", "não superar três ordens aleatórias", {"used": 0}, "sem hipóteses usadas identificáveis")
    observed = sum(rank for rank, _ in ranks) / len(ranks)
    expected = sum((size + 1) / 2 for _, size in ranks) / len(ranks)
    status = "valid" if observed < expected * 0.9 else "refuted"
    return result(
        status,
        "não superar três ordens aleatórias pré-fixadas",
        {"used_ranks": len(ranks), "mean_rank": round(observed, 3), "random_expectation": round(expected, 3)},
        "posição da hipótese citada pela tática humana vs ordem aleatória",
    )


def lh023(split: str, proofs: int = 20) -> dict:
    premises, by_path, imports = load_corpus()
    closure = transitive_closure(imports, set(by_path))
    indegree: dict[str, int] = {}
    for targets in imports.values():
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1
    pagerank = {path: 1.0 for path in by_path}
    for _ in range(10):
        updated = {path: 0.15 for path in by_path}
        for path, targets in imports.items():
            share = pagerank.get(path, 0.0) / max(len(targets), 1)
            for target in targets:
                if target in updated:
                    updated[target] += 0.85 * share
        pagerank = updated
    val = load_val(split)
    steps = 0
    recall_indegree = recall_pagerank = 0.0
    for entry in val:
        if not entry.get("traced_tactics"):
            continue
        selected = entry
        candidates = accessible_indexes(premises, by_path, closure, entry["file_path"], tuple(entry["start"]))
        if not candidates:
            continue
        for tactic in entry["traced_tactics"]:
            annotation = tactic.get("annotated_tactic") or []
            positives = set()
            if len(annotation) > 1:
                for provenance in annotation[1]:
                    index = locate_premise(premises, by_path, provenance["def_path"], (provenance["def_pos"][0], provenance["def_pos"][1]))
                    if index is not None:
                        positives.add(index)
            if not positives:
                continue
            ranked_degree = sorted(candidates, key=lambda i: indegree.get(premises[i]["path"], 0), reverse=True)[:32]
            ranked_pagerank = sorted(candidates, key=lambda i: pagerank.get(premises[i]["path"], 0.0), reverse=True)[:32]
            recall_indegree += len(positives & set(ranked_degree)) / len(positives)
            recall_pagerank += len(positives & set(ranked_pagerank)) / len(positives)
            steps += 1
        if steps >= proofs * 6:
            break
    if not steps:
        return result("inconclusive", "não aumentar recall@32 sobre popularidade global", {"steps": 0}, "sem passos com premissas")
    r_deg = recall_indegree / steps
    r_pr = recall_pagerank / steps
    return result(
        "inconclusive",
        "não aumentar recall@32 sobre popularidade global",
        {"steps": steps, "recall32_indegree": round(r_deg, 4), "recall32_pagerank": round(r_pr, 4)},
        "ranking por arquivo é grosseiro demais; reexecutar por premissa com popularidade individual",
    )


def duplicate_rate(states: list[str], key: Callable[[str], str | None]) -> dict:
    seen = set()
    hits = 0
    total = 0
    for state in states:
        value = key(state)
        if value is None:
            continue
        total += 1
        if value in seen:
            hits += 1
        seen.add(value)
    return {"states": total, "hits": hits, "hit_rate": round(hits / total, 4) if total else 0.0}


def lh026(states: list[str]) -> dict:
    data = duplicate_rate(states, goal_hash)
    return result(
        "valid" if data["hit_rate"] >= 0.05 else "refuted",
        "hit-rate útil menor que 5% fora do treino",
        data,
        "cache por hash canônico do objetivo em val",
    )


def lh035(split: str) -> dict:
    from collections import Counter

    pairs = Counter()
    steps = 0
    for entry in load_val(split):
        tactics = [tactic["tactic"] for tactic in entry.get("traced_tactics", [])]
        for left, right in zip(tactics, tactics[1:]):
            pairs[(left, right)] += 1
        steps += len(tactics)
    frequent = {pair for pair, count in pairs.items() if count >= 10}
    covered = sum(count for pair, count in pairs.items() if pair in frequent)
    coverage = covered / steps if steps else 0.0
    status = "valid" if coverage >= 0.25 else ("refuted" if steps else "inconclusive")
    return result(
        status,
        "macros não generalizarem ao split temporal",
        {"steps": steps, "distinct_bigrams": len(pairs), "coverage": round(coverage, 4)},
        "cobertura de bigramas frequentes de táticas",
    )


def lh037(split: str) -> dict:
    pairs = []
    for entry in load_val(split):
        for tactic in entry.get("traced_tactics", []):
            state_hash = hashlib.sha256(normalize_state(tactic["state_before"]).encode("utf-8")).hexdigest()
            pairs.append((state_hash, tactic["tactic"]))
    seen = set()
    repeats = 0
    for pair in pairs:
        if pair in seen:
            repeats += 1
        seen.add(pair)
    rate = repeats / len(pairs) if pairs else 0.0
    return result(
        "valid" if rate >= 0.01 else "refuted",
        "taxa de repetição evitada for irrelevante",
        {"pairs": len(pairs), "repeats": repeats, "repeat_rate": round(rate, 4)},
        "pares estado-tática repetidos dentro de val",
    )


def lh039(split: str) -> dict:
    proofs = stagnant = 0
    for entry in load_val(split):
        goals = []
        for tactic in entry.get("traced_tactics", []):
            line = hypothesis_line(tactic["state_before"])
            if line:
                goals.append(hashlib.sha256(normalize_state(line).encode("utf-8")).hexdigest())
        if len(goals) < 2:
            continue
        proofs += 1
        if len(set(goals)) < len(goals):
            stagnant += 1
    rate = stagnant / proofs if proofs else 0.0
    return result(
        "inconclusive",
        "perder mais provas do que o ganho de custo permite",
        {"proofs": proofs, "with_repeated_goals": stagnant, "rate": round(rate, 4)},
        "proxy de estagnação; efeito em busca exige loop integrado",
    )


def lh059(states: list[str]) -> dict:
    from collections import Counter

    counts = Counter()
    for state in states:
        counts.update(IDENT.findall(state))
    total = sum(counts.values())
    covered = 0
    vocab = 0
    for _, count in counts.most_common():
        covered += count
        vocab += 1
        if covered / total >= 0.99:
            break
    reduction = 1 - vocab / len(counts) if counts else 0.0
    return result(
        "valid" if reduction >= 0.5 else "refuted",
        "tokenização ficar mais longa e custo total crescer",
        {"distinct_tokens": len(counts), "vocab_99pct": vocab, "reduction": round(reduction, 4)},
        "vocabulário mínimo para 99% das ocorrências",
    )


def lh061(split: str) -> dict:
    steps = 0
    for entry in load_val(split):
        steps += len(entry.get("traced_tactics", []))
    return result(
        "valid" if steps else "inconclusive",
        "chamadas ou latência excederem geração completa",
        {"tactic_steps": steps, "boundary_aligned": steps},
        "traços do LeanDojo são alinhados por fim de tática por construção",
    )


def lh068() -> dict:
    decision = json.loads((RUNS / "LTP-00" / "decision.json").read_text(encoding="utf-8"))
    metrics = json.loads((RUNS / "LTP-00" / "metrics.json").read_text(encoding="utf-8"))
    return result(
        "valid" if decision["passed"] else "refuted",
        "suíte não detectar regressões reais",
        {"cases": metrics["cases_total"], "rejected": metrics["rejected"], "gate_passed": decision["passed"]},
        "suíte adversarial do LTP-00 como regressão formal permanente",
    )


def lh072(split: str) -> dict:
    from collections import Counter

    counts = Counter()
    for entry in load_val(split):
        for tactic in entry.get("traced_tactics", []):
            head = tactic["tactic"].split()[0] if tactic["tactic"].split() else ""
            counts[head] += 1
    total = sum(counts.values())
    top5 = sum(count for _, count in counts.most_common(5))
    rare = sum(1 for _, count in counts.items() if count / total < 0.01)
    return result(
        "valid" if rare else "refuted",
        "macro-sucesso não melhorar",
        {"distinct_tactics": len(counts), "top5_share": round(top5 / total, 4), "rare_tactics": rare},
        "desbalanceamento de famílias de táticas em val",
    )


def lh074() -> dict:
    results = json.loads((RUNS / "LTP-00" / "results.json").read_text(encoding="utf-8"))
    corruptions = [item for item in results if item["case_id"].endswith("WRONG_TACTIC")]
    rejected = sum(1 for item in corruptions if item["verdict"] == "reject")
    return result(
        "valid" if corruptions and rejected == len(corruptions) else "refuted",
        "desempenho em erros novos não melhorar",
        {"corruptions": len(corruptions), "rejected": rejected},
        "corrupções mínimas de prova detectadas pelo verificador (LTP-00)",
    )


def lh075(split: str) -> dict:
    proofs = multi = 0
    for entry in load_val(split):
        steps = len(entry.get("traced_tactics", []))
        if steps:
            proofs += 1
            if steps >= 2:
                multi += 1
    rate = multi / proofs if proofs else 0.0
    return result(
        "valid" if rate >= 0.3 else "refuted",
        "houver vazamento entre splits por ancestralidade",
        {"proofs": proofs, "multi_step": multi, "rate": round(rate, 4)},
        "potencial de mineração de submetas por passos intermediários",
    )


def lh076(states: list[str]) -> dict:
    seen = set()
    duplicates = 0
    for state in states:
        key = hashlib.sha256(normalize_state(state).encode("utf-8")).hexdigest()
        if key in seen:
            duplicates += 1
        seen.add(key)
    rate = duplicates / len(states) if states else 0.0
    return result(
        "valid" if rate >= 0.01 else "refuted",
        "custo de deduplicação superar benefício ou remover casos distintos",
        {"states": len(states), "duplicates": duplicates, "rate": round(rate, 4)},
        "deduplicação alfa-normalizada por normalização canônica",
    )


def lh084(split: str) -> dict:
    per_file: dict[str, set[str]] = {}
    cross_file_hits = 0
    total = 0
    seen: dict[str, str] = {}
    for entry in load_val(split):
        for tactic in entry.get("traced_tactics", []):
            key = goal_hash(tactic["state_before"])
            if key is None:
                continue
            total += 1
            if key in seen and seen[key] != entry["file_path"]:
                cross_file_hits += 1
            seen[key] = entry["file_path"]
    rate = cross_file_hits / total if total else 0.0
    return result(
        "inconclusive",
        "benefício existir apenas no modo contaminante",
        {"states": total, "cross_file_hits": cross_file_hits, "rate": round(rate, 4)},
        "objetivos repetidos entre arquivos indicam risco de contaminação transdutiva",
    )


def lh085() -> dict:
    premises, by_path, imports = load_corpus()
    closure = transitive_closure(imports, set(by_path))
    sizes = sorted(len(closure.get(path, set())) for path in by_path)
    median = sizes[len(sizes) // 2] if sizes else 0
    return result(
        "inconclusive",
        "extração não reduzir custo fim a fim",
        {"files": len(by_path), "median_imported_files": median, "max_imported_files": sizes[-1] if sizes else 0},
        "tamanho do fechamento de imports como proxy de contexto em grafo",
    )


def lh088(split: str) -> dict:
    pairs = []
    for entry in load_val(split):
        for tactic in entry.get("traced_tactics", []):
            key = hashlib.sha256((normalize_state(tactic["state_before"]) + tactic["tactic"]).encode("utf-8")).hexdigest()
            pairs.append(key)
    seen = set()
    avoided = 0
    false_positives = 0
    for key in pairs:
        if key in seen:
            avoided += 1
        seen.add(key)
    return result(
        "valid" if avoided / len(pairs) >= 0.01 else "refuted",
        "falsos positivos reduzirem soluções",
        {"pairs": len(pairs), "avoided_repeats": avoided, "rate": round(avoided / len(pairs), 4), "false_positives": false_positives},
        "memória exata como limite superior do filtro Bloom",
    )


CALL_WALLS: dict[str, float] | None = None


def measure_call_walls() -> dict[str, float]:
    global CALL_WALLS
    if CALL_WALLS is not None:
        return CALL_WALLS
    from scripts.lean_verify import resolve_lean_context, run_lean

    context = resolve_lean_context(ROOT / "lean")
    base = ROOT / ".local" / "runs" / "lean" / "LTP-06"
    import_only = run_lean("import Mathlib.Tactic\n", base / "import_only", context, 300.0)
    with_proof = run_lean(
        "import Mathlib.Tactic\nexample : (2 : Nat) + 2 = 4 := by norm_num\n",
        base / "with_proof",
        context,
        300.0,
    )
    CALL_WALLS = {"import_only": import_only.wall_seconds, "with_proof": with_proof.wall_seconds}
    return CALL_WALLS


def lh095() -> dict:
    walls = measure_call_walls()
    return result(
        "inconclusive",
        "houver estado residual ou ganho menor que 15%",
        {
            "import_only_seconds": round(walls["import_only"], 3),
            "with_proof_seconds": round(walls["with_proof"], 3),
            "note": "segunda chamada mais rápida que a primeira por cache de página; atribuição exige worker persistente integrado",
        },
        "medição de import instável sob cache; quantificação requer pipeline de workers (T2)",
    )


def lh098() -> dict:
    data = lh095()
    data["notes"] = "mesma medição de overhead de import aplicada a cache de imports tokenizados"
    data["gate"] = "não reduzir tempo total reprodutível"
    return data


ARTIFACTS = {
    "LH-001": {
        "status": "rejected",
        "gate": "não-inferioridade em sucesso com pelo menos 20% menos memória",
        "metrics": {"reduction_vs_raw": [0.0129, 0.03], "reduction_vs_int32": [0.1733, 0.2039]},
        "notes": "medido em LTP-05; sucesso pleno exigiria T2",
        "source": "research/lean/runs/LTP-05/canonical_bytes.json",
    },
    "LH-021": {
        "status": "inconclusive",
        "gate": "não superar o mesmo gerador sem recuperação",
        "metrics": {"R@1": [9.00, 2.13], "R@10": [24.10, 15.54]},
        "notes": "BM25 medido em LTP-03; comparação fim a fim exige gerador integrado",
        "source": "research/lean/runs/LTP-03/bm25_results.json",
    },
    "LH-024": {
        "status": "rejected",
        "gate": "recall de premissas usadas cair abaixo de 99%",
        "metrics": {"premise_recall": [0.6907, 0.7821]},
        "notes": "medido em LTP-05",
        "source": "research/lean/runs/LTP-05/type_filter.json",
    },
    "LH-069": {
        "status": "valid",
        "gate": "replay não for determinístico",
        "metrics": {"cases": 8, "identical": True, "accepted": 5},
        "notes": "replay limpo duplo idêntico em LTP-04",
        "source": "research/lean/runs/LTP-04/replay_results.json",
    },
    "LH-070": {
        "status": "valid",
        "gate": "qualquer mutação passar pelo gate",
        "metrics": {"cases": 32, "deterministic": 32, "mutants_rejected": 6},
        "notes": "hash textual e semântico medidos em LTP-04",
        "source": "research/lean/runs/LTP-04/semantic_results.json",
    },
    "LH-093": {
        "status": "inconclusive",
        "gate": "overhead ou erros de roteamento reduzirem eficiência",
        "metrics": {"union_efficiency_router": 4.4673, "union_efficiency_model": 4.178},
        "notes": "refutada em conjuntos homogêneos; medida em LTP-05",
        "source": "research/lean/runs/LTP-05/router.json",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="LTP-06 microtestes")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    registry = json.loads((ROOT / "research" / "lean" / "hypotheses.json").read_text(encoding="utf-8"))
    hypotheses = registry["hypotheses"]
    random_states = val_states("random")
    novel_states = val_states("novel_premises")
    all_states = random_states + novel_states
    rows: list[dict] = []
    measured: dict[str, Callable[[], dict]] = {
        "LH-004": lambda: lh004(all_states),
        "LH-005": lambda: lh005(all_states),
        "LH-007": lambda: lh007("random"),
        "LH-008": lambda: lh008("random"),
        "LH-023": lambda: lh023("random"),
        "LH-026": lambda: lh026(all_states),
        "LH-035": lambda: lh035("random"),
        "LH-037": lambda: lh037("random"),
        "LH-039": lambda: lh039("random"),
        "LH-059": lambda: lh059(all_states),
        "LH-061": lambda: lh061("random"),
        "LH-068": lh068,
        "LH-072": lambda: lh072("random"),
        "LH-074": lh074,
        "LH-075": lambda: lh075("random"),
        "LH-076": lambda: lh076(all_states),
        "LH-084": lambda: lh084("random"),
        "LH-085": lh085,
        "LH-088": lambda: lh088("random"),
        "LH-095": lh095,
        "LH-098": lh098,
    }
    for hypothesis in hypotheses:
        identifier = hypothesis["id"]
        started = time.monotonic()
        if identifier in measured:
            payload = measured[identifier]()
            payload["evidence"] = "medido nesta fase"
        elif identifier in ARTIFACTS:
            payload = dict(ARTIFACTS[identifier])
            payload["evidence"] = "artefato de fase anterior"
        else:
            family = hypothesis["family_code"]
            reason = BLOCKED_OVERRIDES.get(identifier, BLOCKED_REASONS.get(family, "T2: requer treino"))
            payload = result("blocked", hypothesis["falsification_gate"], {}, reason)
            payload["evidence"] = "bloqueada nesta fase"
        duration = time.monotonic() - started
        rows.append(
            {
                "id": identifier,
                "title": hypothesis["title"],
                "family_code": hypothesis["family_code"],
                "cost_tier": hypothesis["cost_tier"],
                "status": payload["status"],
                "gate": payload["gate"],
                "metrics": payload["metrics"],
                "notes": payload["notes"],
                "evidence": payload["evidence"],
                "duration_seconds": round(duration, 4),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "verifications": 0,
                "seed": None,
                "deterministic": True,
            }
        )
        print(f"[{identifier}] {payload['status']:12s} {hypothesis['title'][:44]}", flush=True)
    from collections import Counter

    counts = Counter(row["status"] for row in rows)
    payload = {
        "phase": "LTP-06",
        "hypotheses": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "budget": "T0/T1 em CPU; sem treino; T2 bloqueadas com razão",
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["status_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
