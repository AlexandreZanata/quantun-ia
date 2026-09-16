#!/usr/bin/env python3
"""LH-001 — Estado em bytes canônicos.

Normaliza estados de prova (NFC, espaços, linhas) e mede memória da representação
em bytes canônicos contra bytes crus e contra a tokenização ByT5 do ReProver.
Verifica determinismo e idempotência da normalização.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4"


def normalize_state(state: str) -> str:
    text = unicodedata.normalize("NFC", state)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_states(split: str, limit: int) -> list[str]:
    entries = json.loads((BENCHMARK_ROOT / split / "val.json").read_text(encoding="utf-8"))
    states: list[str] = []
    for entry in entries:
        for tactic in entry.get("traced_tactics", []):
            states.append(tactic["state_before"])
            if len(states) >= limit:
                return states
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description="LH-001 bytes canônicos")
    parser.add_argument("--splits", nargs="+", default=["random", "novel_premises"])
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)

    results = []
    for split in args.splits:
        states = load_states(split, args.limit)
        raw_bytes = canonical_bytes = token_count = 0
        idempotent = 0
        for state in states:
            canonical = normalize_state(state)
            raw_bytes += len(state.encode("utf-8"))
            canonical_bytes += len(canonical.encode("utf-8"))
            if normalize_state(canonical) == canonical:
                idempotent += 1
            if tokenizer is not None:
                token_count += len(tokenizer.encode(canonical).ids)
        count = len(states)
        reduction = 1 - (canonical_bytes / raw_bytes) if raw_bytes else 0.0
        results.append(
            {
                "split": split,
                "states": count,
                "raw_bytes": raw_bytes,
                "canonical_bytes": canonical_bytes,
                "reduction_vs_raw": round(reduction, 4),
                "bytes_per_state_raw": round(raw_bytes / count, 2) if count else 0,
                "bytes_per_state_canonical": round(canonical_bytes / count, 2) if count else 0,
                "token_count": token_count,
                "token_int32_bytes": token_count * 4,
                "reduction_vs_token_int32": round(1 - (canonical_bytes / (token_count * 4)), 4)
                if token_count
                else None,
                "idempotent": idempotent,
                "sample_sha256": sha256(normalize_state(states[0])) if states else None,
            }
        )
        print(f"[LH-001] {split}: {count} estados, redução {reduction:.1%}", flush=True)
    payload = {
        "hypothesis": "LH-001",
        "gate": "não-inferioridade em sucesso com pelo menos 20% menos memória",
        "t0_criterion": "redução de memória >= 20% com normalização determinística e idempotente",
        "success_criterion_status": "pendente T2: exige treino para medir não-inferioridade de sucesso",
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
