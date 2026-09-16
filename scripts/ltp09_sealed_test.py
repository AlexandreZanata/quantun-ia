#!/usr/bin/env python3
"""LTP-09 — abertura única do teste selado do LeanDojo Benchmark 4.

O `random/test.json` é lido uma única vez: os primeiros 16 teoremas com traços e
extração by-style são copiados para `.local/sealed/LTP-09/` e o arquivo original
nunca é aberto de novo. Os três finalistas (LH-070, LH-069, LH-026) são
avaliados no subconjunto congelado, mais linhas de referência simbólica e
ReProver tacgen. `novel_premises/test.json` permanece selado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lean_verify import (  # noqa: E402
    build_semantic_source,
    canonical_type_sha256,
    parse_canonical_type,
    rename_declaration,
    resolve_lean_context,
    run_lean,
    scan_forbidden,
    strip_to_additive,
    with_options,
)
from scripts.ltp05_canonical_bytes import normalize_state  # noqa: E402

BENCHMARK = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4"
MATHLIB = ROOT / "data" / "raw" / "mathlib4_src"
SEALED = ROOT / ".local" / "sealed" / "LTP-09"
NATIVE = MATHLIB
SUBSET_SIZE = 16
TIMEOUT = 60.0
MEMORY_LIMIT = 8 * 1024**3
TACTICS = ["simp", "aesop", "omega", "linarith", "nlinarith"]
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
MODEL = ROOT / "data" / "raw" / "reprover" / "leandojo-lean4-tacgen-byt5-small" / "67a2c53cc36186fe8539d0a342fc42c50edc68fd"
IDENT = re.compile(r"[\w'.]+")
GOAL = re.compile(r"^\s*⊢")
MUTATIONS = ((r"(?<![\d.])0(?![\d.])", "1"), (r"≤", "<"), (r" \+ ", " * "))


def split_declaration(declaration: str) -> str | None:
    depth = 0
    index = 0
    while index < len(declaration) - 1:
        char = declaration[index]
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth -= 1
        elif char == ":" and declaration[index + 1] == "=" and depth == 0:
            remainder = declaration[index + 2 :]
            match = re.match(r"\s*by\b", remainder)
            if match:
                return declaration[: index + 2] + match.group(0)
            return None
        index += 1
    return None


def open_test_once() -> dict:
    SEALED.mkdir(parents=True, exist_ok=True)
    test_path = BENCHMARK / "random" / "test.json"
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "research" / "lean" / "acquisitions" / "leandojo_benchmark_4.json").read_text(encoding="utf-8"))
    manifest_hash = next(
        (item["sha256"] for item in manifest["files"] if item["path"] == "random/test.json"),
        None,
    )
    entries = json.loads(test_path.read_text(encoding="utf-8"))
    subset = []
    attempts = 0
    for entry in entries:
        if not entry.get("traced_tactics"):
            continue
        attempts += 1
        source_path = MATHLIB / entry["file_path"]
        if not source_path.exists():
            continue
        lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        declaration = "".join(lines[entry["start"][0] - 1 : entry["end"][0]])
        declaration_prefix = split_declaration(declaration)
        if declaration_prefix is None:
            continue
        subset.append(
            {
                "full_name": entry["full_name"],
                "file_path": entry["file_path"],
                "start": entry["start"],
                "end": entry["end"],
                "header": "".join(lines[: entry["start"][0] - 1]),
                "decl_prefix": declaration_prefix,
                "statement_sha256": hashlib.sha256(normalize_state(declaration_prefix).encode("utf-8")).hexdigest(),
                "first_state": entry["traced_tactics"][0]["state_before"],
                "first_tactic": entry["traced_tactics"][0]["tactic"],
                "steps": [tactic["state_before"] for tactic in entry["traced_tactics"]],
            }
        )
        if len(subset) >= SUBSET_SIZE:
            break
    payload = {
        "test_file": "random/test.json",
        "test_sha256_before": digest,
        "manifest_sha256": manifest_hash,
        "hash_matches_manifest": digest == manifest_hash,
        "attempts": attempts,
        "subset_size": len(subset),
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": subset,
    }
    subset_path = SEALED / "test_subset.json"
    subset_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["subset_sha256"] = hashlib.sha256(subset_path.read_bytes()).hexdigest()
    (SEALED / "opening.json").write_text(
        json.dumps({key: payload[key] for key in ("test_file", "test_sha256_before", "manifest_sha256", "hash_matches_manifest", "attempts", "subset_size", "opened_at", "subset_sha256")}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return payload


def load_subset() -> dict:
    return json.loads((SEALED / "test_subset.json").read_text(encoding="utf-8"))


def semantic_hash(entry: dict, context, label: str) -> str | None:
    auto = entry["file_path"].startswith("Mathlib/")
    source = build_semantic_source(entry["header"], entry["decl_prefix"], auto_implicit_false=auto)
    run = run_lean(source, SEALED / "scratch" / label, context, TIMEOUT, MEMORY_LIMIT)
    canonical = parse_canonical_type(run.stdout)
    return canonical_type_sha256(canonical) if canonical else None


def lh070(subset: dict, context) -> dict:
    deterministic = 0
    hashes = {}
    for index, entry in enumerate(subset["entries"]):
        first = semantic_hash(entry, context, f"hash_{index:02d}_a")
        second = semantic_hash(entry, context, f"hash_{index:02d}_b")
        hashes[entry["full_name"]] = first
        deterministic += int(first is not None and first == second)
    mutations = []
    for entry in subset["entries"][:4]:
        for pattern, replacement in MUTATIONS:
            mutated = re.sub(pattern, replacement, entry["decl_prefix"], count=1)
            if mutated == entry["decl_prefix"]:
                continue
            textual = hashlib.sha256(normalize_state(mutated).encode("utf-8")).hexdigest() != entry["statement_sha256"]
            mutations.append(textual)
    return {
        "statements": len(subset["entries"]),
        "semantic_deterministic": deterministic,
        "mutations_tested": len(mutations),
        "mutations_detected_textually": sum(1 for value in mutations if value),
    }


def lh026(subset: dict) -> dict:
    steps = [state for entry in subset["entries"] for state in entry["steps"]]
    pairs = []
    for seed in range(5):
        order = list(range(len(steps)))
        np.random.default_rng(seed).shuffle(order)
        seen: set[str] = set()
        for index in order:
            state = steps[index]
            goal_line = next((line for line in state.splitlines() if GOAL.match(line)), state)
            key = hashlib.sha256(normalize_state(goal_line).encode("utf-8")).hexdigest()
            pairs.append(key in seen)
            seen.add(key)
    rate = sum(1 for value in pairs if value) / len(pairs) if pairs else 0.0
    return {"steps": len(steps), "cache_hits": sum(1 for value in pairs if value), "accesses": len(pairs), "hit_rate": round(rate, 4)}


def classify(run) -> str:
    if run.status == "timeout":
        return "timeout"
    if run.status == "ok":
        return "closed"
    if "unsolved goals" in run.stdout:
        return "valid_incomplete"
    return "invalid"


def parse_axioms(stdout: str, name: str) -> list[str] | None:
    if re.search(rf"'([\w.']*\.)?{re.escape(name)}' does not depend on any axioms", stdout):
        return []
    match = re.search(rf"'([\w.']*\.)?{re.escape(name)}' depends on axioms: \[(.*?)\]", stdout, re.DOTALL)
    if not match:
        return None
    inner = match.group(2).strip()
    return [item.strip() for item in inner.split(",") if item.strip()] if inner else []


def lh069(subset: dict, context) -> dict:
    passes = []
    for pass_index in (1, 2):
        outcomes = []
        for index, entry in enumerate(subset["entries"]):
            lines = (MATHLIB / entry["file_path"]).read_text(encoding="utf-8").splitlines(keepends=True)
            declaration = "".join(lines[entry["start"][0] - 1 : entry["end"][0]])
            header = strip_to_additive(entry["header"])
            if not header.endswith("\n"):
                header += "\n"
            options = ["set_option autoImplicit false"] if entry["file_path"].startswith("Mathlib/") else []
            source = with_options(header, options) + declaration + f"\n#print axioms {entry['full_name']}\n"
            run = run_lean(source, SEALED / "replay" / f"pass{pass_index}_{index:02d}", context, TIMEOUT, MEMORY_LIMIT)
            axioms = parse_axioms(run.stdout, entry["full_name"]) if run.status == "ok" else None
            outcomes.append((run.status, axioms is not None and set(axioms) <= ALLOWED_AXIOMS))
        passes.append(outcomes)
    identical = sum(1 for left, right in zip(passes[0], passes[1]) if left == right)
    accepted = sum(1 for left in passes[0] if left[0] == "ok" and left[1])
    return {"cases": len(passes[0]), "accepted": accepted, "identical_across_passes": identical}


def symbolic_reference(subset: dict, context) -> dict:
    solved = []
    calls = 0
    wall = 0.0
    exclusive = []
    for index, entry in enumerate(subset["entries"]):
        header = strip_to_additive(entry["header"])
        if not header.endswith("\n"):
            header += "\n"
        options = ["set_option autoImplicit false"] if entry["file_path"].startswith("Mathlib/") else []
        base = with_options(header, options) + entry["decl_prefix"]
        closed = False
        for tactic in TACTICS:
            calls += 1
            run = run_lean(base + f"\n  {tactic}\n", SEALED / "symbolic" / f"{index:02d}_{tactic}", context, TIMEOUT, MEMORY_LIMIT)
            wall += run.wall_seconds
            if run.status == "ok":
                closed = True
                break
        solved.append(closed)
        exclusive.append(closed)
    return {"goals": len(subset["entries"]), "portfolio_solved": sum(1 for value in solved if value), "verifier_calls": calls, "wall_seconds": round(wall, 3)}


def tacgen_reference(subset: dict, context) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL)).eval()
    solved = []
    calls = 0
    wall = 0.0
    for index, entry in enumerate(subset["entries"]):
        inputs = tokenizer(entry["first_state"], max_length=2048, truncation=True, return_tensors="pt")
        started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=512, num_beams=1, do_sample=False, num_return_sequences=1)
        tactic = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].replace("<a>", "").replace("</a>", "")
        if scan_forbidden(tactic, ["sorry", "admit", "sorryAx"]):
            solved.append(False)
            continue
        header = strip_to_additive(entry["header"])
        if not header.endswith("\n"):
            header += "\n"
        options = ["set_option autoImplicit false"] if entry["file_path"].startswith("Mathlib/") else []
        source = with_options(header, options) + entry["decl_prefix"] + "\n" + "\n".join("  " + line for line in tactic.splitlines()) + "\n"
        run = run_lean(source, SEALED / "tacgen" / f"{index:02d}", context, TIMEOUT, MEMORY_LIMIT)
        wall += time.monotonic() - started
        calls += 1
        solved.append(run.status == "ok")
    return {"goals": len(subset["entries"]), "pass_at_1": sum(1 for value in solved if value), "verifier_calls": calls, "wall_seconds": round(wall, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description="LTP-09 teste selado")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    opened = json.loads((SEALED / "opening.json").read_text(encoding="utf-8")) if (SEALED / "opening.json").exists() else None
    if opened is not None:
        print("AVISO: abertura já registrada; reutilizando subconjunto selado sem reabrir test.json")
        subset = load_subset()
    else:
        subset = open_test_once()
    context = resolve_lean_context(NATIVE)
    started = time.monotonic()
    results = {
        "opening": {key: value for key, value in subset.items() if key != "entries"},
        "LH-070": lh070(subset, context),
        "LH-026": lh026(subset),
        "LH-069": lh069(subset, context),
        "reference_symbolic": symbolic_reference(subset, context),
        "reference_reprover_tacgen": tacgen_reference(subset, context),
    }
    results["wall_seconds"] = round(time.monotonic() - started, 3)
    results["peak_rss_bytes"] = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024
    payload = {
        "phase": "LTP-09",
        "finalists": ["LH-070", "LH-069", "LH-026"],
        "sealed_split": "random/test.json",
        "untouched_split": "novel_premises/test.json",
        "metrics": results,
        "no_reopen": True,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
