#!/usr/bin/env python3
"""Harness adversarial do programa Lean Tiny Prover (fase LTP-00).

Cada candidato passa por quatro estágios independentes:

1. hash canônico da afirmação contra o registro congelado;
2. scanner de tokens proibidos, ignorando comentários e strings;
3. compilação em processo limpo, diretório temporário próprio, namespace de
   rede isolado, limites de CPU/memória e timeout explícito;
4. auditoria formal de axiomas via `#print axioms`.

O harness certifica que a prova fecha a afirmação registrada sem atalhos
proibidos. Ele não avalia superioridade científica de modelo algum.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEAN_PROJECT = ROOT / "lean"
DEFAULT_SMOKE = ROOT / "research" / "lean" / "smoke" / "smoke32.json"
DEFAULT_ADVERSARIAL = ROOT / "research" / "lean" / "smoke" / "adversarial.json"
DEFAULT_RUN_ROOT = ROOT / ".local" / "runs" / "lean" / "LTP-00"
TARGET_NAME = "ltp_target"
DEFAULT_MEMORY_LIMIT_BYTES = 8 * 1024**3
CPU_LIMIT_GRACE_SECONDS = 30
WORD_CHAR = r"[\w']"
REJECT_REASONS = (
    "statement_hash_mismatch",
    "forbidden_token",
    "lean_error",
    "timeout",
    "memory_limit",
    "missing_axiom_report",
    "extra_axiom",
    "spawn_error",
)


@dataclass
class LeanContext:
    lean_bin: str
    lean_path: str
    toolchain: str


@dataclass
class Case:
    id: str
    kind: str
    expectation: str
    statement: str
    statement_sha256: str | None
    proof: str
    preamble: str = ""
    timeout_seconds: float = 120.0
    expected_reason: str | None = None
    tampered_statement: str | None = None
    base: str | None = None
    mutation: str | None = None


@dataclass
class LeanRun:
    status: str
    exit_code: int | None
    wall_seconds: float
    peak_rss_bytes: int
    stdout: str
    stderr: str
    command: list[str]


def elan_home() -> Path:
    return Path(os.environ.get("ELAN_HOME", str(Path.home() / ".elan")))


def resolve_tool(name: str) -> str:
    candidate = elan_home() / "bin" / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"ferramenta não encontrada: {name}")


def _capture(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"comando falhou ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return result.stdout


def resolve_lean_context(project: Path = LEAN_PROJECT) -> LeanContext:
    lake = resolve_tool("lake")
    env = os.environ.copy()
    env["PATH"] = f"{elan_home() / 'bin'}:{env.get('PATH', '')}"
    raw_path_lines = _capture([lake, "env", "printenv", "LEAN_PATH"], project, env).strip().splitlines()
    lean_bin_lines = _capture([lake, "env", "which", "lean"], project, env).strip().splitlines()
    if not raw_path_lines or not lean_bin_lines:
        raise RuntimeError("não foi possível resolver LEAN_PATH/lean via lake")
    entries = []
    for entry in raw_path_lines[-1].split(":"):
        if not entry:
            continue
        candidate = Path(entry)
        entries.append(str((project / candidate).resolve()) if not candidate.is_absolute() else str(candidate))
    toolchain = (project / "lean-toolchain").read_text(encoding="utf-8").strip()
    return LeanContext(lean_bin=lean_bin_lines[-1], lean_path=":".join(entries), toolchain=toolchain)


def strip_lean(text: str, drop_strings: bool) -> str:
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        nxt = text[index + 1] if index + 1 < length else ""
        if char == "-" and nxt == "-":
            newline = text.find("\n", index)
            index = length if newline == -1 else newline
            out.append(" ")
            continue
        if char == "/" and nxt == "-":
            depth = 1
            index += 2
            while index < length and depth > 0:
                if text[index] == "/" and index + 1 < length and text[index + 1] == "-":
                    depth += 1
                    index += 2
                elif text[index] == "-" and index + 1 < length and text[index + 1] == "/":
                    depth -= 1
                    index += 2
                else:
                    index += 1
            out.append(" ")
            continue
        if char == '"':
            if drop_strings:
                index += 1
                while index < length:
                    if text[index] == "\\":
                        index += 2
                        continue
                    if text[index] == '"':
                        index += 1
                        break
                    index += 1
                out.append('""')
                continue
            out.append(char)
            index += 1
            while index < length:
                out.append(text[index])
                if text[index] == "\\" and index + 1 < length:
                    out.append(text[index + 1])
                    index += 2
                    continue
                if text[index] == '"':
                    index += 1
                    break
                index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def normalize_statement(text: str) -> str:
    code = strip_lean(text, drop_strings=False)
    code = unicodedata.normalize("NFC", code)
    return re.sub(r"\s+", " ", code).strip()


def statement_sha256(text: str) -> str:
    return hashlib.sha256(normalize_statement(text).encode("utf-8")).hexdigest()


SEMANTIC_TARGET = "ltp_target"
DECL_KEYWORD = re.compile(r"\b(theorem|lemma|example|instance|def)\b")


def rename_declaration(declaration: str, new_name: str) -> str:
    keyword = DECL_KEYWORD.search(declaration)
    if keyword is None or keyword.group(1) == "example":
        return declaration
    start = keyword.end()
    while start < len(declaration) and declaration[start].isspace():
        start += 1
    end = start
    while end < len(declaration) and (declaration[end].isalnum() or declaration[end] in "._'!?"):
        end += 1
    if end == start:
        return declaration
    return declaration[:start] + new_name + declaration[end:]


TO_ADDITIVE_LINE = re.compile(r"^\s*@\[[^\]]*to_additive[^\]]*\]\s*$", re.MULTILINE)


def strip_to_additive(header: str) -> str:
    return TO_ADDITIVE_LINE.sub("", header)


def with_options(header: str, options: list[str]) -> str:
    lines = header.splitlines(keepends=True)
    insert_at = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "public import ", "import all ")) or stripped == "prelude":
            insert_at = index + 1
    options_text = "".join(option + "\n" for option in options)
    return "".join(lines[:insert_at]) + options_text + "".join(lines[insert_at:])


def build_semantic_source(
    header: str,
    decl_prefix: str,
    target: str = SEMANTIC_TARGET,
    auto_implicit_false: bool = True,
) -> str:
    declaration = rename_declaration(decl_prefix, target)
    separator = "" if header.endswith("\n") else "\n"
    options = ["set_option autoImplicit false"] if auto_implicit_false else []
    prepared = with_options(strip_to_additive(header) + separator, options)
    if not prepared.endswith("\n"):
        prepared += "\n"
    return (
        prepared
        + declaration
        + "\n  sorry\n"
        + "set_option pp.all true\n"
        + f"#check @{target}\n"
    )


def parse_canonical_type(stdout: str, target: str = SEMANTIC_TARGET) -> str | None:
    pattern = re.compile(rf"@?([\w.']*\.)?{re.escape(target)}(\.\{{[^}}]*\}})?\s*:\s*")
    matches = list(pattern.finditer(stdout))
    if not matches:
        return None
    remainder = stdout[matches[-1].end():]
    canonical = re.sub(r"\s+", " ", remainder).strip()
    return canonical or None


def canonical_type_sha256(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_forbidden(code: str, tokens: list[str]) -> list[str]:
    cleaned = strip_lean(code, drop_strings=True)
    found: list[str] = []
    for token in tokens:
        pattern = rf"(?<!{WORD_CHAR}){re.escape(token)}(?!{WORD_CHAR})"
        if re.search(pattern, cleaned):
            found.append(token)
    return found


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_suites(smoke_path: Path, adversarial_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    smoke = load_json(smoke_path)
    adversarial = load_json(adversarial_path)
    return smoke, adversarial


def validate_smoke(smoke: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    theorems = smoke.get("theorems", [])
    if len(theorems) != 32:
        errors.append(f"esperados 32 teoremas, encontrados {len(theorems)}")
    expected_ids = [f"SMOKE-{index:03d}" for index in range(1, 33)]
    if [item.get("id") for item in theorems] != expected_ids:
        errors.append("IDs do smoke devem cobrir SMOKE-001…SMOKE-032 em ordem")
    for item in theorems:
        identifier = item.get("id", "<sem-id>")
        for field_name in ("statement", "proof", "statement_sha256"):
            if not item.get(field_name):
                errors.append(f"{identifier}: campo {field_name} ausente")
        if item.get("statement_sha256") and item["statement_sha256"] != statement_sha256(item["statement"]):
            errors.append(f"{identifier}: statement_sha256 não confere")
    policy = smoke.get("policy", {})
    for field_name in ("allowed_axioms", "forbidden_tokens", "timeout_seconds", "max_heartbeats"):
        if not policy.get(field_name):
            errors.append(f"policy: campo {field_name} ausente")
    return errors


def validate_adversarial(adversarial: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not adversarial.get("mutations"):
        errors.append("suíte adversarial sem mutações")
    if not adversarial.get("specials"):
        errors.append("suíte adversarial sem casos especiais")
    valid_reasons = set(REJECT_REASONS) | {None}
    for case in adversarial.get("specials", []):
        if case.get("expect") not in ("accept", "reject"):
            errors.append(f"{case.get('id')}: expect inválido")
        if case.get("reason") not in valid_reasons:
            errors.append(f"{case.get('id')}: reason inválido")
        if case.get("expect") == "reject" and not case.get("reason"):
            errors.append(f"{case.get('id')}: caso reject sem reason")
    return errors


def seal_smoke(smoke_path: Path) -> None:
    smoke = load_json(smoke_path)
    for item in smoke["theorems"]:
        item["statement_sha256"] = statement_sha256(item["statement"])
    write_json(smoke_path, smoke)


def materialize_cases(smoke: dict[str, Any], adversarial: dict[str, Any]) -> list[Case]:
    default_timeout = float(smoke["policy"]["timeout_seconds"])
    by_id = {item["id"]: item for item in smoke["theorems"]}
    cases: list[Case] = []
    for item in smoke["theorems"]:
        cases.append(
            Case(
                id=f"{item['id']}-VALID",
                kind="sealed_valid",
                expectation="accept",
                statement=item["statement"],
                statement_sha256=item["statement_sha256"],
                proof=item["proof"],
                timeout_seconds=default_timeout,
                base=item["id"],
            )
        )
    for mutation in adversarial["mutations"]:
        targets = list(by_id) if mutation["applies_to"] == "all" else list(mutation["applies_to"])
        for theorem_id in targets:
            base = by_id.get(theorem_id)
            if base is None:
                continue
            if mutation["kind"] == "comment_disguised_forbidden":
                proof = mutation.get("proof_prefix", "") + base["proof"]
            else:
                proof = mutation.get("proof_template", "")
            cases.append(
                Case(
                    id=f"{theorem_id}-{mutation['kind'].upper()}",
                    kind="mutation",
                    expectation=mutation["expect"],
                    statement=base["statement"],
                    statement_sha256=base["statement_sha256"],
                    proof=proof,
                    timeout_seconds=default_timeout,
                    expected_reason=mutation.get("reason"),
                    base=theorem_id,
                    mutation=mutation["kind"],
                )
            )
    for special in adversarial["specials"]:
        base = by_id.get(special.get("base", ""))
        if special.get("kind") == "statement_tamper":
            if base is None:
                continue
            cases.append(
                Case(
                    id=special["id"],
                    kind="statement_tamper",
                    expectation=special["expect"],
                    statement=base["statement"],
                    statement_sha256=base["statement_sha256"],
                    proof=special["proof"],
                    timeout_seconds=float(special.get("timeout_seconds", default_timeout)),
                    expected_reason=special.get("reason"),
                    tampered_statement=special["tampered_statement"],
                    base=base["id"],
                )
            )
            continue
        statement = special.get("statement") or (base or {}).get("statement")
        if statement is None:
            continue
        cases.append(
            Case(
                id=special["id"],
                kind="special",
                expectation=special["expect"],
                statement=statement,
                statement_sha256=statement_sha256(statement),
                proof=special["proof"],
                preamble=special.get("preamble", ""),
                timeout_seconds=float(special.get("timeout_seconds", default_timeout)),
                expected_reason=special.get("reason"),
                base=(base or {}).get("id"),
            )
        )
    return cases


def build_source(case: Case, smoke: dict[str, Any]) -> str:
    statement = case.tampered_statement or case.statement
    lines = [f"import {name}" for name in smoke["imports"]]
    lines.append("set_option autoImplicit false")
    lines.append(f"set_option maxHeartbeats {smoke['policy']['max_heartbeats']}")
    if case.preamble.strip():
        lines.append(case.preamble.strip())
    lines.append(f"theorem {TARGET_NAME} : {statement} := by")
    proof = case.proof.strip("\n")
    lines.append(textwrap.indent(proof, "  ") if proof.strip() else "  ")
    lines.append(f"#print axioms {TARGET_NAME}")
    return "\n".join(lines) + "\n"


def read_vmhwm(pid: int) -> tuple[int, int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            peak = 0
            current = 0
            for line in handle:
                if line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) * 1024
                elif line.startswith("VmRSS:"):
                    current = int(line.split()[1]) * 1024
            return peak, current
    except OSError:
        return 0, 0


def run_lean(
    source: str,
    workdir: Path,
    context: LeanContext,
    timeout_seconds: float,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> LeanRun:
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    candidate = workdir / "Candidate.lean"
    candidate.write_text(source, encoding="utf-8")
    unshare = shutil.which("unshare")
    command: list[str] = []
    if unshare and os.environ.get("LTP_NO_NETNS", "0") != "1":
        command.extend([unshare, "-rn"])
    command.extend([context.lean_bin, str(candidate)])
    child_env = {
        "PATH": f"{elan_home() / 'bin'}:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", str(Path.home())),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LEAN_PATH": context.lean_path,
    }
    cpu_limit = int(timeout_seconds) + CPU_LIMIT_GRACE_SECONDS
    address_limit = memory_limit_bytes * 3

    def prepare_child() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + CPU_LIMIT_GRACE_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))

    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(workdir),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
            preexec_fn=prepare_child,
        )
    except OSError as error:
        return LeanRun("spawn_error", None, 0.0, 0, "", str(error), command)

    state = {"peak_rss": 0, "memory_exceeded": False, "status": None}

    def monitor() -> None:
        while process.poll() is None:
            peak, current = read_vmhwm(process.pid)
            state["peak_rss"] = max(state["peak_rss"], peak)
            if current > memory_limit_bytes:
                state["memory_exceeded"] = True
                state["status"] = "memory_limit"
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
            time.sleep(0.05)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    status = "ok"
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        status = "timeout"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    watcher.join(timeout=2)
    wall = time.monotonic() - started
    if state["status"] == "memory_limit":
        status = "memory_limit"
    elif status == "ok" and process.returncode != 0:
        status = "lean_error"
    return LeanRun(status, process.returncode, wall, state["peak_rss"], stdout or "", stderr or "", command)


AXIOM_DEPENDS = "depends on axioms:"
AXIOM_FREE = "does not depend on any axioms"


def parse_axioms(stdout: str, target: str = TARGET_NAME) -> list[str] | None:
    if f"'{target}' {AXIOM_FREE}" in stdout:
        return []
    pattern = rf"'{re.escape(target)}' {AXIOM_DEPENDS} \[(.*?)\]"
    match = re.search(pattern, stdout, re.DOTALL)
    if not match:
        return None
    inner = match.group(1).strip()
    if not inner:
        return []
    return [item.strip() for item in inner.split(",") if item.strip()]


def verify_case(
    case: Case,
    smoke: dict[str, Any],
    run_dir: Path,
    context: LeanContext,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "case_id": case.id,
        "kind": case.kind,
        "base": case.base,
        "mutation": case.mutation,
        "expectation": case.expectation,
        "expected_reason": case.expected_reason,
        "verdict": "reject",
        "reason": None,
        "forbidden_tokens": [],
        "axioms": None,
        "statement_sha256": case.statement_sha256,
        "lean_invoked": False,
        "wall_seconds": None,
        "peak_rss_bytes": None,
        "exit_code": None,
        "transcript": None,
    }
    effective_statement = case.tampered_statement or case.statement
    if case.statement_sha256 is not None:
        actual = statement_sha256(effective_statement)
        if actual != case.statement_sha256:
            verdict["reason"] = "statement_hash_mismatch"
            return verdict
    found = scan_forbidden(case.preamble + "\n" + case.proof, smoke["policy"]["forbidden_tokens"])
    if found:
        verdict["verdict"] = "reject"
        verdict["reason"] = "forbidden_token"
        verdict["forbidden_tokens"] = found
        return verdict
    source = build_source(case, smoke)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / f"{case.id}.lean").write_text(source, encoding="utf-8")
    run = run_lean(
        source,
        run_dir / "scratch" / case.id,
        context,
        case.timeout_seconds,
        memory_limit_bytes,
    )
    verdict["lean_invoked"] = True
    verdict["wall_seconds"] = round(run.wall_seconds, 3)
    verdict["peak_rss_bytes"] = run.peak_rss_bytes
    verdict["exit_code"] = run.exit_code
    transcripts_dir = run_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / f"{case.id}.out").write_text(run.stdout, encoding="utf-8")
    (transcripts_dir / f"{case.id}.err").write_text(run.stderr, encoding="utf-8")
    verdict["transcript"] = f"transcripts/{case.id}.out"
    if run.status != "ok":
        verdict["reason"] = run.status
        return verdict
    axioms = parse_axioms(run.stdout)
    verdict["axioms"] = axioms
    if axioms is None:
        verdict["reason"] = "missing_axiom_report"
        return verdict
    allowed = set(smoke["policy"]["allowed_axioms"])
    extra = [axiom for axiom in axioms if axiom not in allowed]
    if extra:
        verdict["reason"] = "extra_axiom"
        verdict["extra_axioms"] = extra
        return verdict
    verdict["verdict"] = "accept"
    verdict["reason"] = None
    proofs_dir = run_dir / "proofs"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    (proofs_dir / f"{case.id}.lean").write_text(source, encoding="utf-8")
    return verdict


def evaluate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for result in results:
        if result["verdict"] != result["expectation"]:
            failures.append(
                {
                    "case_id": result["case_id"],
                    "expected": result["expectation"],
                    "observed": result["verdict"],
                    "reason": result["reason"],
                }
            )
        elif result["expectation"] == "reject" and result.get("expected_reason"):
            if result["reason"] != result["expected_reason"]:
                failures.append(
                    {
                        "case_id": result["case_id"],
                        "expected_reason": result["expected_reason"],
                        "observed_reason": result["reason"],
                    }
                )
    return {"passed": not failures, "failures": failures, "cases": len(results)}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def summarize_metrics(results: list[dict[str, Any]], wall_total: float) -> dict[str, Any]:
    lean_times = [item["wall_seconds"] for item in results if item.get("wall_seconds") is not None]
    peaks = [item["peak_rss_bytes"] for item in results if item.get("peak_rss_bytes")]
    reasons: dict[str, int] = {}
    for item in results:
        key = item["reason"] or item["verdict"]
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "cases_total": len(results),
        "expected_accept": sum(1 for item in results if item["expectation"] == "accept"),
        "expected_reject": sum(1 for item in results if item["expectation"] == "reject"),
        "accepted": sum(1 for item in results if item["verdict"] == "accept"),
        "rejected": sum(1 for item in results if item["verdict"] == "reject"),
        "lean_calls": sum(1 for item in results if item["lean_invoked"]),
        "wall_total_seconds": round(wall_total, 3),
        "wall_lean_p50_seconds": percentile(lean_times, 0.5),
        "wall_lean_p95_seconds": percentile(lean_times, 0.95),
        "peak_rss_max_bytes": max(peaks) if peaks else None,
        "reason_counts": dict(sorted(reasons.items())),
    }


def git_snapshot() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        return result.stdout.strip()

    return {
        "head": run_git(["rev-parse", "HEAD"]),
        "status_short": run_git(["status", "--short"]).splitlines(),
    }


def collect_environment() -> dict[str, Any]:
    def capture(command: list[str], cwd: Path = ROOT) -> str | None:
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
            output = result.stdout.strip()
            return output or None
        except (OSError, subprocess.TimeoutExpired):
            return None

    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    meminfo = Path("/proc/meminfo")
    mem_total = None
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
                break
    gpu = capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ]
    )
    manifest_path = LEAN_PROJECT / "lake-manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    mathlib_pin = next(
        (
            {"name": package.get("name"), "rev": package.get("rev"), "inputRev": package.get("inputRev")}
            for package in manifest.get("packages", [])
            if package.get("name") == "mathlib"
        ),
        None,
    )
    env = os.environ.copy()
    env["PATH"] = f"{elan_home() / 'bin'}:{env.get('PATH', '')}"
    registries = {
        name: file_sha256(ROOT / "research" / "lean" / name)
        for name in ("hypotheses.json", "datasets.json", "baselines.json")
    }
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": {"model": cpu_model, "logical_cores": os.cpu_count()},
        "memory_total_bytes": mem_total,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "gpu": gpu,
        "tools": {
            "elan": capture([str(elan_home() / "bin" / "elan"), "--version"], LEAN_PROJECT),
            "lean": capture([str(elan_home() / "bin" / "lean"), "--version"], LEAN_PROJECT),
            "lake": capture([str(elan_home() / "bin" / "lake"), "--version"], LEAN_PROJECT),
        },
        "pins": {
            "lean_toolchain": (LEAN_PROJECT / "lean-toolchain").read_text(encoding="utf-8").strip(),
            "lean_toolchain_sha256": file_sha256(LEAN_PROJECT / "lean-toolchain"),
            "lakefile_sha256": file_sha256(LEAN_PROJECT / "lakefile.toml"),
            "lake_manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else None,
            "mathlib": mathlib_pin,
        },
        "program_registries_sha256": registries,
        "network_namespace_isolation": bool(shutil.which("unshare")) and os.environ.get("LTP_NO_NETNS", "0") != "1",
    }


def write_statements_artifact(smoke: dict[str, Any], run_dir: Path) -> None:
    lines = [f"{item['statement_sha256']}  {item['id']}" for item in smoke["theorems"]]
    (run_dir / "statements.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    normalized = [f"{item['id']}\t{normalize_statement(item['statement'])}" for item in smoke["theorems"]]
    (run_dir / "statements.tsv").write_text("\n".join(normalized) + "\n", encoding="utf-8")


def run_pass(
    cases: list[Case],
    smoke: dict[str, Any],
    run_dir: Path,
    context: LeanContext,
    memory_limit_bytes: int,
) -> tuple[list[dict[str, Any]], float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("scratch", "candidates", "proofs", "transcripts"):
        shutil.rmtree(run_dir / name, ignore_errors=True)
    started = time.monotonic()
    results = [verify_case(case, smoke, run_dir, context, memory_limit_bytes) for case in cases]
    wall = time.monotonic() - started
    write_json(run_dir / "results.json", results)
    write_json(run_dir / "metrics.json", summarize_metrics(results, wall))
    return results, wall


def run_gate(args: argparse.Namespace) -> int:
    smoke, adversarial = load_suites(Path(args.smoke), Path(args.adversarial))
    smoke_errors = validate_smoke(smoke)
    adversarial_errors = validate_adversarial(adversarial)
    errors = smoke_errors + adversarial_errors
    if errors:
        for error in errors:
            print(f"ERRO: {error}", file=sys.stderr)
        return 1
    context = resolve_lean_context()
    run_root = Path(args.run_root)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = materialize_cases(smoke, adversarial)
    write_json(run_dir / "environment.json", collect_environment())
    write_json(
        run_dir / "run.json",
        {
            "phase": "LTP-00",
            "run_id": run_id,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": sys.argv,
            "git": git_snapshot(),
            "suites": {
                "smoke": {"path": str(args.smoke), "sha256": file_sha256(Path(args.smoke))},
                "adversarial": {"path": str(args.adversarial), "sha256": file_sha256(Path(args.adversarial))},
            },
            "policy": smoke["policy"],
            "imports": smoke["imports"],
            "target_name": TARGET_NAME,
            "lean": {"toolchain": context.toolchain, "lean_bin": context.lean_bin},
            "sandbox": {
                "network_namespace": bool(shutil.which("unshare")) and os.environ.get("LTP_NO_NETNS", "0") != "1",
                "memory_limit_bytes": args.memory_limit_bytes,
                "cpu_limit_grace_seconds": CPU_LIMIT_GRACE_SECONDS,
            },
            "replay_enabled": args.replay,
        },
    )
    write_statements_artifact(smoke, run_dir)
    write_json(run_dir / "cases.json", [asdict(case) for case in cases])
    results, wall = run_pass(cases, smoke, run_dir, context, args.memory_limit_bytes)
    gate = evaluate_results(results)
    replay_comparison = None
    if args.replay:
        replay_results, replay_wall = run_pass(
            cases,
            smoke,
            run_dir / "replay",
            context,
            args.memory_limit_bytes,
        )
        replay_gate = evaluate_results(replay_results)
        mismatches = [
            {"case_id": primary["case_id"], "primary": primary["verdict"], "replay": replay["verdict"]}
            for primary, replay in zip(results, replay_results)
            if primary["verdict"] != replay["verdict"] or primary["reason"] != replay["reason"]
        ]
        replay_comparison = {
            "identical": not mismatches,
            "mismatches": mismatches,
            "replay_wall_total_seconds": round(replay_wall, 3),
            "replay_gate": replay_gate,
        }
        write_json(run_dir / "replay" / "results.json", replay_results)
        write_json(run_dir / "replay" / "metrics.json", summarize_metrics(replay_results, replay_wall))
        write_json(run_dir / "replay_comparison.json", replay_comparison)
        if not replay_comparison["identical"] or not replay_gate["passed"]:
            gate["passed"] = False
            gate["failures"].append({"replay": "divergência ou falha no replay limpo", "detail": replay_comparison})
    metrics = summarize_metrics(results, wall)
    decision = {
        "phase": "LTP-00",
        "gate": "100% das provas válidas aceitas em ambiente limpo e 100% das provas deliberadamente inválidas rejeitadas",
        "passed": gate["passed"],
        "criteria": {
            "all_expected_accept_accepted": all(
                item["verdict"] == "accept" for item in results if item["expectation"] == "accept"
            ),
            "all_expected_reject_rejected": all(
                item["verdict"] == "reject" for item in results if item["expectation"] == "reject"
            ),
            "expected_reasons_matched": not any(
                failure.get("observed_reason") for failure in gate["failures"]
            ),
            "replay_identical": None if replay_comparison is None else replay_comparison["identical"],
        },
        "failures": gate["failures"],
        "metrics": metrics,
        "limitations": [
            "A auditoria de axiomas cobre o termo da prova; ela não audita a fidelidade da afirmação formal ao enunciado em linguagem natural.",
            "O scanner de tokens é uma camada de rejeição antecipada; a garantia de ausência de atalhos vem do exit code e do relatório `#print axioms`.",
            "O isolamento usa namespace de rede unprivilegiado (`unshare -rn`) e não é uma sandbox de sistema completa.",
            "Cada caso tem tempo de parede e pico de RSS registrados, mas não medição de energia.",
            "Os 32 teoremas são internos e não substituem avaliação em miniF2F ou outros benchmarks.",
        ],
    }
    write_json(run_dir / "decision.json", decision)
    print(json.dumps({"run_dir": str(run_dir), "passed": gate["passed"], "metrics": metrics}, indent=2, ensure_ascii=False))
    return 0 if gate["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harness adversarial do Lean Tiny Prover")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-suite", help="valida os registros do smoke e adversarial")
    validate_parser.add_argument("--smoke", default=str(DEFAULT_SMOKE))
    validate_parser.add_argument("--adversarial", default=str(DEFAULT_ADVERSARIAL))

    seal_parser = subparsers.add_parser("seal-suite", help="calcula e grava statement_sha256 do smoke")
    seal_parser.add_argument("--smoke", default=str(DEFAULT_SMOKE))

    hash_parser = subparsers.add_parser("hash-statement", help="imprime o hash canônico de uma afirmação")
    hash_parser.add_argument("statement")

    case_parser = subparsers.add_parser("build-case", help="gera o arquivo Lean de um caso")
    case_parser.add_argument("--smoke", default=str(DEFAULT_SMOKE))
    case_parser.add_argument("--adversarial", default=str(DEFAULT_ADVERSARIAL))
    case_parser.add_argument("--case-id", required=True)
    case_parser.add_argument("--out", required=True)

    gate_parser = subparsers.add_parser("run-gate", help="executa o gate adversarial completo")
    gate_parser.add_argument("--smoke", default=str(DEFAULT_SMOKE))
    gate_parser.add_argument("--adversarial", default=str(DEFAULT_ADVERSARIAL))
    gate_parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    gate_parser.add_argument("--run-id", default=None)
    gate_parser.add_argument("--memory-limit-bytes", type=int, default=DEFAULT_MEMORY_LIMIT_BYTES)
    gate_parser.add_argument("--replay", action="store_true", help="reexecuta toda a suíte em ambiente limpo")

    environment_parser = subparsers.add_parser("environment", help="grava o registro de ambiente")
    environment_parser.add_argument("--out", required=True)

    scanner_parser = subparsers.add_parser("scan", help="escaneia um arquivo em busca de tokens proibidos")
    scanner_parser.add_argument("--smoke", default=str(DEFAULT_SMOKE))
    scanner_parser.add_argument("file")

    args = parser.parse_args(argv)
    if args.command == "validate-suite":
        smoke, adversarial = load_suites(Path(args.smoke), Path(args.adversarial))
        errors = validate_smoke(smoke) + validate_adversarial(adversarial)
        if errors:
            for error in errors:
                print(f"ERRO: {error}", file=sys.stderr)
            return 1
        print("OK: smoke32 e suíte adversarial estruturalmente válidos")
        return 0
    if args.command == "seal-suite":
        seal_smoke(Path(args.smoke))
        print(f"selado: {args.smoke}")
        return 0
    if args.command == "hash-statement":
        print(statement_sha256(args.statement))
        return 0
    if args.command == "build-case":
        smoke, adversarial = load_suites(Path(args.smoke), Path(args.adversarial))
        cases = {case.id: case for case in materialize_cases(smoke, adversarial)}
        case = cases.get(args.case_id)
        if case is None:
            print(f"ERRO: caso inexistente: {args.case_id}", file=sys.stderr)
            return 1
        Path(args.out).write_text(build_source(case, smoke), encoding="utf-8")
        print(f"gerado: {args.out}")
        return 0
    if args.command == "run-gate":
        return run_gate(args)
    if args.command == "environment":
        write_json(Path(args.out), collect_environment())
        print(f"gravado: {args.out}")
        return 0
    if args.command == "scan":
        smoke = load_json(Path(args.smoke))
        code = Path(args.file).read_text(encoding="utf-8")
        found = scan_forbidden(code, smoke["policy"]["forbidden_tokens"])
        if found:
            print(f"PROIBIDO: {', '.join(found)}")
            return 1
        print("OK: nenhum token proibido")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
