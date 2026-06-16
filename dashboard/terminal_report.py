"""ASCII benchmark report for the terminal — 90s BBS style."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.benchmark_data import best_row, load_records, to_leaderboard_rows

GREEN = "\033[32m"
AMBER = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _bar(value: float, width: int = 20) -> str:
    filled = int(round((value / 100) * width))
    return "█" * filled + "░" * (width - filled)


def print_benchmark_report() -> None:
    records = load_records()
    rows = to_leaderboard_rows(records)

    print()
    print(f"{GREEN}{BOLD}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{GREEN}{BOLD}║  QUANTUN-IA BENCHMARK TERMINAL v1.0          [C] 1996 ML SYSTEMS  ║{RESET}")
    print(f"{GREEN}{BOLD}╠══════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{GREEN}║  STATUS: {'ONLINE' if rows else 'NO DATA':<10}   LOG: logs/experiments.jsonl{' ' * 18}║{RESET}")
    print(f"{GREEN}{BOLD}╚══════════════════════════════════════════════════════════════════╝{RESET}")
    print()

    if not rows:
        print(f"{AMBER}  >> No benchmarks found. Run an experiment first.{RESET}")
        print(f"{DIM}     python experiments/exp_001_quantum_vs_classical/run.py{RESET}")
        print()
        return

    header = f"  {'EXP':<12} {'MODEL':<24} {'ACC%':>6} {'LOSS':>8} {'TIME':>6} {'EP':>4}"
    print(f"{CYAN}{header}{RESET}")
    print(f"{DIM}  {'-' * 12} {'-' * 24} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 4}{RESET}")
    print(f"{DIM}  (multi-seed holdout mean · bootstrap 95% CI when available){RESET}")

    for r in sorted(rows, key=lambda x: x["accuracy"] or 0, reverse=True):
        acc = f"{r['accuracy']:>5.1f}" if r["accuracy"] is not None else "   n/a"
        loss = f"{r['loss']:>8.4f}" if r["loss"] is not None else "     n/a"
        print(
            f"  {r['exp_id']:<12} {r['model']:<24} {acc} {loss} "
            f"{r['elapsed_s']:>5.2f}s {r['epochs']:>4}"
        )

    best = best_row(rows)
    print()
    if best:
        print(f"{AMBER}{BOLD}  ★ LEADERBOARD #1{RESET}")
        print(f"{AMBER}    Model    : {best['model']}{RESET}")
        print(f"{AMBER}    Accuracy : {best['accuracy']:.1f}%  {_bar(best['accuracy'])}{RESET}")
        print(f"{AMBER}    Experiment: {best['exp_id']}{RESET}")

    print()
    print(f"{DIM}  >> Dashboard: http://localhost:8501{RESET}")
    print()


if __name__ == "__main__":
    print_benchmark_report()
