"""
EXP 101 — Open image corpus ingest verification (Phase G).

CPU synthesis — no GPU training:
  MLFLOW_DISABLE=1 python experiments/exp_101_open_image_corpus_ingest/run.py \\
    --profile publication --write-results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.open_images import (
    PACK_LOADERS,
    is_pack_complete,
    load_image_pack_arrays,
    summarize_packs,
)
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_101_open_image_corpus_ingest"
EXP_ID = "exp_101"
ROOT = Path(__file__).resolve().parents[2]
IMAGES_ROOT = ROOT / "data" / "open" / "images"

EXPECTED_SHAPES = {
    "cifar10": (32, 32, 3),
    "fashion_mnist": (28, 28),
    "flowers102": (64, 64, 3),
}


@dataclass(frozen=True)
class OpenImageIngestResult:
    n_packs_complete: int
    n_packs_expected: int
    n_splits_ready: int
    smoke_ok: bool
    pack_status: tuple[dict, ...]
    elapsed_s: float
    profile: str
    hypothesis_confirmed: bool


def _processed_stats(pack: str) -> Path:
    return IMAGES_ROOT / pack / "processed" / "v1" / "stats.json"


def _split_npz(pack: str) -> Path:
    return IMAGES_ROOT / pack / "processed" / "v1" / "split_indices.npz"


def run_exp_101(*, profile: str = "ci", verbose: bool = True) -> OpenImageIngestResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    n_smoke = int(cfg.get("n_smoke_rows", 8))
    init_correlation_id()
    t0 = time.perf_counter()

    statuses: list[dict] = []
    smoke_ok = True
    for pack in PACK_LOADERS:
        complete = is_pack_complete(pack)
        stats_ok = _processed_stats(pack).is_file() and _split_npz(pack).is_file()
        shape_ok = False
        err = None
        if complete:
            try:
                batch = load_image_pack_arrays(pack, n_train=n_smoke, n_test=n_smoke)
                shape_ok = tuple(batch["spatial_shape"]) == EXPECTED_SHAPES[pack]
                if not shape_ok:
                    smoke_ok = False
            except Exception as exc:  # noqa: BLE001
                smoke_ok = False
                err = str(exc)
        else:
            smoke_ok = False
        statuses.append(
            {
                "pack": pack,
                "complete": complete,
                "stats_ok": stats_ok,
                "shape_ok": shape_ok,
                "error": err,
            }
        )
        if verbose:
            print(f"{pack}: complete={complete} stats={stats_ok} shape={shape_ok} err={err}")

    n_complete = sum(1 for s in statuses if s["complete"])
    n_splits = sum(1 for s in statuses if s["stats_ok"])
    confirmed = n_complete == len(PACK_LOADERS) and n_splits == len(PACK_LOADERS) and smoke_ok
    elapsed = round(time.perf_counter() - t0, 3)

    log_event(
        "info",
        "exp_101 open image ingest summary",
        exp_id=EXP_ID,
        profile=profile,
        n_packs_complete=n_complete,
        n_splits_ready=n_splits,
        smoke_ok=smoke_ok,
        hypothesis_confirmed=confirmed,
        elapsed_s=elapsed,
    )

    # Append JSONL synthesis record
    record = {
        "exp_id": EXP_ID,
        "model_name": f"{EXP_ID}_open_image_ingest",
        "started_at": datetime.now().isoformat(),
        "profile": profile,
        "elapsed_s": elapsed,
        "hypothesis_confirmed": confirmed,
        "eval_set": "synthesis",
        "n_packs_complete": n_complete,
        "n_splits_ready": n_splits,
        "smoke_ok": smoke_ok,
        "packs": summarize_packs(),
    }
    logs = ROOT / "logs" / "experiments.jsonl"
    logs.parent.mkdir(exist_ok=True)
    with logs.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return OpenImageIngestResult(
        n_packs_complete=n_complete,
        n_packs_expected=len(PACK_LOADERS),
        n_splits_ready=n_splits,
        smoke_ok=smoke_ok,
        pack_status=tuple(statuses),
        elapsed_s=elapsed,
        profile=profile,
        hypothesis_confirmed=confirmed,
    )


def gate_passed(result: OpenImageIngestResult) -> bool:
    return result.hypothesis_confirmed


def _build_results_md(result: OpenImageIngestResult) -> str:
    rows = "\n".join(
        f"| {s['pack']} | {s['complete']} | {s['stats_ok']} | {s['shape_ok']} |"
        for s in result.pack_status
    )
    verdict = "confirmed" if result.hypothesis_confirmed else "rejected"
    return "\n".join(
        [
            "# Results — EXP 101: Open image corpus ingest",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** CPU synthesis (verifies workstation downloads)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Packs complete: **{result.n_packs_complete}/{result.n_packs_expected}**",
            f"- Splits ready: **{result.n_splits_ready}/{result.n_packs_expected}**",
            f"- Smoke shapes: **{result.smoke_ok}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "| Pack | Complete | Stats | Shape OK |",
            "|------|----------|-------|----------|",
            rows,
            "",
            "## Verdict",
            f"**Hypothesis {verdict}** — Phase G P0 accept pack gate.",
            "",
            "## Limitations",
            "- Caption packs (G-T3) not included.",
            "- Raw blobs gitignored; checksums in download_stats.json.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 101 — open image corpus ingest")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_101(profile=args.profile, verbose=not args.quiet)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")
    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
