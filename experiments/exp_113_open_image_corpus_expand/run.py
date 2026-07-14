"""
EXP 113 — Cycle v4 open image corpus expansion (Phase L).

CPU ingest gate — no GPU training:
  MLFLOW_DISABLE=1 python experiments/exp_113_open_image_corpus_expand/run.py \\
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

from src.data.open_captions import is_coco_captions_ready
from src.data.open_images import (
    PACK_LOADERS_V4,
    is_pack_complete,
    load_image_pack_arrays,
)
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_113_open_image_corpus_expand"
EXP_ID = "exp_113"
ROOT = Path(__file__).resolve().parents[2]
IMAGES_ROOT = ROOT / "data" / "open" / "images"

EXPECTED_SHAPES = {
    "stl10": (96, 96, 3),
    "tiny_imagenet": (64, 64, 3),
}
P0_CLASS_PACKS = PACK_LOADERS_V4
MIN_P0 = 3


@dataclass(frozen=True)
class OpenImageExpandResult:
    n_packs_complete: int
    n_packs_expected: int
    n_splits_ready: int
    coco_ready: bool
    smoke_ok: bool
    pack_status: tuple[dict, ...]
    elapsed_s: float
    profile: str
    hypothesis_confirmed: bool


def _processed_stats(pack: str) -> Path:
    return IMAGES_ROOT / pack / "processed" / "v1" / "stats.json"


def _split_npz(pack: str) -> Path:
    return IMAGES_ROOT / pack / "processed" / "v1" / "split_indices.npz"


def gate_passed(result: OpenImageExpandResult) -> bool:
    return bool(result.hypothesis_confirmed)


def run_exp_113(*, profile: str = "ci", verbose: bool = True) -> OpenImageExpandResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    n_smoke = int(cfg.get("n_smoke_rows", 8))
    min_p0 = int(cfg.get("min_p0_packs", MIN_P0))
    init_correlation_id()
    t0 = time.perf_counter()

    statuses: list[dict] = []
    smoke_ok = True
    for pack in P0_CLASS_PACKS:
        complete = is_pack_complete(pack)
        stats_ok = _processed_stats(pack).is_file() and _split_npz(pack).is_file()
        shape_ok = False
        err = None
        if complete and stats_ok:
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

    coco_ready = is_coco_captions_ready()
    coco_stats = (IMAGES_ROOT / "coco_captions" / "processed" / "v1" / "stats.json").is_file()
    statuses.append(
        {
            "pack": "coco_captions",
            "complete": coco_ready,
            "stats_ok": coco_stats,
            "shape_ok": coco_ready,
            "error": None if coco_ready else "missing pairs.parquet or download marker",
        }
    )
    if not coco_ready:
        smoke_ok = False
    if verbose:
        print(f"coco_captions: ready={coco_ready} stats={coco_stats}")

    n_complete = sum(1 for s in statuses if s["complete"])
    n_splits = sum(1 for s in statuses if s["stats_ok"])
    expected = len(P0_CLASS_PACKS) + 1
    confirmed = n_complete >= min_p0 and n_complete == expected and smoke_ok
    elapsed = round(time.perf_counter() - t0, 3)

    log_event(
        "info",
        "exp_113 open image expand summary",
        exp_id=EXP_ID,
        profile=profile,
        n_packs_complete=n_complete,
        n_splits_ready=n_splits,
        smoke_ok=smoke_ok,
        hypothesis_confirmed=confirmed,
        elapsed_s=elapsed,
    )

    record = {
        "exp_id": EXP_ID,
        "model_name": f"{EXP_ID}_open_image_expand",
        "started_at": datetime.now().isoformat(),
        "profile": profile,
        "seed": 42,
        "elapsed_s": elapsed,
        "hypothesis_confirmed": confirmed,
        "eval_set": "ingest",
        "n_packs_complete": n_complete,
        "n_splits_ready": n_splits,
        "smoke_ok": smoke_ok,
    }
    logs_path = ROOT / "logs" / "experiments.jsonl"
    logs_path.parent.mkdir(exist_ok=True)
    with open(logs_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return OpenImageExpandResult(
        n_packs_complete=n_complete,
        n_packs_expected=expected,
        n_splits_ready=n_splits,
        coco_ready=coco_ready,
        smoke_ok=smoke_ok,
        pack_status=tuple(statuses),
        elapsed_s=elapsed,
        profile=profile,
        hypothesis_confirmed=confirmed,
    )


def write_results(result: OpenImageExpandResult, path: Path) -> None:
    verdict = "Confirmed" if result.hypothesis_confirmed else "Rejected"
    lines = [
        f"# Results — EXP 113: Open image corpus expansion ({date.today().isoformat()})",
        "",
        f"**Verdict:** {verdict}",
        f"**Profile:** `{result.profile}`",
        f"**Packs complete:** {result.n_packs_complete}/{result.n_packs_expected}",
        f"**Splits ready:** {result.n_splits_ready}",
        f"**COCO ready:** {result.coco_ready}",
        f"**Smoke OK:** {result.smoke_ok}",
        f"**Elapsed (s):** {result.elapsed_s}",
        "",
        "## Per-pack",
        "",
        "| Pack | Complete | Splits | Smoke shape |",
        "|------|----------|--------|-------------|",
    ]
    for row in result.pack_status:
        lines.append(
            f"| {row['pack']} | {row['complete']} | {row['stats_ok']} | {row['shape_ok']} |"
        )
    lines.extend(
        [
            "",
            "## Gate (Phase L)",
            "",
            "- Win: ≥3 P0 packs (STL-10, Tiny-ImageNet, COCO micro) ready + smoke.",
            f"- Outcome: **{verdict}**.",
            "",
            "## Ablation suggestion",
            "",
            "- What if you add AFHQ 64×64 as a fourth I2I domain pack?",
            "",
            f"*Logged via ExperimentLogger · {datetime.now().isoformat()}*",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ci", "publication"), default="ci")
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args(argv)
    result = run_exp_113(profile=args.profile)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        write_results(result, out)
        print(f"wrote {out}", flush=True)
    return 0 if gate_passed(result) or args.profile == "ci" else 1


if __name__ == "__main__":
    raise SystemExit(main())
