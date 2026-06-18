"""Champion/challenger checkpoint promotion for continuous training."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.training import checkpoints

MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_LINK_NAME = "checkpoint"
DEFAULT_PROMOTE_MAX_DELTA_PP = 0.5
DEFAULT_ROLLBACK_REGRESSION_PP = 1.0


def champion_dir() -> Path:
    return checkpoints.ARTIFACTS_ROOT / "champion"


@dataclass(frozen=True)
class ChampionManifest:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    holdout_accuracy: float
    checkpoint_path: str
    promoted_at: str

    @classmethod
    def from_dict(cls, payload: dict) -> ChampionManifest:
        return cls(
            exp_id=str(payload["exp_id"]),
            model_name=str(payload["model_name"]),
            dataset=str(payload["dataset"]),
            seed=int(payload["seed"]),
            holdout_accuracy=float(payload["holdout_accuracy"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            promoted_at=str(payload["promoted_at"]),
        )


def holdout_delta_pp(challenger_accuracy: float, champion_accuracy: float) -> float:
    """Absolute holdout accuracy gap in percentage points."""
    return abs(challenger_accuracy - champion_accuracy) * 100.0


def should_promote(
    challenger_accuracy: float,
    champion_accuracy: float,
    *,
    max_delta_pp: float = DEFAULT_PROMOTE_MAX_DELTA_PP,
) -> bool:
    """Promote challenger when holdout is within tolerance of champion (not worse by > max_delta_pp)."""
    regression_pp = (champion_accuracy - challenger_accuracy) * 100.0
    return regression_pp <= max_delta_pp


def should_rollback(
    challenger_accuracy: float,
    champion_accuracy: float,
    *,
    max_regression_pp: float = DEFAULT_ROLLBACK_REGRESSION_PP,
) -> bool:
    """Block promotion when challenger regresses more than max_regression_pp vs champion."""
    regression_pp = (champion_accuracy - challenger_accuracy) * 100.0
    return regression_pp > max_regression_pp


def read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_holdout_record(record: dict) -> bool:
    if record.get("test_accuracy") is None:
        return False
    return record.get("eval_set", "holdout_test") == "holdout_test"


def latest_holdout_record(
    records: list[dict],
    exp_id: str,
    model_name: str,
) -> dict | None:
    """Return the newest holdout record for exp_id + model_name."""
    latest: dict | None = None
    for record in records:
        if record.get("exp_id") != exp_id:
            continue
        if record.get("model_name") != model_name:
            continue
        if not _is_holdout_record(record):
            continue
        if latest is None or (record.get("started_at") or "") >= (latest.get("started_at") or ""):
            latest = record
    return latest


def load_champion_manifest(directory: Path | None = None) -> ChampionManifest | None:
    path = (directory or champion_dir()) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ChampionManifest.from_dict(payload)


def save_champion_manifest(manifest: ChampionManifest, directory: Path | None = None) -> Path:
    root = directory or champion_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_FILENAME
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return path


def _checkpoint_model_name(model_name: str, dataset: str) -> str:
    return f"{model_name}_{dataset}"


def promote_champion(manifest: ChampionManifest, directory: Path | None = None) -> Path:
    """Symlink checkpoint into artifacts/champion/ and persist manifest."""
    root = directory or champion_dir()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(manifest.checkpoint_path)
    if not source.is_dir():
        source = checkpoints.resolve_checkpoint_dir(
            manifest.exp_id,
            manifest.model_name,
            manifest.dataset,
            seed=manifest.seed,
        )
    if not source.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {source}")

    link = root / CHECKPOINT_LINK_NAME
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()

    link.symlink_to(source.resolve(), target_is_directory=True)
    save_champion_manifest(manifest, directory=root)
    return root


def manifest_from_training(
    *,
    exp_id: str,
    model_name: str,
    dataset: str,
    seed: int,
    holdout_accuracy: float,
    checkpoint_path: str | None,
) -> ChampionManifest:
    ckpt = checkpoint_path or str(
        checkpoints.resolve_checkpoint_dir(exp_id, model_name, dataset, seed=seed)
    )
    return ChampionManifest(
        exp_id=exp_id,
        model_name=model_name,
        dataset=dataset,
        seed=seed,
        holdout_accuracy=holdout_accuracy,
        checkpoint_path=ckpt,
        promoted_at=datetime.now(timezone.utc).isoformat(),
    )
