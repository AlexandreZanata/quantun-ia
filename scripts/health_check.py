#!/usr/bin/env python3
"""Pre-flight checks before large experiment runs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MIN_DISK_GB = 1.0


def _check_disk(path: Path) -> tuple[bool, str]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    ok = free_gb >= MIN_DISK_GB
    msg = f"disk free at {path}: {free_gb:.1f} GB (min {MIN_DISK_GB} GB)"
    return ok, msg


def _check_writable(path: Path) -> tuple[bool, str]:
    path.mkdir(parents=True, exist_ok=True)
    test_file = path / ".health_check"
    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        return True, f"writable: {path}"
    except OSError as exc:
        return False, f"not writable: {path} ({exc})"


def _check_mlflow() -> tuple[bool, str]:
    if os.environ.get("MLFLOW_DISABLE", "").lower() in {"1", "true", "yes"}:
        return True, "mlflow: disabled (MLFLOW_DISABLE)"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    if tracking_uri.startswith("file:"):
        path = Path(tracking_uri.removeprefix("file:"))
        path.mkdir(parents=True, exist_ok=True)
        return True, f"mlflow: local tracking at {path}"
    return True, f"mlflow: remote tracking at {tracking_uri}"


def _check_gpu() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "gpu: torch not installed"
    if not torch.cuda.is_available():
        return False, "gpu: CUDA not available (required for local real gate)"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    return True, f"gpu: {name} ({vram_gb:.1f} GB VRAM, CUDA {torch.version.cuda})"


def _check_dvc() -> tuple[bool, str]:
    from scripts.validate_dvc import validate_dvc

    ok, issues = validate_dvc()
    infos = [i for i in issues if "informational" in i]
    errors = [i for i in issues if "informational" not in i]
    if errors:
        return False, f"dvc: {errors[0]}"
    if infos:
        return True, f"dvc: pipeline OK ({infos[0].split(' — ')[0]})"
    return True, "dvc: pipeline OK"


def run_health_check(strict: bool = False, *, gpu: bool = False) -> int:
    checks: list[tuple[str, bool, str]] = []

    probe_list: list[tuple[str, Any]] = [
        ("disk", lambda: _check_disk(ROOT)),
        ("logs", lambda: _check_writable(ROOT / "logs")),
        ("artifacts", lambda: _check_writable(ROOT / "artifacts")),
        ("mlflow", _check_mlflow),
        ("dvc", _check_dvc),
    ]
    if gpu:
        probe_list.insert(0, ("gpu", _check_gpu))

    for name, fn in probe_list:
        ok, msg = fn()
        checks.append((name, ok, msg))
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {msg}")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(f"Health check failed: {', '.join(failed)}", file=sys.stderr)
        return 1 if strict else 0
    print("Health check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify environment before experiment runs")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 on any failure",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Verify NVIDIA CUDA GPU (local real gate)",
    )
    args = parser.parse_args()
    return run_health_check(strict=args.strict, gpu=args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
