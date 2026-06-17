#!/usr/bin/env python3
"""Bootstrap a local DVC filesystem remote using the project venv."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "localstore"
DEFAULT_STORAGE_NAME = "quantun-ia-dvc-storage"
_REMOTE_SECTION = re.compile(r"""^\['remote\s+"(?P<name>[^"]+)"'\]\s*$""")


def _read_dvc_config(path: Path) -> dict[str, dict[str, str]]:
    """Parse DVC config (non-standard INI with quoted remote sections)."""
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    if not path.is_file():
        return sections
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        remote_match = _REMOTE_SECTION.match(line)
        if remote_match:
            current = remote_match.group("name")
            sections.setdefault(current, {})
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if "=" in line and current:
            key, value = line.split("=", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def dvc_module_available() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "dvc", "--version"],
            capture_output=True,
            check=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def resolve_dvc_cmd() -> list[str]:
    if not dvc_module_available():
        raise RuntimeError(
            "DVC is not installed in this environment. "
            "Run: python -m pip install dvc  (or make dvc-setup)"
        )
    return [sys.executable, "-m", "dvc"]


def resolve_storage_path(storage_path: Path | None, *, root: Path = ROOT) -> Path:
    if storage_path is not None:
        return storage_path.expanduser().resolve()
    return (root.parent / DEFAULT_STORAGE_NAME).resolve()


def remote_configured(root: Path, remote_name: str) -> bool:
    sections = _read_dvc_config(root / ".dvc" / "config")
    return remote_name in sections and "url" in sections[remote_name]


def read_remote_url(root: Path, remote_name: str) -> str | None:
    sections = _read_dvc_config(root / ".dvc" / "config")
    return sections.get(remote_name, {}).get("url")


def setup_local_remote(
    *,
    storage_path: Path | None = None,
    remote_name: str = DEFAULT_REMOTE,
    root: Path = ROOT,
) -> dict[str, str]:
    """Create storage dir and configure default local DVC remote (idempotent)."""
    if not (root / ".dvc").is_dir():
        raise RuntimeError(f"missing {root / '.dvc'} — run dvc init in repo root first")

    storage = resolve_storage_path(storage_path, root=root)
    storage.mkdir(parents=True, exist_ok=True)

    if remote_configured(root, remote_name):
        return {
            "status": "already_configured",
            "remote": remote_name,
            "url": read_remote_url(root, remote_name) or str(storage),
            "storage": str(storage),
        }

    dvc_cmd = resolve_dvc_cmd()
    subprocess.run(
        [*dvc_cmd, "remote", "add", "-d", remote_name, str(storage)],
        cwd=root,
        check=True,
    )
    return {
        "status": "created",
        "remote": remote_name,
        "url": str(storage),
        "storage": str(storage),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure local DVC filesystem remote")
    parser.add_argument(
        "--storage",
        type=Path,
        default=None,
        help=f"Storage directory (default: ../{DEFAULT_STORAGE_NAME})",
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="DVC remote name")
    args = parser.parse_args()

    try:
        result = setup_local_remote(storage_path=args.storage, remote_name=args.remote)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: dvc command failed (exit {exc.returncode})", file=sys.stderr)
        return exc.returncode or 1

    print(f"DVC remote {result['remote']}: {result['status']} → {result['url']}")
    print("Next: make replay-publication-artifacts && make dvc-push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
