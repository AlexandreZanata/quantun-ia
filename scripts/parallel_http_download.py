#!/usr/bin/env python3
"""Parallel HTTP range download helper for large open-data archives."""
from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from urllib.request import Request, urlopen


def fetch_range(url: str, part: Path, start: int, end: int) -> Path:
    expected = end - start + 1
    if part.is_file() and part.stat().st_size == expected:
        print(f"cached {part.name}", flush=True)
        return part
    headers = {"Range": f"bytes={start}-{end}"}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=180) as resp, part.open("wb") as handle:  # noqa: S310
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            handle.write(chunk)
    print(f"done {part.name} ({expected} bytes)", flush=True)
    return part


def download(url: str, out: Path, *, n_parts: int = 8) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    head = Request(url, method="HEAD")
    with urlopen(head, timeout=30) as resp:  # noqa: S310
        size = int(resp.headers["Content-Length"])
        accept_ranges = resp.headers.get("Accept-Ranges", "").lower()
    print(f"size={size} Accept-Ranges={accept_ranges}", flush=True)
    if accept_ranges != "bytes":
        print("server lacks range support — falling back to single stream", flush=True)
        req = Request(url)
        with urlopen(req, timeout=600) as resp, out.open("wb") as handle:  # noqa: S310
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                handle.write(chunk)
        return out

    chunk = size // n_parts
    jobs: list[tuple[str, Path, int, int]] = []
    for i in range(n_parts):
        start = i * chunk
        end = size - 1 if i == n_parts - 1 else start + chunk - 1
        jobs.append((url, out.parent / f".{out.name}.part{i}", start, end))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parts) as pool:
        parts = list(pool.map(lambda j: fetch_range(*j), jobs))

    with out.open("wb") as handle:
        for part in parts:
            handle.write(part.read_bytes())
            part.unlink(missing_ok=True)
    if out.stat().st_size != size:
        msg = f"size mismatch: {out.stat().st_size} != {size}"
        raise RuntimeError(msg)
    print(f"wrote {out} ({size} bytes)", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=8)
    args = parser.parse_args()
    download(args.url, args.out, n_parts=args.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
