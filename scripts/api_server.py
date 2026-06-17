#!/usr/bin/env python3
"""Run the quantun-ia REST API server."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="quantun-ia REST API server")
    parser.add_argument("--host", default=os.environ.get("API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "8000")))
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    uvicorn.run(
        "src.presentation.http.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
