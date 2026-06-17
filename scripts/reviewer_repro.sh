#!/usr/bin/env bash
# One-click reproduction script for artifact evaluators and replication challengers.
# Usage: ./scripts/reviewer_repro.sh   or   make reviewer-repro
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== quantun-ia reviewer reproduction (fast path) =="
echo "Repository: $ROOT"
echo "Date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
  echo "Using venv: .venv"
else
  PYTHON="${PYTHON:-python3}"
  echo "Using system Python: $($PYTHON --version 2>&1)"
fi

if ! $PYTHON -c "import torch" 2>/dev/null; then
  echo "Installing PyTorch (CPU) and dev dependencies..."
  $PYTHON -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  $PYTHON -m pip install -q -r requirements-dev.txt
  $PYTHON -m pip install -q -e .
fi

export MLFLOW_DISABLE=1

echo ""
echo "== Step 1/3: smoke imports =="
$PYTHON -m pytest tests/smoke/ -q

echo ""
echo "== Step 2/3: make repro (golden CI bounds) =="
make repro

echo ""
echo "== Step 3/3: integration + contract smoke =="
$PYTHON -m pytest tests/integration/ tests/contracts/ -q --tb=short

echo ""
echo "SUCCESS: Reviewer reproduction path complete."
echo "For full publication replay (long): make replay-publication"
echo "For paper artifacts: make paper-build && make arxiv-bundle"
echo "Report replication: open a GitHub issue using 'Experiment Replication Challenge' template."
