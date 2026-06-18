#!/usr/bin/env bash
# Weekly publication reproduction: real exp_024 CI run + paper-build from logs.
# Usage: ./scripts/repro_publication_ci.sh   or   make repro-publication-ci
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

export MLFLOW_DISABLE=1
mkdir -p logs

echo "== quantun-ia publication reproduction (CI path) =="
echo "Repository: $ROOT"
echo "Date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo ""
echo "== Step 1/4: exp_024 CI smoke (real JSONL) =="
$PYTHON -m pytest tests/integration/test_exp_024_smoke.py -v --tb=short

echo ""
echo "== Step 2/4: merge publication fixture summaries for paper pipeline =="
PUBLICATION_FIXTURE="$ROOT/tests/contracts/fixtures/publication_experiments.jsonl"
if [[ ! -f "$PUBLICATION_FIXTURE" ]]; then
  echo "Missing fixture: $PUBLICATION_FIXTURE" >&2
  exit 1
fi
cat "$PUBLICATION_FIXTURE" >> logs/experiments.jsonl

echo ""
echo "== Step 3/4: paper artifacts from logs =="
make paper-artifacts
if command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex found — building PDF"
  $PYTHON scripts/build_paper.py
else
  echo "pdflatex not found — skipping PDF (tables + figures exported)"
fi

echo ""
echo "== Step 4/4: release bundle (verify-only) =="
$PYTHON scripts/prepare_release.py --verify-only dist/release 2>/dev/null || {
  echo "Building release bundle..."
  $PYTHON scripts/prepare_release.py
  $PYTHON scripts/prepare_release.py --verify-only dist/release
}

echo ""
echo "SUCCESS: Publication reproduction CI path complete."
echo "Artifacts: logs/experiments.jsonl, paper/tables/, figures/, dist/release/"
