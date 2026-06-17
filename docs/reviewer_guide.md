# Reviewer & Artifact Evaluation Guide

This guide supports **ACM artifact evaluation**, **NeurIPS reproducibility review**, and independent
replication challengers. Target software version: **v0.9.14** (Phase 23).

---

## Fast path (< 15 minutes, CPU)

From a clean clone:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e .

make reviewer-repro
```

`make reviewer-repro` runs:

1. Smoke import tests
2. `make repro` — exp_001 CI profile with golden bounds (`tests/regression/golden_ci.json`)
3. Integration + contract tests (holdout protocol, citation metadata, results.md uniformity)

**Expected outcome:** all steps exit 0.

---

## Medium path (single experiment, ~5–30 min)

Re-run one headline experiment and compare to `experiments/exp_NNN_*/results.md`:

```bash
export MLFLOW_DISABLE=1
python experiments/exp_001_quantum_vs_classical/run.py
python experiments/exp_021_qml_backend_parity/run.py   # publication — long
```

Verify holdout accuracy and seed count match the published `results.md` within bootstrap CI overlap.

---

## Full publication replay (2–8 hours, CPU)

```bash
make replay-publication          # publication_large on all experiments + export
make replay-publication-artifacts  # export only from existing logs
```

Compare exported CSV and figures to the Zenodo release bundle (`make release`).

---

## Statistical invariants to verify

| Invariant | Where enforced |
|-----------|----------------|
| Split before scale | `src/data/scaling.py`, `tests/unit/test_splits.py` |
| Multi-seed + bootstrap CI | `src/training/statistics.py` |
| Holm-Bonferroni correction | `src/training/statistics.py`, `tests/unit/test_holm.py` |
| Parameter-matched baselines | `src/training/param_match.py` |
| Uniform results.md | `tests/contracts/test_results_md_uniform.py` |

---

## Author-run statement

All metrics in the primary paper draft were produced by the corresponding author using the
`publication` profile in `config/experiments.yaml` (10 seeds, stratified 70/30 holdout).
Compute environment is documented in [compute_environment.md](compute_environment.md).
Timestamps and seeds are logged append-only to `logs/experiments.jsonl` via `ExperimentLogger`.

---

## Reporting replication

Open a GitHub issue using the **Experiment Replication Challenge** template
(`.github/ISSUE_TEMPLATE/experiment_replication.yml`). Include:

- Experiment ID and profile
- Commit SHA or release tag
- Your holdout metrics vs published `results.md`
- Verdict: match / partial / mismatch

---

## Related docs

- [reproducibility.md](reproducibility.md) — NeurIPS-style checklist
- [zenodo.md](zenodo.md) — release bundle and DOI
- [arxiv.md](arxiv.md) — paper submission bundle
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contribution and replication policy
