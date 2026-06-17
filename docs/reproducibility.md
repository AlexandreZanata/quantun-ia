# Reproducibility Checklist

NeurIPS-style reproducibility statement for the quantun-ia research lab.
Use this document when submitting papers, workshop abstracts, or Zenodo releases.

---

## 1. Code availability

| Item | Status | Evidence |
|------|--------|----------|
| Public repository | ✅ | [github.com/AlexandreZanata/quantun-ia](https://github.com/AlexandreZanata/quantun-ia) |
| License | ✅ | MIT — `LICENSE` |
| Citation metadata | ✅ | `CITATION.cff` |
| Contribution guide | ✅ | `CONTRIBUTING.md` |

---

## 2. Dependencies and environment

| Item | Status | Evidence |
|------|--------|----------|
| Pinned runtime deps | ✅ | `requirements.lock` |
| Dev / test deps | ✅ | `requirements-dev.txt` |
| Python version | ✅ | `>=3.11` (CI uses 3.12) |
| Docker reproducibility | ✅ | `Dockerfile`, `docker-compose.test.yml` |
| GPU optional | ✅ | `docker-compose.gpu.yml`, `QML_DEVICE` env |

**Install (local):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

---

## 3. Random seeds and splits

| Item | Status | Evidence |
|------|--------|----------|
| Global seed utility | ✅ | `src/training/reproducibility.py` |
| Seeds in config | ✅ | `config/experiments.yaml` — `publication` profile: 10 seeds |
| Split before preprocessing | ✅ | `src/data/splits.py`, `src/data/scaling.py` |
| Seed logged per run | ✅ | `ExperimentLogger` writes `seed` to JSONL |

Default publication seeds: `42, 123, 456, 789, 1000, 2000, 3000, 4000, 4242, 5000`

---

## 4. Compute requirements

| Profile | n_samples | Seeds | Epochs (typical) | Hardware | Est. runtime |
|---------|-----------|-------|------------------|----------|--------------|
| `ci` | 50 | 1 | 5 | CPU | ~3 s (exp_001 smoke) |
| `publication` | 500 | 10 | 50 | CPU | ~5–15 min per experiment |
| `publication_large` | 1000 | 10 | 50 | CPU | ~15–45 min per experiment |
| Full replay | 1000 | 10 | 50 | CPU | ~2–8 h (all experiments) |

**Commands:**

```bash
make repro                         # CI profile smoke + golden bounds
make experiment-large              # publication_large for all experiments
make replay-publication-artifacts  # export CSV, figures, LaTeX from existing logs
make replay-publication            # publication_large runs + full export pipeline
MLFLOW_DISABLE=1 python experiments/exp_001_quantum_vs_classical/run.py
```

GPU: set `QML_DEVICE=cuda` or use `docker-compose.gpu.yml` (NVIDIA runtime required).

---

## 5. Experiment artifacts

| Artifact | Location | Versioned |
|----------|----------|-----------|
| Append-only logs | `logs/experiments.jsonl` | Gitignored (local); export via DVC |
| CSV export | `data/exports/results.csv` | DVC (`dvc.yaml`) |
| Model checkpoints | `artifacts/` | Gitignored |
| MLflow runs | `mlruns/` | Gitignored; optional |
| Publication figures | `figures/*.pdf` | Gitignored; generated via `make figures` |
| LaTeX tables | `paper/tables/*.tex` | Tracked (auto-generated) |

---

## 6. Statistical methodology

| Method | Module | Used for |
|--------|--------|----------|
| Stratified holdout | `holdout.py` | 30% test split before training |
| Bootstrap 95% CI | `statistics.py` | Multi-seed mean accuracy |
| Paired Wilcoxon | `statistics.py` | Per-seed paired comparisons |
| Holm-Bonferroni | `statistics.py` | Multiple comparison correction |
| Parameter matching | `param_match.py` | Fair quantum vs classical baselines |

---

## 7. Reproduction workflow

```bash
# 1. Verify environment
pytest tests/smoke/ -v

# 2. Fast reproducibility check (CI profile)
make repro

# 3. Full publication run (optional, long)
make experiment-large

# 4. Export artifacts
make export-results
make figures
make latex-tables
# Or in one step from existing logs:
make replay-publication-artifacts

# 5. Release bundle (for Zenodo)
make release
```

**Full publication replay** (long — runs `publication_large` on every experiment first):

```bash
make replay-publication
dvc push    # after configuring remote — see docs/dvc_remote.md
```

Expected outcome: `make repro` passes; metrics within golden bounds in `tests/regression/golden_ci.json`.

---

## 8. Known limitations

- Fresh clone has empty `logs/experiments.jsonl` — run experiments or `dvc pull` for dashboard data.
- exp_011–014 have `hypothesis.md` but `results.md` pending first publication runs.
- MLflow tracking is optional (`MLFLOW_DISABLE=1` in CI).
- Zenodo DOI: enable via [docs/zenodo.md](zenodo.md) after GitHub release `v0.9.1`.

---

## 9. Checklist for reviewers

- [ ] Clone repo and install from `requirements.lock`
- [ ] `make repro` passes
- [ ] `pytest tests/ --cov-fail-under=70` passes
- [ ] Re-run one experiment (e.g. exp_001) and compare holdout accuracy to `results.md`
- [ ] Verify split-then-scale in `src/data/scaling.py` (no test-set leakage)
- [ ] Confirm Holm correction applied in multi-comparison experiments
