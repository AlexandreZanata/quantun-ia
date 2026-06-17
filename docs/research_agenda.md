# Research Agenda — 12-Month Roadmap

**Lab:** Quantum-Inspired Micro ML Lab (`quantun-ia`)  
**Version:** v0.9.5  
**Last updated:** 2026-06-17  
**Primary narrative:** Holdout-fair comparison of hybrid quantum–classical classifiers on real and synthetic benchmarks (Option C — see `paper/sections/introduction.tex`).

This document is the **public** research roadmap. Every listed experiment has (or will have) a pre-written `hypothesis.md` before any `run.py` run, per lab workflow.

---

## Guiding questions

1. **Fairness:** When classical and quantum models are parameter-matched and evaluated on identical holdout splits, where does QML win, tie, or lose?
2. **Robustness:** Do hybrid topologies and encodings change behaviour under label noise (exp_017) or augmentation (exp_013)?
3. **Reproducibility:** Do simulator backend choices (`default.qubit` vs `lightning.qubit`) change scientific conclusions?
4. **Impact:** Can we ship a versioned **MicroQML Bench** others can cite without re-implementing our protocol?

---

## Q3 2026 (Jul–Sep) — Simulator fidelity and protocol hardening

| ID | Experiment | Falsifiable claim | Status |
|----|------------|-------------------|--------|
| **021** | [QML backend parity](../experiments/exp_021_qml_backend_parity/hypothesis.md) | `default.qubit` and `lightning.qubit` holdout accuracies within 2 pp on breast cancer QNN | **Active** |
| 022 | Encoding × backend interaction (planned) | Amplitude vs angle encoding parity holds across backends on PCA-MNIST | Planned |
| 023 | Gradient diagnostic parity (planned) | Epoch-1 gradient norms agree within 10% across backends for 4q/2l ansatz | Planned |

**Exit criteria:** exp_021 `results.md` with multi-seed verdict; backends logged in every JSONL line; OSF pre-registration filed before publication-profile runs.

---

## Q4 2026 (Oct–Dec) — Benchmark productization

| ID | Experiment / deliverable | Falsifiable claim | Status |
|----|--------------------------|-------------------|--------|
| 024 | MicroQML Bench contract v0.1 (planned) | External replicator runs `make repro` + exp_011/021 in &lt;1 h CPU | Planned |
| — | Leaderboard JSON schema | Contract tests in `tests/contracts/` pass on sample exports | Planned |
| — | `docs/compute_environment.md` | Publication numbers traceable to hardware profile | Planned |

**Exit criteria:** Versioned bench name, schema semver, and Zenodo data bundle for synthetic + UCI tasks.

---

## Q1 2027 (Jan–Mar) — Platform without compromising science

| Phase | Scope | Falsifiable claim |
|-------|--------|-------------------|
| 16 | JWT auth + async GPU job queue | ✅ Authenticated async job completes with same holdout accuracy as local `nano_train` |
| — | Tenant-scoped SQLite → PostgreSQL migration path | Every repository query includes `tenantId` in contract tests |

Platform work runs **in parallel** with the paper track; it must not change split order or preprocessing invariants.

---

## Q2 2027 (Apr–Jun) — External impact

| Deliverable | Target |
|-------------|--------|
| Paper submission | Primary narrative (exp_011, 012, 017 + negative results) |
| MicroQML Bench v1 | Public release with DOI (Phase 17) |
| Replication challenge | GitHub issue template + `CONTRIBUTING.md` reproduction section |

---

## Completed foundation (reference)

Experiments **001–020** established hypothesis-first workflow, holdout evaluation, Holm-corrected Wilcoxon tests, negative results culture, REST API smoke (exp_020), and uniform statistical reporting (exp_011–018, Phase 14).

Headline comparisons and literature caveats: [`docs/baselines.md`](baselines.md).  
Honest failures: [`docs/negative_results.md`](negative_results.md).  
Reproducibility checklist: [`docs/reproducibility.md`](reproducibility.md).

---

## Pre-registration policy

From **exp_021** onward, publication-profile runs require an OSF pre-registration (or equivalent) linked in `hypothesis.md` before execution. CI and `ci` profile runs are exempt — they validate wiring only.

---

## How to propose a new experiment

1. Copy `experiments/template/` → `experiments/exp_NNN_<name>/`
2. Complete `hypothesis.md` (expectation, mechanism, falsification, metrics)
3. Add entry to `config/experiments.yaml` with `ci` and `publication` profiles
4. Open a PR — CI blocks placeholder hypotheses
5. After runs, fill `results.md` via `make results-new` where applicable

See [`hypothesis-workflow.md`](hypothesis-workflow.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md).
