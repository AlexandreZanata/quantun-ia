# Literature Review — Phase 4 Innovation Context

This document supports the Phase 4 research contribution: **gradient-variance adaptive
learning rate for variational quantum classifiers** (`exp_015_adaptive_qnn`).

---

## 1. Barren plateaus in variational quantum circuits

| Work | Contribution | Relevance to quantun-ia |
|------|--------------|-------------------------|
| McClean et al. (2018) | Barren plateaus: gradients vanish exponentially in depth | Motivates exp_006 gradient diagnostics |
| Grant et al. (2019) | Plateau structure depends on initialization and entanglement | Informs 6q×3l plateau-prone architecture choice |
| Cincio et al. (2022) | Cost-function-dependent plateaus in shallow QNNs | Cited in `docs/baselines.md`; supports adaptive training need |

**Our measurement:** exp_006 quantifies gradient variance decay from 2→10 qubits with bootstrap CIs.
exp_015 uses the 4q-scale variance (~0.015) as `var_target` for LR scaling.

---

## 2. Adaptive and layer-wise learning rates in QML

| Work | Method | Gap we address |
|------|--------|----------------|
| Stokes et al. (2020) | Quantum Natural Gradient | Heavy metric tensor; we use lightweight variance proxy |
| Ostaszewski et al. (2021) | Adaptive shot allocation | Focuses on measurement budget, not LR |
| Zhang et al. (2022) | Learning rate schedules for VQCs | Schedules are time-based, not gradient-variance-based |

**Novelty claim:** Per-step gradient variance feedback (from same backward pass) to scale Adam LR,
without reducing qubits or layers — tested with multi-seed holdout + Cohen's d.

---

## 3. Classical baselines and fair comparison

| Work | Baseline type | Our implementation |
|------|---------------|-------------------|
| Schuld & Petruccione (2018) | Standard VQC | `QuantumNetBasic` — exp_001, exp_015 |
| Farhi & Neven (2018) | Hybrid QNN | exp_002 (not primary for exp_015) |
| Bouthillier et al. (2021) | ML evaluation rigor | 10-seed holdout + Holm-Bonferroni |

Parameter-matched classical MLP (`build_param_matched_classical`) ensures fair capacity comparison.

---

## 4. Negative results informing Phase 4

Prior experiments reduce risk of over-claiming:

| Experiment | Finding | Implication for exp_015 |
|------------|---------|-------------------------|
| exp_003/009 | No entanglement benefit on circles | Architecture tuning alone is insufficient |
| exp_005 | Curriculum hurts on circles | Training dynamics matter — adaptive LR is complementary |
| exp_006 | Confirmed vanishing gradients at 6q+ | Justifies variance-targeted LR intervention |

See `docs/negative_results.md`.

---

## 5. Alternative Phase 4 directions (deferred)

| Option | Experiment | Status |
|--------|------------|--------|
| B — Hybrid NAS | exp_016_hybrid_nas | **Complete** — Optuna search vs EXP 002 baselines (Phase 6) |
| C — Poisoning × topology | exp_017_poison_topology | **Complete** — hybrid layouts under label poison (Phase 7) |
| D — Feature fusion | exp_018_feature_fusion | **Complete** — Transformer → QNN on phase sequences (Phase 8) |

**Selected direction:** Option A (`exp_015`) — highest leverage from existing exp_006 infrastructure.

---

## 6. Pre-registration note

Optional Open Science Framework timestamp before first `run.py` execution.
Hypothesis and ablation plan are frozen in `experiments/exp_015_adaptive_qnn/hypothesis.md`
before results collection, satisfying pre-registration intent.

---

## References

- McClean, J. R., et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9, 4812.
- Grant, E., et al. (2019). An initialization strategy for addressing barren plateaus in parametrized quantum circuits. *Quantum*, 3, 214.
- Cincio, C., et al. (2022). Cost function dependent barren plateaus in shallow parametrized quantum circuits. *Nature Communications*, 13, 1798.
- Stokes, J., et al. (2020). Quantum Natural Gradient. *Quantum*, 4, 269.
- Ostaszewski, M., et al. (2021). Structure optimization for parameterized quantum circuits. *Quantum*, 5, 391.
- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *MLSys*.
- Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*. Springer.
