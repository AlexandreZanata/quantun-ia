# Negative Results

This document records **honest failures** from the quantun-ia benchmark suite.
International research labs treat negative results as evidence of rigor, not weakness.
Each entry links to the experiment `results.md` and the statistical test outcome.

---

## Summary table

| Experiment | Hypothesis (brief) | Outcome | Statistical verdict |
|------------|-------------------|---------|---------------------|
| exp_005 | Curriculum (margin batches) beats random ordering | **Rejected** | Wilcoxon p=0.125; margin worse (−4.9 pp) |
| exp_007 | Self-play hard-example mining improves holdout | **Rejected** | Holm p=1.000; zero gain at n=500 and n=1000 |
| exp_003 | Entanglement topology improves re-upload QNN | **Rejected** | No Holm-significant pairwise vs `none` |
| exp_009 | Entanglement helps basic QNN (no re-upload) | **Rejected** | All topologies cluster 56–61%; no Holm significance |
| exp_046 | nano_xl beats nano_l by ≥0.3 pp AUC on HIGGS 805K | **Rejected** | nano_xl − nano_l = −0.03 pp; plateau at ~1.14M params |
| exp_044 | LargeNanoMLP beats logistic by ≥0.5 pp on NIHR val | **Rejected** | AUC 0.831 passes; nano − logistic = −0.16 pp |
| exp_045 | PR-AUC ≥ 0.55 on temporal GoBug val | **Rejected** | PR-AUC ≈ 0.31 (chance); temporal holdout defeats tabular nano |
| exp_052 | Warm-start hybrid beats e2e by ≥0.5 pp AUC on HIGGS | **Rejected** | Δ = −0.42 pp (3 seeds); Wilcoxon p = 0.5 |
| exp_053 | Dynamic entanglement schedule beats fixed by ≥1.0 pp | **Rejected** | Δ = −0.78 pp vs `none` (3 seeds); Wilcoxon p = 0.75 |
| exp_055 | Depolarizing noise beats noiseless hybrid by ≥0.5 pp PR-AUC | **Rejected** | Δ = +0.50 pp on temporal test (inconclusive vs gate) |
| exp_056 | Re-upload depth curriculum wins ≥2/3 ladder rungs | **Rejected** | 1/3 wins (PCA-MNIST only); BC/HIGGS losses |

---

## exp_005 — Curriculum quantum (re-upload base)

**Profile:** circles, n=500, noise=0.2, 10 seeds

| Method | Mean holdout | 95% CI |
|--------|--------------|--------|
| curriculum_random | **60.2%** | [57.7%, 62.8%] |
| curriculum_margin_batches | 55.3% | [50.9%, 59.4%] |

**Finding:** Margin-based curriculum staging **hurts** generalization on noisy circles.
At `publication_large` (n=1000), the gap widens: random 63.0% vs margin 54.9%.

**Lesson:** Difficulty ordering must be validated per dataset; curriculum is not universally beneficial.

See: `experiments/exp_005_curriculum_quantum/results.md`

---

## exp_007 — Self-play quantum

**Profile:** circles, n=500, noise=0.2, 10 seeds

| Phase | Mean holdout | 95% CI |
|-------|--------------|--------|
| self_play_best | 58.5% | [56.7%, 60.4%] |
| self_play_base | 57.4% | [54.3%, 60.1%] |

**Finding:** Technique is *applicable* (base accuracy above gate) but provides **no significant gain**
after Holm-Bonferroni correction (p=1.000). At n=1000, best = base = 62.9%.

**Lesson:** Hard-example mining on small 2D synthetic tasks does not transfer gains to holdout.

See: `experiments/exp_007_self_play/results.md`, `docs/publication_large_summary.md`

---

## exp_003 — Entanglement effect (re-upload QNN)

**Profile:** circles, n=500, 4 qubits, 3 layers, 10 seeds

| Entanglement | Mean | 95% CI |
|--------------|------|--------|
| none | **65.4%** | [63.9%, 67.1%] |
| chain_half | 64.4% | [61.9%, 67.1%] |
| ring | 62.0% | [58.8%, 65.3%] |
| chain | 60.9% | [57.6%, 64.3%] |

**Finding:** `none` (no entanglement) **outperforms** all entangled topologies — opposite of the
original hypothesis. No pairwise comparison survives Holm correction.

**Lesson:** Entanglement cost may exceed benefit on low-dimensional synthetic data at this depth.

See: `experiments/exp_003_entanglement_effect/results.md`

---

## exp_009 — Entanglement basic ablation

**Profile:** basic QNN (no re-upload), 4 qubits, 2 layers, 10 seeds

| Entanglement | Mean | 95% CI |
|--------------|------|--------|
| chain_half | **61.3%** | [58.7%, 63.9%] |
| none | 60.5% | [57.5%, 63.3%] |
| chain | 57.8% | [54.0%, 61.7%] |
| ring | 56.4% | [53.9%, 59.1%] |

**Finding:** All topologies cluster near chance-adjusted performance with **no Holm-significant**
entanglement effect. The exp_003 reversal appears tied to re-upload expressivity, not entanglement alone.

See: `experiments/exp_009_entanglement_basic/results.md`

---

## exp_046 — Model scale curve on HIGGS

**Profile:** HIGGS 805K train / 172.5K val, 8 epochs, seed 42, RTX 4060

| Variant | Params | Val ROC-AUC | Peak VRAM (MB) |
|---------|--------|-------------|----------------|
| nano_s | 85K | 0.8270 | 69 |
| nano_m | 309K | 0.8313 | 104 |
| nano_l | 1.14M | **0.8316** | 181 |
| nano_xl | 4.45M | 0.8314 | 359 |
| nano_xxl | 9.03M | 0.8316 | 429 |

**Finding:** Validation AUC **plateaus at nano_l** (~1.14M params). nano_xl does not beat nano_l
by the pre-registered 0.3 pp gate (−0.03 pp). nano_xxl fits in 8 GB VRAM but adds no gain.

**Lesson:** On HIGGS at this epoch budget, wider MLPs waste VRAM/time without AUC benefit;
champion selection should stay at nano_l unless multi-seed overnight runs show otherwise.

See: `experiments/exp_046_model_scale_curve/results.md`

---

## exp_044 — NIHR synthetic CV baseline

**Profile:** NIHR 70K train / 15K val, 12 epochs, seed 42, RTX 4060

| Model | Val ROC-AUC |
|-------|-------------|
| Logistic (QRISK-style) | **0.8322** |
| LargeNanoMLP (~1.11M) | 0.8306 |

**Finding:** Primary AUC gate passes (0.831 ≥ 0.70), but nano does **not** beat logistic by
the pre-registered 0.5 pp advantage gate (−0.16 pp). On realistic-prevalence NIHR (~6.6%
positive), a simple logistic baseline is a strong ceiling for the current MLP architecture.

**Lesson:** Use NIHR for honest clinical metrics and calibration (exp_047); do not assume
million-param MLP beats logistic on low-dimensional epidemiological features without ablation.

See: `experiments/exp_044_nihr_cv_baseline/results.md`

---

## exp_045 — GoBug file-level defect baseline

**Profile:** GoBug 27k train / 5.8k temporal val, 12 epochs, seed 42, RTX 4060

| Model | Val PR-AUC |
|-------|------------|
| Logistic | 0.3097 |
| LargeNanoMLP (~82k) | 0.3097 |

**Finding:** Under **commit-order temporal split** (sha-sorted 70/15/15), both models score
PR-AUC ≈ **0.31** — at prevalence baseline (~27–32% on val). Pre-registered gate (≥ 0.55)
**rejected**. Tabular nano MLP does not generalize to later commits without drift-aware training.

**Lesson:** Phase 2 programming domain needs commit timestamps + proper temporal CV
(go-bug-collector methodology), not sha-lexicographic proxy alone. Consider exp_048 synthetic
code metrics for controlled ablations before re-attempting full GoBug scale.

See: `experiments/exp_045_code_defect_gobug/results.md`

---

## exp_052 — Quantum warm-start on HIGGS hybrid sandwich

**Profile:** HIGGS 50k train / 10k val, HybridSandwich 4q×2L re-upload, 3 seeds, RTX 4060

| Method | Mean val ROC-AUC |
|--------|------------------|
| end-to-end hybrid | **0.7541** |
| classical-first warm-start (70/30) | 0.7499 |

**Finding:** Warm-start underperforms end-to-end by **−0.42 pp** (gate ≥ +0.5 pp).
Paired wins: 1/3 · Wilcoxon p = 0.5.

**Lesson:** Freezing QNN while training pre/post does not transfer better than joint
optimization on million-row tabular HIGGS — schedule hypothesis rejected for HybridSandwich.

See: `experiments/exp_052_quantum_warmstart_higgs/results.md`

---

## exp_053 — Dynamic entanglement schedule on breast cancer

**Profile:** UCI Wisconsin breast cancer, re-upload QNN 4q×2L, 5 stages × 10 epochs, 3 seeds, RTX 4060

| Method | Mean holdout |
|--------|--------------|
| Best fixed (`none`) | **96.69%** |
| Dynamic schedule (none→chain→ring) | 95.91% |

**Finding:** Growing entanglement underperforms best fixed topology by **−0.78 pp** (gate ≥ +1.0 pp).
Paired wins: 1/3 · Wilcoxon p = 0.75.

**Lesson:** Mid-training topology swaps disrupt learned post-head geometry even when QNN
weights are preserved — fixed `none` remains best on this cohort.

See: `experiments/exp_053_entangle_schedule_bc/results.md`

---

## exp_055 — Depolarizing noise on GoBug hybrid QNN

**Profile:** GoBug file-level, HybridSandwich 4q×2L, p=0.03, full train, temporal test split, RTX 4060

| Model | Temporal test PR-AUC |
|-------|----------------------|
| Noiseless hybrid | 0.3231 |
| Noisy hybrid (p=0.03) | **0.3281** |

**Finding:** Noise improves PR-AUC by **+0.50 pp** but gate requires **≥ +0.5 pp** strictly — **inconclusive /
honest negative** (rounded advantage 0.498).

**Lesson:** Depolarizing regularization does not materially help temporal generalization on GoBug;
hybrid QNN remains near chance on sha-order test (~0.32 PR-AUC).

See: `experiments/exp_055_noise_reg_gobug/results.md`

---

## exp_056 — Re-upload depth curriculum ladder

**Profile:** 3 rungs (PCA-MNIST, breast cancer, HIGGS 50k), re-upload QNN 4q, layers 1→2→3, RTX 4060

| Rung | Curriculum | Fixed L=3 | Δ pp | Won |
|------|------------|-----------|------|-----|
| pca_mnist_binary | 87.33% | 80.67% | +6.67 | yes |
| breast_cancer | 62.57% | 95.32% | −32.75 | no |
| higgs_50k (AUC) | 0.7201 | 0.7274 | −0.73 | no |

**Finding:** Only **1/3** rungs pass gate (need ≥ 2). Depth growth hurts breast cancer badly when
fixed depth converges quickly.

**Lesson:** Re-upload depth curriculum may help low-dim PCA tasks but does not transfer to
clinical/tabular rungs — do not promote to MicroQML Bench v2 flagship.

See: `experiments/exp_056_reupload_curriculum_ladder/results.md`

---

## How we use negative results

1. **Applicability gates** (`src/training/protocol.py`) prevent running expensive techniques on tasks where baselines already fail.
2. **Holm-Bonferroni** correction is mandatory for all multi-comparison claims.
3. **Publication figures** (`make figures`) include all experiments — successes and failures alike.
4. Future work (Phase 4) should target regimes where these techniques might succeed (higher dimension, real data).
