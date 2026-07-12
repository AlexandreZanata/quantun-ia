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
| exp_074 | Dynamic entanglement schedule beats fixed by ≥0.5 pp PR-AUC (NIHR) | **Rejected** | Δ = −3.66 pp vs `ring` (3 seeds); Wilcoxon p = 0.25 |
| exp_064 | Dynamic entanglement schedule beats fixed by ≥0.5 pp ROC-AUC (ACYD) | **Rejected** | Δ = −1.08 pp vs `none` (3 seeds); Wilcoxon p = 0.75 |
| exp_055 | Depolarizing noise beats noiseless hybrid by ≥0.5 pp PR-AUC | **Rejected** | Δ = +0.50 pp on temporal test (inconclusive vs gate) |
| exp_056 | Re-upload depth curriculum wins ≥2/3 ladder rungs | **Rejected** | 1/3 wins (PCA-MNIST only); BC/HIGGS losses |
| exp_057 | Parameter-shift within 1 pp + ≥2× lower grad variance | **Rejected** | Δ holdout = 20.99 pp; var ratio = 0.08 (param-shift higher) |
| exp_058 | LargeNanoMLP ≥ best conventional + 0.5 pp (HIGGS) | **Rejected** | sklearn MLP 0.8429 vs nano 0.8358 (−0.71 pp full val) |
| exp_061 | LargeNanoMLP ≥ best conventional + 0.5 pp (ACYD) | **Rejected** | HistGB 0.6941 vs nano 0.6777 (−1.64 pp temporal val) |
| exp_069 | LargeNanoMLP ≥ logistic + 1.0 pp PR-AUC (NIHR) | **Rejected** | logistic 0.2382 vs nano 0.2370 (−0.12 pp) |
| exp_076 | LargeNanoMLP ≥ best conventional + 0.5 pp PR-AUC (NIHR) | **Rejected** | logistic 0.2382 vs nano 0.2393 (+0.12 pp) |
| exp_077 | LargeNanoMLP ≥ best conventional + 0.5 pp PR-AUC (GoBug) | **Rejected** | HistGB 0.3276 vs nano 0.3174 (−1.02 pp) |
| exp_070 | LargeNanoMLP ≥ logistic + 2.0 pp PR-AUC (GoBug) | **Rejected** | logistic 0.3097 vs nano 0.3100 (+0.03 pp) |
| exp_080 | Fused ACYD champion ≥ best hybrid + 0.5 pp (and C4 −1.0 pp) | **Rejected** | champion 0.6709 vs hybrid 0.6771 (−0.62 pp); parity vs C4 OK (−0.68 pp) |
| exp_079 | HIGGS→ACYD QNN head transfer beats scratch by ≥ +0.5 pp | **Rejected** | transfer +0.35 pp (below gate); honest-negative hypothesis confirmed |

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

## exp_057 — Parameter-shift vs autograd on deep re-upload QNN

**Profile:** breast cancer, 4q × 3L re-upload, 10 seeds, RTX 4060

| Method | Mean holdout acc | Grad variance |
|--------|------------------|---------------|
| autograd (backprop) | 88.07% | 0.000973 |
| parameter-shift | 67.08% | 0.012134 |

**Finding:** Parameter-shift **underperforms** autograd by **20.99 pp** on holdout and shows **higher**
(not lower) gradient variance (ratio 0.08 vs gate ≥ 2.0). Single-sample SGD required by PennyLane #4462
adds ~19 min/seed wall time with no accuracy benefit.

**Lesson:** Keep backprop for production re-upload training; reserve parameter-shift for diagnostics only
(exp_006). Do not set `QML_GRADIENT_METHOD=parameter_shift` in `nanotrainer.yaml`.

See: `experiments/exp_057_param_shift_ablation/results.md`

---

## exp_058 — Conventional HIGGS baselines vs LargeNanoMLP

**Profile:** HIGGS v1, 805K train / 172.5K val, seed 42, RTX 4060

| Model | Val ROC-AUC | Train (s) |
|-------|-------------|-----------|
| MLPClassifier (sklearn, 2048-512-64) | **0.8429** | 679 |
| LargeNanoMLP (shipped exp_032) | 0.8358 | 0.2 |
| HistGradientBoosting | 0.8097 | 3.3 |
| XGBoost shallow | 0.7773 | 2.2 |
| LogisticRegression | 0.6849 | 0.8 |

**Finding:** Matched-topology sklearn MLP **beats** the shipped LargeNanoMLP checkpoint by **0.71 pp**
on full val (gate requires nano ≥ best conventional + 0.5 pp). LargeNanoMLP still beats logistic
(+15.1 pp), HistGradientBoosting (+2.6 pp), and XGBoost (+5.8 pp).

**Lesson:** Do not claim universal superiority over all conventional stacks on million-row tabular;
retrain PyTorch nano or adopt sklearn MLP when matched-topology CPU training wins. CI slice (50K rows)
favored LargeNanoMLP (+4.14 pp) — slice size matters for headline claims.

See: `experiments/exp_058_conventional_higgs_baselines/results.md`

---

## exp_061 — Conventional ACYD baselines vs LargeNanoMLP

**Profile:** acyd_soy_brazil_v1, 50,107 train / 5,830 val, seed 42, RTX 4060

| Model | Val ROC-AUC | Train (s) |
|-------|-------------|-----------|
| HistGradientBoosting (sklearn) | **0.6941** | 0.4 |
| XGBoost shallow | 0.6882 | 0.7 |
| LargeNanoMLP (exp_060 checkpoint) | 0.6777 | 0.1 |
| MLPClassifier (sklearn, 2048-512-64) | 0.6736 | 29.6 |
| LogisticRegression | 0.6391 | 2.9 |

**Gate:** LargeNanoMLP ≥ best conventional + 0.5 pp → **failed** (−1.64 pp).

**Interpretation:** On ACYD temporal val, gradient boosting beats the deep nano anchor despite exp_060 beating logistic by +3.86 pp. Honest negative for “nano beats all classical” claims on agro-climate tabular; C4 anchor still valid vs logistic gate.

See: `experiments/exp_061_conventional_acyd_baselines/results.md`

---

## exp_083 — Conventional ACYD maize baselines vs LargeNanoMLP (C4b)

**Profile:** acyd_maize_brazil_v1, 151,956 train / 13,566 val, seed 42, RTX 4060

| Model | Val ROC-AUC | Train (s) |
|-------|-------------|-----------|
| HistGradientBoosting (sklearn) | **0.8178** | 1.3 |
| LargeNanoMLP (exp_081 checkpoint) | 0.8086 | 0.1 |
| MLPClassifier (sklearn, 2048-512-64) | 0.8018 | 158.4 |
| XGBoost shallow | 0.7706 | 1.2 |
| LogisticRegression | 0.6983 | 4.5 |

**Gate:** LargeNanoMLP ≥ best conventional + 0.5 pp → **failed** (−0.92 pp).

**Interpretation:** Same agro pattern as exp_061 soybean — HistGB beats C4b nano on temporal val while nano still clears logistic by +11.03 pp (exp_081). Honest negative for “nano beats all classical” on maize; C4b anchor remains valid vs logistic.

See: `experiments/exp_083_conventional_acyd_maize_baselines/results.md`

---

## exp_069 — LargeNanoMLP on NIHR synthetic CV (C2 anchor)

**Profile:** nihr_cv_synthetic_v1, 70K train / 15K val, seed 42, RTX 4060

| Model | Val PR-AUC |
|-------|------------|
| Logistic (QRISK-style) | **0.2382** |
| LargeNanoMLP | 0.2370 |

**Gate:** LargeNanoMLP ≥ logistic + 1.0 pp PR-AUC → **failed** (−0.12 pp).

**Interpretation:** At ~8% prevalence, logistic remains a strong ceiling on PR-AUC; deep nano does not beat it on this synthetic cohort. C2 checkpoint still ships for hybrid/QNN ablations (exp_051+).

See: `experiments/exp_069_large_nano_nihr/results.md`

---

## exp_076 — Conventional NIHR baselines vs LargeNanoMLP

**Profile:** nihr_cv_synthetic_v1, 70K train / 15K val, seed 42, RTX 4060

| Model | Val PR-AUC | Train (s) |
|-------|------------|-----------|
| LargeNanoMLP (exp_069 checkpoint) | **0.2393** | 0.1 |
| LogisticRegression (sklearn) | 0.2382 | 0.2 |
| XGBoost shallow | 0.2344 | 0.7 |
| MLPClassifier (sklearn, 2048-512-64) | 0.2327 | 38.0 |
| HistGradientBoosting | 0.2304 | 0.2 |

**Gate:** LargeNanoMLP ≥ best conventional + 0.5 pp PR-AUC → **failed** (+0.12 pp vs logistic).

**Interpretation:** Nano barely edges logistic on PR-AUC but does not clear the 0.5 pp conventional sweep gate; sklearn matched MLP underperforms. C2 anchor remains the comparison floor for quantum heads.

See: `experiments/exp_076_conventional_nihr_baselines/results.md`

---

## exp_077 — Conventional GoBug baselines vs LargeNanoMLP

**Profile:** code_defects_gobug_v1, 27,172 train / 5,822 val, seed 42, RTX 4060

| Model | Val PR-AUC | Train (s) |
|-------|------------|-----------|
| HistGradientBoosting (sklearn) | **0.3276** | 0.2 |
| XGBoost shallow | 0.3192 | 0.7 |
| LargeNanoMLP (exp_070 checkpoint) | 0.3174 | 0.1 |
| LogisticRegression | 0.3097 | 0.8 |
| MLPClassifier (sklearn, 2048-512-64) | 0.3039 | 19.8 |

**Gate:** LargeNanoMLP ≥ best conventional + 0.5 pp PR-AUC → **failed** (−1.02 pp vs HistGB).

**Interpretation:** Gradient boosting beats the deep nano anchor on GoBug temporal val despite exp_070 tying logistic. Honest negative for C3 vs full classical panel; nano still valid as hybrid ablation floor.

See: `experiments/exp_077_conventional_gobug_baselines/results.md`

---

## exp_070 — LargeNanoMLP on GoBug code defects (C3 anchor)

**Profile:** code_defects_gobug_v1, 27,172 train / 5,822 val, seed 42, RTX 4060

| Model | Val PR-AUC |
|-------|------------|
| Logistic | 0.3097 |
| LargeNanoMLP (2048-512-64, ~1.13M) | 0.3100 |

**Gate:** LargeNanoMLP ≥ logistic + 2.0 pp → **failed** (+0.03 pp).

**Interpretation:** Full nano template does not beat logistic on GoBug temporal val; matches exp_045 tie with reduced topology. C3 checkpoint shipped for hybrid ablations (exp_071+).

See: `experiments/exp_070_large_nano_gobug/results.md`

---

## exp_068a — Seasonal angle encoding on ACYD (H-Q8)

**Profile:** acyd_soy_brazil_v1, 50,107 train / 5,830 val, seed 42, RTX 4060

| Head | Val ROC-AUC |
|------|-------------|
| Classical (frozen C4) | 0.6777 |
| Seasonal angle QNN | 0.4979 |
| Seasonal amplitude QNN | 0.5137 |

**Gate:** angle ≥ classical + 0.5 pp and angle ≥ amplitude + 0.5 pp → **failed** (−17.98 pp vs classical).

**Interpretation:** Cyclic sin/cos features alone cannot substitute the frozen nano backbone; seasonal QNN heads collapse toward chance. Angle encoding does not beat amplitude on this 4-feature seasonal slice.

See: `experiments/exp_068a_angle_encoding_acyd/results.md`

---

## exp_072 — Quantum warm-start on NIHR (C2 replication)

**Profile:** nihr_cv_synthetic_v1, 50,000 train / 15,000 val, seeds 42/123/456, RTX 4060

| Condition | Mean val PR-AUC |
|-----------|-----------------|
| End-to-end hybrid | 0.2343 |
| Warm-start hybrid | 0.2307 |

**Gate:** warm-start ≥ e2e + 0.5 pp → **failed** (−0.35 pp; Wilcoxon p=0.5).

**Interpretation:** Classical-first schedule does not help NIHR HybridSandwich — confirms exp_052 honest negative cross-domain.

See: `experiments/exp_072_quantum_warmstart_nihr/results.md`

---

## exp_073 — Quantum warm-start on GoBug (C3 replication)

**Profile:** code_defects_gobug_v1, 27,172 train / 5,822 val, seeds 42/123/456, RTX 4060

| Condition | Mean val PR-AUC |
|-----------|-----------------|
| End-to-end hybrid | 0.3032 |
| Warm-start hybrid | 0.3067 |

**Gate:** warm-start ≥ e2e + 0.5 pp → **failed** (+0.35 pp; Wilcoxon p=0.75).

**Interpretation:** Warm-start shows a small positive delta on GoBug but does not meet the publication gate; H-Q2 remains a cross-domain honest negative.

See: `experiments/exp_073_quantum_warmstart_gobug/results.md`

---

## exp_074 — Dynamic entanglement schedule on NIHR (C2 replication)

**Profile:** nihr_cv_synthetic_v1, 10,000 train / 3,000 val, seeds 42/123/456, RTX 4060

| Condition | Mean val PR-AUC |
|-----------|-----------------|
| Dynamic schedule | 0.1963 |
| Best fixed (ring) | 0.2329 |

**Gate:** schedule ≥ best fixed + 0.5 pp → **failed** (−3.66 pp; Wilcoxon p=0.25).

**Interpretation:** Growing entanglement underperforms fixed ring on NIHR — confirms exp_053 honest negative cross-domain.

See: `experiments/exp_074_entangle_schedule_nihr/results.md`

---

## exp_064 — Dynamic entanglement schedule on ACYD (C4 / H-Q3)

**Profile:** acyd_soy_brazil_v1, 10,000 train / 3,000 val, seeds 42/123/456, RTX 4060

| Condition | Mean val ROC-AUC |
|-----------|------------------|
| Dynamic schedule | 0.6314 |
| Best fixed (none) | 0.6422 |

**Gate:** schedule ≥ best fixed + 0.5 pp → **failed** (−1.08 pp; Wilcoxon p=0.75).

**Interpretation:** Growing entanglement underperforms fixed `none` on ACYD — confirms exp_053/exp_074 honest negative on agro-climate tabular.

See: `experiments/exp_064_entangle_schedule_acyd/results.md`

---

## exp_068b — Compound stress label on ACYD (H-Q12)

**Profile:** acyd_soy_brazil_v1, compound label (low yield ∧ drought/heat), 50,107 train / 5,830 val, RTX 4060

| Condition | Val ROC-AUC |
|-----------|-------------|
| Logistic regression | 0.8462 |
| Hybrid QNN head (frozen C4) | 0.8074 |

**Gate:** hybrid ≥ logistic + 1.0 pp → **failed** (−3.88 pp).

**Interpretation:** Compound stress label is highly imbalanced (5.5% train / 18.4% val positives); logistic dominates the frozen-backbone QNN head — H-Q12 interaction claim rejected.

See: `experiments/exp_068b_compound_stress_acyd/results.md`

---

## exp_080 — Quantum champion fusion on ACYD (C4)

**Profile:** acyd_soy_brazil_v1, frozen C4 + warm-start + noise p=0.03 + GV-ALR, 50,107 train / 5,830 val, RTX 4060

| Condition | Val ROC-AUC |
|-----------|-------------|
| Classical C4 (exp_060) | 0.6777 |
| Best frozen hybrid (exp_065 fixed) | 0.6771 |
| Champion fusion (noiseless eval) | 0.6709 |

**Gate:** ≥ classical − 1.0 pp **and** ≥ best hybrid + 0.5 pp → **failed** (parity −0.68 pp OK; lift −0.62 pp).

**Interpretation:** Stacking ACYD-validated components on `LargeNanoHybrid` does not beat the best single frozen-hybrid recipe — fusion is not free; keep winners as separate serve/ablation options.

See: `experiments/exp_080_quantum_champion_fusion_acyd/results.md`

---

## exp_079 — Cross-domain quantum head transfer (HIGGS → ACYD / H-Q13)

**Profile:** frozen C4 backbone; scratch vs `exp_037` HIGGS head init + fine-tune; 50,107 train / 5,830 val; RTX 4060

| Condition | Val ROC-AUC |
|-----------|-------------|
| Scratch head on C4 | 0.6749 |
| Transfer head (HIGGS → ACYD) | 0.6785 |

**Gate:** transfer − scratch ≥ +0.5 pp → **failed** (+0.35 pp). Honest-negative hypothesis **confirmed**.

**Interpretation:** Head-only quantum transfer across physics→agro tabular does not deliver a publication-scale win — do not market cross-domain QNN head reuse as a free boost.

See: `experiments/exp_079_quantum_transfer_higgs_to_acyd/results.md`

---

## How we use negative results

1. **Applicability gates** (`src/training/protocol.py`) prevent running expensive techniques on tasks where baselines already fail.
2. **Holm-Bonferroni** correction is mandatory for all multi-comparison claims.
3. **Publication figures** (`make figures`) include all experiments — successes and failures alike.
4. Future work (Phase 4) should target regimes where these techniques might succeed (higher dimension, real data).
