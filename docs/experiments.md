# Experiments

## Overview

Twenty-five experiments compare classical and quantum ML on synthetic and real-world tasks.
Configuration is centralized in `config/experiments.yaml`.

| ID | Name | Question |
|----|------|----------|
| 001 | Quantum vs Classical | Which learns faster on binary classification? |
| 002 | Hybrid Architecture | Does combining classical + quantum beat both alone? |
| 003 | Entanglement Effect | Does CNOT entanglement help or hurt learning? |
| 004 | Data Poisoning | Angle vs amplitude encoding under label noise? |
| 005 | Curriculum Quantum | Does staged easy→hard training beat random order? |
| 006 | Barren Plateau | How does gradient variance scale with qubit count? |
| 007 | Self-Play Quantum | Can a QNN improve by re-training on hard examples? |
| 008 | Data Re-upload | Does re-uploading beat basic QNN at matched qubits? |
| 009 | Entanglement Basic | Does entanglement help without re-upload? |
| 010 | Poison Re-upload Ablation | Do fewer layers or lower LR improve poison robustness? |
| 011 | UCI Tabular QML | Does QNN beat parameter-matched MLP on breast cancer? |
| 012 | MNIST PCA QML | Does amplitude encoding beat angle on PCA-reduced MNIST? |
| 013 | Augmentation Robustness | Does Gaussian augmentation improve QNN on noisy circles? |
| 014 | Sequence Baselines | Do RNN/Transformer beat flattened QNN on sequential data? |
| 015 | Adaptive QNN | Does gradient-variance LR beat fixed LR on plateau-prone 6q QNN? |
| 016 | Hybrid NAS | Does Optuna NAS beat fixed EXP 002 hybrid presets? |
| 017 | Poison × Topology | Does hybrid layout affect label-poison robustness? |
| 018 | Feature Fusion | Does Transformer → QNN beat PCA/flat QNN on phase sequences? |
| 019 | Nano Trainer Smoke | Does every registry model train via the app orchestrator? |
| 020 | API Smoke | Does REST API training match the Nano Trainer path? |
| 021 | QML Backend Parity | Do `default.qubit` and `lightning.qubit` agree within 2 pp on breast cancer QNN? |
| 022 | Nano Quantum Parity | Does hybrid sandwich beat param-matched classical on UCI tabular? |
| 023 | Encoding × Backend | Do angle vs amplitude and backend choice interact on PCA-MNIST? |
| 024 | QuantumNano-BC | Does hybrid sandwich match or beat logistic regression on full breast cancer? |
| 025 | Pima Generalization | Does exp_024 parity extend to Pima Indians Diabetes (OpenML id=37)? |

**Publication profile defaults:** `circles`, `noise=0.2`, `n_samples=500`, **10 seeds**, 30% holdout.

## Running an Experiment

```bash
# 1. Write hypothesis (required)
vim experiments/exp_003_entanglement_effect/hypothesis.md

# 2. Run
source .venv/bin/activate
python experiments/exp_003_entanglement_effect/run.py

# 3. View in dashboard
make dashboard-local    # http://localhost:8501 → [ REFRESH DATA ]

# 4. Document results
vim experiments/exp_003_entanglement_effect/results.md
```

## Experiment Details

### EXP 001 — Quantum vs Classical

**Models:** `classical_8`, `classical_32`, `quantum_4q_2l`, `quantum_6q_3l`, `quantum_reupload_4q_3l`  
**Eval:** 30% holdout, 10 seeds, bootstrap 95% CI  
**Result:** classical_32 best on n=500; see `results.md` for publication_large profile

### EXP 002 — Hybrid Architecture

**Architectures:** HybridSandwich, QuantumFirst, ClassicalFirst (re-upload QNN)  
**Eval:** 30% holdout, 10 seeds  
**Result:** No clear hybrid winner at n=500; QuantumFirst matches classical_32 at n=1000

### EXP 003 — Entanglement Effect

**Variants:** `none`, `chain`, `chain_half`, `ring` (re-upload QNN)  
**Eval:** 30% holdout, 10 seeds, Holm-corrected Wilcoxon vs `none`  
**Result:** `none` leads; no Holm-significant topology effect

### EXP 004 — Data Poisoning

**Poison rates:** 0%, 5%, 10%, 20%, 30%  
**Encodings:** `angle` (QuantumNetBasic) vs `amplitude` (QuantumNetAmplitude)  
**Eval:** train on poisoned labels, evaluate on **clean holdout test set** (30% split)  
**Compares:** Classical MLP vs both quantum encodings

### EXP 005 — Curriculum Quantum

**Methods:**
- `random` — shuffled baseline (eval on holdout)
- `margin_batches` — staged easy→hard batches (`curriculum_stages: 4`, `epochs_per_stage: 12`)

> Global margin ordering alone caused poor results (~50% acc). Batched curriculum fixes exposure bias.

### EXP 006 — Barren Plateau

**Qubit counts:** 2, 4, 6, 8, 10  
**Metric:** Mean gradient variance (concatenated parameter gradients, 50 random inits)  
**Implementation:** `src/training/gradients.py`

### EXP 007 — Self-Play Quantum

**Loop:** Predict on train pool → select misclassified → fine-tune → repeat (5 rounds)  
**Eval:** holdout test accuracy each round (no train/eval leakage)  
**Applicability gate:** base model must exceed `learnability_threshold` (0.55)

### EXP 008 — Data Re-upload

**Models:** `quantum_basic`, `quantum_reupload`, `classical_matched` (parameter-matched)  
**Eval:** 30% holdout, 10 seeds, paired Wilcoxon vs basic and classical  
**Implementation:** `src/quantum/qnn_reupload.py`, `src/training/param_match.py`

### EXP 009 — Entanglement Basic

**Variants:** `none`, `chain`, `chain_half`, `ring` (basic QNN, no re-upload)  
**Eval:** 30% holdout, 10 seeds, Holm-corrected Wilcoxon vs `none`  
**Goal:** Isolate entanglement effect without re-upload expressivity

### EXP 010 — Poison Re-upload Ablation

**Variants:** `reupload_3l`, `reupload_2l`, `reupload_lr_low`  
**Poison rates:** 0%, 30%  
**Eval:** 30% holdout, 10 seeds, Holm-corrected Wilcoxon vs `reupload_3l` baseline  
**Goal:** Test whether depth or learning rate fixes re-upload under poisoning

## Known Flags & Mitigations

| Flag | Mitigation applied |
|------|-------------------|
| exp_006 NaN gradients | Fixed: concatenate all param gradients before `.var()` |
| curriculum_margin ~50% | Replaced with `margin_batches` staged training |
| quantum poison collapse | Added amplitude encoding comparison + clean holdout eval |
| self-play 96% suspicious | Eval moved to held-out test set (30% split) |
| exp_005 curriculum worse at n=1000 | Documented as negative result in `results.md` |

## Suggested Ablations

| Experiment | Ablation |
|------------|----------|
| EXP 003 | `chain_half` vs full `chain` (already in config) |
| EXP 004 | `angle` vs `amplitude` under poison (already in config) |
| EXP 005 | Increase `curriculum_stages` or `epochs_per_stage` |
| EXP 006 | Add parameter-shift rule vs autograd for deep circuits |
| EXP 007 | Reduce `rounds` to check overfitting on holdout |
| EXP 008 | Vary `n_layers` for re-upload (see exp_010) |
| EXP 009 | Compare with exp_003 (re-upload vs basic entanglement) |
| EXP 010 | Add amplitude encoding baseline under same poison rates |
| EXP 011 | Run Optuna HPO (`make hpo`) before publication_large |
| EXP 012 | Compare PCA components 4 vs 8 vs 16 |
| EXP 013 | Vary `augment_sigma` |
| EXP 014 | Increase `seq_len` to test long-range dependencies |

## Real-Data Experiments (011–014)

| ID | Dataset | Models | Notes |
|----|---------|--------|-------|
| 011 | breast_cancer (UCI) | perceptron, classical_matched, quantum_angle | See `docs/baselines.md` |
| 012 | MNIST 0 vs 1 + PCA | quantum_angle, quantum_amplitude | 8 PCA components, 4 qubits |
| 013 | circles + noise | baseline vs augmented QNN | Uses `augmentation.py` |
| 014 | sequential_binary | RNNMini, TransformerMini, flattened QNN | Uses `rnn_mini.py`, `transformer_mini.py` |

### EXP 015 — Adaptive QNN (Phase 4 innovation)

**Models:** `quantum_6q_3l_fixed`, `quantum_6q_3l_adaptive`, `quantum_4q_2l_fixed`, `classical_matched`  
**Novelty:** Per-step gradient variance scales Adam LR (`src/training/adaptive_lr.py`)  
**Stats:** Paired Wilcoxon + Cohen's d + Holm-Bonferroni  
**Literature:** `docs/literature_review.md`  
**Ablation plan:** See `hypothesis.md` (var_target, qubit depth, warmup)

### EXP 016 — Hybrid NAS (Phase 6)

**Models:** `nas_best` (Optuna) vs `hybrid_sandwich`, `quantum_first`, `classical_first` (EXP 002 presets)  
**Search:** architecture × qubits × layers × LR × re-upload (`src/training/hpo.py`)  
**Trials:** 20 (publication), 3 (ci) — `make nas` or `python experiments/exp_016_hybrid_nas/run.py`  
**Stats:** Paired Wilcoxon vs each baseline + Holm-Bonferroni  

### EXP 017 — Poison × Topology (Phase 7)

**Models:** `hybrid_sandwich`, `quantum_first`, `classical_first`, `nas_preset` (EXP 016 best)  
**Poison:** train on flipped labels (0–30%); evaluate on clean 30% holdout  
**Stats:** `measure_robustness` per topology + Wilcoxon at 0% and 30% poison  
**Command:** `make poison-topology` or `python experiments/exp_017_poison_topology/run.py`  

### EXP 018 — Feature Fusion (Phase 8)

**Models:** `transformer_qnn_fusion`, `transformer_mini`, `quantum_pca`, `quantum_flat`  
**Dataset:** `sequential_phase` (12×4, phase-sensitive; PCA on flat windows insufficient)  
**Novelty:** `src/quantum/transformer_qnn_fusion.py` — encoder pools sequence → QNN  
**Command:** `make fusion` or `python experiments/exp_018_feature_fusion/run.py`  

### EXP 021 — QML Backend Parity (Phase 15)

**Models:** `quantum_default` (`default.qubit`), `quantum_lightning` (`lightning.qubit`)  
**Dataset:** breast cancer (same protocol as exp_011)  
**Claim:** Holdout accuracies within 2 pp across backends — simulator choice should not change conclusions  
**Command:** `python experiments/exp_021_qml_backend_parity/run.py`  
**Roadmap:** [`docs/research_agenda.md`](research_agenda.md)

### EXP 019 — Nano Trainer Smoke (Phase 9, infrastructure)

**Scope:** Validates `train_nanomodel.execute` for every model in `config/nanotrainer.yaml`  
**Profile:** `ci` — not a publication benchmark (`infrastructure: true` in config)  
**Models:** all registry pairs (tabular + sequence)  
**Success:** holdout accuracy ∈ [0.35, 1.0]; JSONL records with `exp_id=nano_train`  
**Command:** `make train-demo` or `python experiments/exp_019_nanotrainer_smoke/run.py`

### EXP 020 — API Smoke (Phase 10, infrastructure)

**Scope:** REST API `POST /api/v1/training-jobs` + SQLite persistence  
**Profile:** `ci` — infrastructure validation only  
**Pair:** perceptron + breast_cancer  
**Success:** `201 COMPLETED`, `GET /health` and `GET /ready` return 200  
**Command:** `make api-demo` or `python experiments/exp_020_api_smoke/run.py`

### EXP 022 — Nano Quantum Parity (Phase 17)

**Models:** `hybrid_sandwich` vs parameter-matched `classical_mlp`  
**Datasets:** breast_cancer, wine_binary (UCI tabular)  
**Claim:** Quantum nano model beats classical by ≥2 pp (Holm-significant)  
**Command:** `make nano-parity-bench` or `python experiments/exp_022_nano_quantum_parity/run.py`

### EXP 023 — Encoding × Backend (Phase 18)

**Models:** 2×2 factorial — angle/amplitude × `default.qubit`/`lightning.qubit`  
**Dataset:** MNIST 0 vs 1, PCA-8, 4 qubits, 2 layers  
**Claim:** Encoding gap within 2 pp across backends; no interaction term  
**Command:** `python experiments/exp_023_encoding_backend/run.py`  
**Pre-registration:** OSF link required before publication-profile runs (see `hypothesis.md`)

### EXP 024 — QuantumNano-BC (Phase 30 flagship)

**Models:** `hybrid_sandwich`, `logistic_regression`, `xgboost_shallow`, `perceptron`, parameter-matched classical  
**Dataset:** breast cancer (full 569 samples, no subsampling)  
**Claim:** Hybrid within 2 pp of logistic regression (parity) OR ≥3 pp advantage (Holm-significant)  
**Seeds:** 30 (publication profile)  
**Artifacts:** `model_cards/quantum_nano_bc.md`, checkpoints under `artifacts/exp_024/`  
**Command:** `python experiments/exp_024_quantum_nano_bc/run.py --profile publication`

### EXP 025 — Pima Generalization (Phase F)

**Folder:** `experiments/exp_025_pima_generalization/`  
**Dataset:** `pima_diabetes` (OpenML id=37, 768 samples, 8 features)  
**Claim:** Hybrid within 2 pp of logistic regression on second tabular benchmark (generalization vs exp_024)  
**Verdict:** Accepted (parity) — hybrid 76.2% vs logistic 77.2% (Δ=−1.0 pp, 30 seeds)  
**Seeds:** 30 (publication profile)  
**Command:** `QML_DEVICE=cuda python experiments/exp_025_pima_generalization/run.py --profile publication --write-results`

### EXP 032 — LargeNanoMLP on HIGGS (Phase L)

**Models:** `LargeNanoMLP` (~1.14M params) vs `LogisticRegression`  
**Dataset:** `higgs_v1` (805K train / 172.5K val, 28 features)  
**Claim:** Val ROC-AUC ≥ logistic + 1.0 pp  
**Verdict:** Accepted — 0.8258 vs 0.6849 (+14.09 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_032_large_nano_higgs/run.py --profile publication --write-results`

### EXP 058 — Conventional HIGGS baselines

**Models:** Shipped `LargeNanoMLP` vs sklearn `LogisticRegression`, `MLPClassifier`, `HistGradientBoosting`, `XGBoost`  
**Dataset:** `higgs_v1` (same split/scaler as exp_032)  
**Claim:** LargeNanoMLP ≥ best conventional + 0.5 pp val ROC-AUC  
**Verdict:** Rejected on publication (sklearn MLP 0.8429 vs nano 0.8358, −0.71 pp); CI slice accepted (+4.14 pp)  
**Command:** `python experiments/exp_058_conventional_higgs_baselines/run.py --profile publication --write-results`  
**Shortcut:** `python scripts/compare_higgs_conventional.py --profile ci`

### EXP 060 — LargeNanoMLP on ACYD Brazil soybean (C4 anchor)

**Models:** `LargeNanoMLP` (~1.16M params) vs `LogisticRegression`  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val, 37 features, temporal split)  
**Claim:** Val ROC-AUC ≥ logistic + 2.0 pp  
**Verdict:** Accepted — 0.6777 vs 0.6391 (+3.86 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_060_large_nano_acyd_soy/run.py --profile publication --write-results`  
**Ship:** `qml-ship --model large_nano_mlp_acyd_soy --skip-train`

### EXP 061 — Conventional ACYD baselines

**Models:** exp_060 `LargeNanoMLP` vs sklearn `LogisticRegression`, `MLPClassifier`, `HistGradientBoosting`, `XGBoost`  
**Dataset:** `acyd_soy_brazil_v1` (same temporal split/scaler as exp_060)  
**Claim:** LargeNanoMLP ≥ best conventional + 0.5 pp val ROC-AUC  
**Verdict:** Rejected — HistGradientBoosting 0.6941 vs nano 0.6777 (−1.64 pp); beats logistic (+3.86 pp, exp_060)  
**Command:** `QML_DEVICE=cuda python experiments/exp_061_conventional_acyd_baselines/run.py --profile publication --write-results`

### EXP 069 — LargeNanoMLP on NIHR synthetic CV (C2 anchor)

**Models:** `LargeNanoMLP` (~1.11M params) vs `LogisticRegression`  
**Dataset:** `nihr_cv_synthetic_v1` (70K train / 15K val, 13 features, ~8% prevalence)  
**Claim:** Val PR-AUC ≥ logistic + 1.0 pp  
**Verdict:** Rejected — logistic 0.2382 vs nano 0.2370 (−0.12 pp); checkpoint saved for hybrid ablations  
**Command:** `QML_DEVICE=cuda python experiments/exp_069_large_nano_nihr/run.py --profile publication --write-results`  
**Ship:** `qml-ship --model large_nano_mlp_nihr --skip-train --skip-gate`

### EXP 070 — LargeNanoMLP on GoBug code defects (C3 anchor)

**Models:** `LargeNanoMLP` (~1.14M params) vs `LogisticRegression`  
**Dataset:** `code_defects_gobug_v1` (27,172 train / 5,822 val, 23 features, temporal split)  
**Claim:** Val PR-AUC ≥ logistic + 2.0 pp  
**Verdict:** Rejected — logistic 0.3097 vs nano 0.3100 (+0.03 pp); checkpoint shipped for hybrid ablations  
**Command:** `QML_DEVICE=cuda python experiments/exp_070_large_nano_gobug/run.py --profile publication --write-results`  
**Ship:** `qml-ship --model large_nano_mlp_gobug --skip-train --skip-gate`

### Phase 1 closure — four classical nano anchors (C1–C4)

**Status:** Closed 2026-06-19 (RTX 4060, publication profile, `make ship-all-p0`).

| Anchor | Registry key | Experiment | Primary metric | Verdict |
|--------|--------------|------------|----------------|---------|
| C1 HIGGS | `large_nano_mlp_higgs` | exp_032 | ROC-AUC +14 pp vs logistic | Accepted |
| C2 NIHR | `large_nano_mlp_nihr` | exp_069 | PR-AUC −0.12 pp vs logistic | Rejected (shipped) |
| C3 GoBug | `large_nano_mlp_gobug` | exp_070 | PR-AUC +0.03 pp vs logistic | Rejected (shipped) |
| C4 ACYD | `large_nano_mlp_acyd_soy` | exp_060 | ROC-AUC +3.86 pp vs logistic | Accepted |

### EXP 076 — Conventional NIHR baselines

**Models:** exp_069 `LargeNanoMLP` vs sklearn `LogisticRegression`, `MLPClassifier`, `HistGradientBoosting`, `XGBoost`  
**Dataset:** `nihr_cv_synthetic_v1` (70K train / 15K val, 13 features, ~8% prevalence)  
**Claim:** Val PR-AUC ≥ best conventional + 0.5 pp  
**Verdict:** Rejected — logistic 0.2382 vs nano 0.2393 (+0.12 pp); nano ranks first but below gate  
**Command:** `QML_DEVICE=cuda python experiments/exp_076_conventional_nihr_baselines/run.py --profile publication --write-results`

### EXP 077 — Conventional GoBug baselines

**Models:** exp_070 `LargeNanoMLP` vs sklearn `LogisticRegression`, `MLPClassifier`, `HistGradientBoosting`, `XGBoost`  
**Dataset:** `code_defects_gobug_v1` (27,172 train / 5,822 val, 23 features, temporal split)  
**Claim:** Val PR-AUC ≥ best conventional + 0.5 pp  
**Verdict:** Rejected — HistGB 0.3276 vs nano 0.3174 (−1.02 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_077_conventional_gobug_baselines/run.py --profile publication --write-results`

### Phase 2 closure — four conventional sweeps (C1–C4)

**Status:** Closed 2026-06-19 (RTX 4060). All four domains swept; honest negatives documented.

| Domain | Experiment | Best conventional | Nano vs best |
|--------|------------|-------------------|--------------|
| HIGGS (C1) | exp_058 | sklearn MLP | −0.71 pp ROC-AUC |
| NIHR (C2) | exp_076 | logistic | +0.12 pp PR-AUC (gate rejected) |
| GoBug (C3) | exp_077 | HistGB | −1.02 pp PR-AUC |
| ACYD (C4) | exp_061 | HistGB | −1.64 pp ROC-AUC |

### EXP 062 — Hybrid QNN head on frozen ACYD LargeNanoMLP (C4)

**Models:** Frozen `LargeNanoMLP` (exp_060) + 4-qubit re-upload QNN head (~289 trainable params) vs classical sigmoid head (same backbone)  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val, temporal val 2019–2021)  
**Claim:** Hybrid val ROC-AUC ≥ classical head − 1.0 pp  
**Verdict:** Accepted — classical 0.6777 vs hybrid 0.6758 (−0.19 pp; within gate)  
**Command:** `QML_DEVICE=cuda python experiments/exp_062_hybrid_nano_acyd_soy/run.py --profile publication --write-results`  
**Checkpoint:** `artifacts/exp_062/large_nano_hybrid_acyd_soy/seed_42/best.pt`

### EXP 068 — Nano grand comparison synthesis (C1–C4)

**Scope:** Curated aggregation of publication metrics — no new GPU training  
**Domains:** HIGGS (C1), NIHR (C2), GoBug (C3), ACYD (C4)  
**Claim:** No quantum recipe wins on ≥3/4 domains with Δ ≥ +0.5 pp  
**Verdict:** Hypothesis **confirmed** — QNN head 4q complete on C1–C4; best +0.04 pp (HIGGS), none ≥ +0.5 pp  
**Artifacts:** `dist/leaderboards/nano_grand_comparison.json`, `paper/tables/grand_comparison.tex`  
**Command:** `python experiments/exp_068_nano_grand_comparison/run.py --profile publication --write-results`

### EXP 071 — Hybrid QNN head on frozen GoBug LargeNanoMLP (C3)

**Models:** Frozen `LargeNanoMLP` (exp_070) + 4-qubit re-upload QNN head (~289 trainable params) vs classical sigmoid head (same backbone)  
**Dataset:** `code_defects_gobug_v1` (27,172 train / 5,822 val, temporal val split)  
**Claim:** Hybrid val PR-AUC ≥ classical head − 1.0 pp  
**Verdict:** Accepted — classical 0.3174 vs hybrid 0.3175 (+0.02 pp; within gate)  
**Command:** `QML_DEVICE=cuda python experiments/exp_071_hybrid_nano_gobug/run.py --profile publication --write-results`  
**Checkpoint:** `artifacts/exp_071/large_nano_hybrid_gobug/seed_42/best.pt`

### EXP 068a — Seasonal angle encoding on ACYD (H-Q8)

**Models:** Frozen C4 classical head vs seasonal **angle** QNN vs seasonal **amplitude** QNN (4 cyclic features from in-season weather)  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val, temporal val 2019–2021)  
**Claim:** Angle val ROC-AUC ≥ classical + 0.5 pp and ≥ amplitude + 0.5 pp  
**Verdict:** Honest negative — classical 0.6777 vs angle 0.4979 (−17.98 pp) vs amplitude 0.5137 (−1.58 pp angle−amp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_068a_angle_encoding_acyd/run.py --profile publication --write-results`  
**Lesson:** Seasonal-only QNN head without backbone hidden state loses C4 representation; angle encoding does not rescue agro-climate tabular.

### EXP 072 — Quantum warm-start on NIHR hybrid (C2 replication)

**Models:** `HybridSandwich` end-to-end vs classical-first warm-start (70/30 epoch split)  
**Dataset:** `nihr_cv_synthetic_v1` (50,000 train / 15,000 val, 3 seeds)  
**Claim:** Warm-start val PR-AUC ≥ e2e hybrid + 0.5 pp  
**Verdict:** Honest negative — mean e2e 0.2343 vs warm-start 0.2307 (−0.35 pp); 1/3 paired wins  
**Command:** `QML_DEVICE=cuda python experiments/exp_072_quantum_warmstart_nihr/run.py --profile publication --write-results`  
**Lesson:** H-Q2 warm-start failure on HIGGS replicates on NIHR clinical tabular.

### EXP 073 — Quantum warm-start on GoBug hybrid (C3 replication)

**Models:** `HybridSandwich` end-to-end vs classical-first warm-start (70/30 epoch split)  
**Dataset:** `code_defects_gobug_v1` (27,172 train / 5,822 val, 3 seeds)  
**Claim:** Warm-start val PR-AUC ≥ e2e hybrid + 0.5 pp  
**Verdict:** Honest negative — mean e2e 0.3032 vs warm-start 0.3067 (+0.35 pp); 2/3 paired wins (gate not met)  
**Command:** `QML_DEVICE=cuda python experiments/exp_073_quantum_warmstart_gobug/run.py --profile publication --write-results`  
**Lesson:** Marginal warm-start gain on GoBug does not clear +0.5 pp gate — H-Q2 remains inconclusive on software tabular.

### EXP 074 — Dynamic entanglement schedule on NIHR (C2 replication)

**Models:** `QuantumNetEntangled` curriculum none→chain→ring vs fixed topologies  
**Dataset:** `nihr_cv_synthetic_v1` (10,000 train / 3,000 val, 3 seeds)  
**Claim:** Schedule val PR-AUC ≥ best fixed + 0.5 pp  
**Verdict:** Honest negative — mean schedule 0.1963 vs best fixed (ring) 0.2329 (−3.66 pp); 0/3 paired wins  
**Command:** `QML_DEVICE=cuda python experiments/exp_074_entangle_schedule_nihr/run.py --profile publication --write-results`  
**Lesson:** H-Q3 dynamic entanglement failure on breast cancer replicates on NIHR clinical tabular.

### EXP 075 — GV-ALR on frozen hybrid QNN head (NIHR C2 replication)

**Models:** Frozen C2 backbone + QNN head — fixed LR vs GV-ALR  
**Dataset:** `nihr_cv_synthetic_v1` (50,000 train / 15,000 val)  
**Claim:** |Δ PR-AUC| ≤ 0.3 pp and adaptive epochs ≤ 70% of fixed  
**Verdict:** Accepted — fixed 0.2392 vs GV-ALR 0.2369 (−0.24 pp); 5/8 epochs; wall-time ratio 0.58  
**Command:** `QML_DEVICE=cuda python experiments/exp_075_adaptive_hybrid_nihr/run.py --profile publication --write-results`  
**Lesson:** H-Q4 GV-ALR efficiency win on HIGGS replicates on NIHR clinical tabular.

**Next:** Phase 3/4 — exp_068b compound stress label (ACYD).

## Publication Profiles

| Profile | `n_samples` | Seeds | Command |
|---------|-------------|-------|---------|
| `publication` | 500 | 10 | `python experiments/exp_NNN/run.py` |
| `publication_large` | 1000 | 10 | `QML_PROFILE=publication_large python ...` or `make experiment-large` |

## References

- [PennyLane Docs](https://docs.pennylane.ai/)
- [QML Book (Schuld)](https://arxiv.org/abs/2103.09522)
- [Barren Plateaus Demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus/)
