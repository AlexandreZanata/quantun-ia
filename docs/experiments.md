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

### EXP 068b — Compound stress label on ACYD (H-Q12)

**Models:** Frozen C4 backbone + QNN head vs logistic on compound-stress label  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val, compound label)  
**Claim:** Hybrid val ROC-AUC ≥ logistic + 1.0 pp  
**Verdict:** Honest negative — logistic 0.8462 vs hybrid 0.8074 (−3.88 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_068b_compound_stress_acyd/run.py --profile publication --write-results`  
**Lesson:** QNN head does not capture drought∧heat interaction better than logistic on compound stress labels.

### EXP 078 — Agro Risk Lab human validation (8 Brazilian cases)

**Models:** `exp_060` LargeNanoMLP (C4) on hand-crafted municipality scenarios  
**Dataset:** `acyd_soy_brazil_v1` (inference only — no retrain)  
**Claim:** Spearman ρ ≥ 0.85 and min(H) − max(L) separation > 0  
**Verdict:** Accepted — ρ=0.9762; separation +23.08 pp; dashboard `06_agro_risk_lab.py`  
**Command:** `QML_DEVICE=cuda python experiments/exp_078_agro_clinical_cases/run.py --write-results`  
**Lesson:** C4 ranks Brazilian drought/heat scenarios in agronomically sensible order for human-facing Agro Risk Lab.

### EXP 063 — Quantum warm-start on ACYD hybrid (C4 / H-Q9)

**Models:** `HybridSandwich` end-to-end vs classical-first warm-start (70/30 epoch split)  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val, 3 seeds)  
**Claim:** Warm-start val ROC-AUC ≥ e2e hybrid + 0.5 pp  
**Verdict:** Accepted — mean e2e 0.6574 vs warm-start 0.6680 (+1.06 pp); 3/3 paired wins  
**Command:** `QML_DEVICE=cuda python experiments/exp_063_quantum_warmstart_acyd/run.py --profile publication --write-results`  
**Lesson:** H-Q2 warm-start failure on HIGGS/NIHR does **not** replicate on agro-climate tabular — phenology-style classical-first helps on ACYD.

### EXP 064 — Dynamic entanglement schedule on ACYD (C4 / H-Q3)

**Models:** `QuantumNetEntangled` curriculum none→chain→ring vs fixed topologies  
**Dataset:** `acyd_soy_brazil_v1` (10,000 train / 3,000 val, 3 seeds)  
**Claim:** Schedule val ROC-AUC ≥ best fixed + 0.5 pp  
**Verdict:** Honest negative — mean schedule 0.6314 vs best fixed (none) 0.6422 (−1.08 pp); 1/3 paired wins  
**Command:** `QML_DEVICE=cuda python experiments/exp_064_entangle_schedule_acyd/run.py --profile publication --write-results`  
**Lesson:** H-Q3 dynamic entanglement failure replicates on ACYD agro tabular.

### EXP 065 — GV-ALR on frozen hybrid QNN head (ACYD C4 / H-Q4)

**Models:** Frozen C4 backbone + QNN head — fixed LR vs GV-ALR  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val)  
**Claim:** |Δ ROC-AUC| ≤ 0.3 pp and adaptive epochs ≤ 70% of fixed  
**Verdict:** Accepted — fixed 0.6771 vs GV-ALR 0.6763 (−0.08 pp); 5/8 epochs; wall-time ratio 1.08  
**Command:** `QML_DEVICE=cuda python experiments/exp_065_gv_alr_hybrid_acyd/run.py --profile publication --write-results`  
**Lesson:** H-Q4 GV-ALR efficiency win replicates on ACYD (metric parity; epoch budget met).

### EXP 066 — Depolarizing noise reg on ACYD (H-Q10 / H-Q5)

**Models:** `HybridSandwich` vs `NoiseRegularizedHybridSandwich` (p=0.03)  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val / 5,856 temporal test ≥2022)  
**Claim:** Noisy test ROC-AUC ≥ noiseless + 0.5 pp  
**Verdict:** Accepted — noiseless 0.6293 vs noisy 0.6392 (+0.99 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_066_noise_reg_acyd/run.py --profile publication --write-results`  
**Lesson:** Noise regularization that was inconclusive on GoBug **passes** on ACYD temporal test-year generalization.

### EXP 067 — Re-upload climate feature-block ladder (ACYD / H-Q11)

**Models:** `QuantumNetReupload` depth curriculum 1→2→3 vs fixed L=3 on climate feature masks  
**Dataset:** `acyd_soy_brazil_v1` (rungs: temp-only, temp+precip, full 37-d)  
**Claim:** Curriculum wins ≥ 2/3 rungs by ≥ +0.3 pp ROC-AUC  
**Verdict:** Accepted — 2/3 wins (temp_only +1.08 pp; full_37 +4.21 pp; temp_precip −0.75 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_067_reupload_ladder_acyd/run.py --profile publication --write-results`  
**Lesson:** Climate feature-block curriculum succeeds on ACYD where cross-domain ladder (exp_056) failed.

### Phase 3 closure — ACYD quantum replications (C4)

**Status:** Closed 2026-07-12 (RTX 4060, publication profile).

| Exp | Method | Verdict | Δ / note |
|-----|--------|---------|----------|
| exp_063 | Warm-start | Accepted | +1.06 pp vs e2e |
| exp_064 | Entangle schedule | Rejected | −1.08 pp vs none |
| exp_065 | GV-ALR head | Accepted | −0.08 pp · 5/8 ep |
| exp_066 | Noise reg | Accepted | +0.99 pp vs noiseless |
| exp_067 | Re-upload ladder | Accepted | 2/3 rung wins |

### EXP 080 — Quantum champion fusion on ACYD (C4)

**Models:** Frozen C4 + fused recipe (head warm-start + depolarizing p=0.03 train-only + GV-ALR) vs C4 classical and best frozen hybrid (exp_065 fixed LR 0.6771)  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val)  
**Claim:** Champion ≥ classical − 1.0 pp **and** ≥ best hybrid + 0.5 pp  
**Verdict:** Honest negative — classical 0.6777 vs champion 0.6709 (−0.68 pp parity OK); vs best hybrid −0.62 pp (lift gate failed)  
**Command:** `QML_DEVICE=cuda python experiments/exp_080_quantum_champion_fusion_acyd/run.py --profile publication --write-results`  
**Lesson:** Combining ACYD winners on the serve-compatible `LargeNanoHybrid` path does not beat the best single frozen-hybrid recipe; keep components as separate ablations.

### EXP 082 — Isotonic calibration on ACYD C4 (agro)

**Model:** `exp_060` LargeNanoMLP · isotonic on temporal val  
**Dataset:** `acyd_soy_brazil_v1` (5,830 val rows; fit 80% / eval 20%)  
**Claim:** ECE ≤ 0.08 and improved; Brier not worse; AUC Δ ≥ −0.005; agro Spearman ρ ≥ 0.85  
**Verdict:** Accepted — ECE 0.0538→0.0355; Brier 0.2282→0.2237; AUC 0.6789→0.6767; ρ=0.9820  
**Command:** `QML_DEVICE=cuda python experiments/exp_082_calibration_acyd/run.py --profile publication --write-results`  
**Artifact:** `artifacts/exp_082/large_nano_mlp_acyd_soy_brazil_v1/seed_42/calibration_isotonic.json`  
**Lesson:** Isotonic improves Agro Risk Lab probability calibration without hurting ranking.

### EXP 079 — Cross-domain quantum head transfer (HIGGS → ACYD / H-Q13)

**Models:** Frozen C4 backbone + scratch QNN head vs HIGGS-pretrained head (`exp_037`) fine-tuned on ACYD  
**Dataset:** `acyd_soy_brazil_v1` (50,107 train / 5,830 val)  
**Claim:** Transfer does **not** beat scratch by ≥ +0.5 pp (honest-negative design)  
**Verdict:** Honest negative confirmed — scratch 0.6749 vs transfer 0.6785 (+0.35 pp; below gate)  
**Command:** `QML_DEVICE=cuda python experiments/exp_079_quantum_transfer_higgs_to_acyd/run.py --profile publication --write-results`  
**Lesson:** Cross-domain QNN head transfer from physics to agro does not clear a +0.5 pp win — transfer hype unsupported.

### EXP 081 — LargeNanoMLP on ACYD Brazil maize (C4b)

**Models:** `LargeNanoMLP` (~1.16M) vs logistic  
**Dataset:** `acyd_maize_brazil_v1` (151,956 train / 13,566 val / 13,537 temporal test ≥2022)  
**Claim:** Val ROC-AUC ≥ logistic + 2.0 pp  
**Verdict:** Accepted — logistic 0.6983 vs nano 0.8086 (+11.03 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_081_large_nano_acyd_maize/run.py --profile publication --write-results`  
**Lesson:** Same LargeNanoMLP recipe as C4 soybean transfers strongly to maize (corn) on the shared climate/soil feature panel.

### EXP 083 — Conventional baselines vs LargeNanoMLP on ACYD maize (C4b floor)

**Models:** Logistic / sklearn MLP / HistGB / XGBoost vs `exp_081` LargeNanoMLP checkpoint  
**Dataset:** `acyd_maize_brazil_v1` (151,956 train / 13,566 val)  
**Claim:** Nano ≥ best conventional + 0.5 pp ROC-AUC  
**Verdict:** Honest negative — HistGB 0.8178 vs nano 0.8086 (−0.92 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_083_conventional_acyd_maize_baselines/run.py --profile publication --write-results`  
**Lesson:** Mirrors soybean exp_061 — boosting beats nano on agro tabular while nano still dominates logistic.

### EXP 084 — Residual / NarrowDeep / FT-lite nano vs HistGB (Phase A H-N1)

**Models:** ResidualNanoMLP, NarrowDeepNano, FTLiteNano vs HistGB  
**Dataset:** `acyd_maize_brazil_v1` (151,956 train / 13,566 val)  
**Claim:** Best nano ≥ HistGB + 0.5 pp ROC-AUC  
**Verdict:** Honest negative — HistGB 0.8178 vs ResidualNano 0.8086 (−0.92 pp); FT-lite collapsed (0.5602)  
**Command:** `QML_DEVICE=cuda python experiments/exp_084_residual_ft_nano_maize/run.py --profile publication --write-results`  
**Lesson:** Architecture swaps alone do not close the agro boosting gap — prefer HistGB→nano distillation (exp_092).

### EXP 092 — HistGB → ResidualNano soft-label distillation (Phase D H-N3)

**Models:** HistGB teacher · hard ResidualNano control · soft-label ResidualNano student  
**Dataset:** `acyd_maize_brazil_v1` (151,956 train / 13,566 val)  
**Claim:** Distill student ≥ HistGB − 1.0 pp ROC-AUC  
**Verdict:** Accepted — HistGB 0.8178 vs distill 0.8130 (−0.48 pp); distill beats hard control by +0.55 pp  
**Command:** `QML_DEVICE=cuda python experiments/exp_092_histgb_distill_nano_maize/run.py --profile publication --write-results`  
**Lesson:** Soft BCE distillation closes most of the agro boosting gap without beating HistGB outright.

### EXP 085 — Sample-efficiency curves (HistGB vs distill nano)

**Models:** HistGB · hard ResidualNano · distill ResidualNano at 1/5/20/100% stratified row budgets  
**Dataset:** `acyd_maize_brazil_v1` (full temporal val)  
**Claim:** Distill ≥ HistGB on ≥ 2/4 budgets  
**Verdict:** Honest negative — 0/4 wins; AULC distill 0.7831 < HistGB 0.7973  
**Command:** `QML_DEVICE=cuda python experiments/exp_085_sample_efficiency_agro/run.py --profile publication --write-results`  
**Artifact:** `artifacts/exp_085/curves_publication.json`  
**Lesson:** HistGB remains more sample-efficient than distill nano on agro tabular; distill still helps vs hard labels at mid/full budgets.

### EXP 086 — Residual-skip QNN vs plain QNN on distill ResidualNano (H-Q2.1)

**Models:** Frozen exp_092 distill backbone · plain 4q re-upload · residual-skip 4q  
**Dataset:** `acyd_maize_brazil_v1` (50k hybrid fine-tune / 13,566 val)  
**Claim:** Residual-skip ≥ plain + 0.5 pp; both ≥ classical − 1.0 pp  
**Verdict:** Honest negative — residual +0.07 pp vs plain; parity OK (plain −0.09 / residual −0.02 pp vs classical 0.8129)  
**Command:** `QML_DEVICE=cuda python experiments/exp_086_residual_qnn_head_maize/run.py --profile publication --write-results`  
**Lesson:** Classical skip around QNN does not unlock a ≥0.5 pp lift; hybrids already sit at classical parity.

### EXP 087 — Fourier re-upload vs flat angle head (H-Q2.2)

**Models:** Frozen exp_092 distill backbone · flat tanh→AngleEmbedding · Fourier sin/cos→AngleEmbedding · layers {1,2,3}  
**Dataset:** `acyd_maize_brazil_v1` (30k hybrid fine-tune / 13,566 val)  
**Claim:** Fourier wins ≥ 2/3 re-upload depth rungs vs flat  
**Verdict:** Honest negative — 0/3 rung wins; deepest Fourier −0.24 pp vs classical 0.8130 (parity OK)  
**Command:** `QML_DEVICE=cuda python experiments/exp_087_fourier_reupload_climate/run.py --profile publication --write-results`  
**Lesson:** Seasonal Fourier map into re-upload does not beat flat angle encoding on maize distill features.

### EXP 089 — Measurement-dropout QNN calibration (H-Q2.4)

**Models:** Frozen exp_092 distill backbone · plain 4q re-upload · measurement-dropout 4q (p=0.2, MC=16)  
**Dataset:** `acyd_maize_brazil_v1` (30k hybrid fine-tune / 13,566 val)  
**Claim:** ECE relative ≥ 20% vs plain without AUC drop > 0.5 pp  
**Verdict:** Honest negative — ECE 0.0242→0.0374 (−54.8% rel.); AUC −0.08 pp (floor held)  
**Command:** `QML_DEVICE=cuda python experiments/exp_089_measurement_dropout_cal/run.py --profile publication --write-results`  
**Lesson:** Stochastic measurement masking hurts calibration on this hybrid; keep classical isotonic (exp_082) for ECE.

### EXP 091 — Circuit-cut effective 6q head (H-Q2.5)

**Models:** Frozen exp_092 distill backbone · classical bottleneck head · plain 4q · circuit-cut 2×4q (effective 6q)  
**Dataset:** `acyd_maize_brazil_v1` (30k hybrid fine-tune / 13,566 val)  
**Claim:** Circuit-cut ≥ classical head − 1.0 pp  
**Verdict:** Accepted — cut 0.8125 vs classical 0.8129 (−0.03 pp); plain 4q 0.8128  
**Command:** `QML_DEVICE=cuda python experiments/exp_091_circuit_cut_6q/run.py --profile publication --write-results`  
**Lesson:** Soft overlapping 4q fragments reach classical parity on a 4060; no AUC lift vs plain 4q (−0.03 pp).

### EXP 093 — Projected quantum kernel ridge head (H-Q2.6)

**Models:** LogisticRegression · KernelRidge RBF on 1-local 4q projections · Nyström→logistic · HistGB honesty  
**Dataset:** `acyd_maize_brazil_v1` (15k train / 13,566 val)  
**Claim:** KernelRidge(PQK) ≥ logistic + 0.5 pp  
**Verdict:** Honest negative — KernelRidge 0.5307 vs logistic 0.6972 (−16.65 pp); φ logistic 0.5082  
**Command:** `QML_DEVICE=cuda python experiments/exp_093_pqk_ridge_head/run.py --profile publication --write-results`  
**Lesson:** 1-local projected quantum features destroy ranking vs raw logistic; close Phase B quantum backlog.

### EXP 097 — SPEI-proxy curriculum (D-T3)

**Models:** ResidualNanoMLP · random staged curriculum · SPEI easy→hard staged curriculum · HistGB honesty  
**Dataset:** `acyd_maize_brazil_v1` (full train / 13,566 val)  
**Claim:** SPEI curriculum ≥ random + 0.5 pp  
**Verdict:** Accepted — SPEI 0.8025 vs random 0.7942 (**+0.83 pp**)  
**Command:** `QML_DEVICE=cuda python experiments/exp_097_spei_curriculum_agro/run.py --profile publication --write-results`  
**Lesson:** Drought-severity easy→hard batch order helps ResidualNano vs matched random stages; still below HistGB 0.8203.

### EXP 098 — Continual crop-year fine-tune (D-T4)

**Models:** ResidualNanoMLP joint · ResidualNanoMLP year-by-year continual · HistGB honesty  
**Dataset:** ACYD maize with year column (`processed/continual_v1`, 37 train years ≤ 2018 / val 2019–2021)  
**Claim:** Continual ≥ joint − 1.0 pp  
**Verdict:** Honest negative — continual 0.7713 vs joint 0.8005 (**−2.91 pp**); backward mean 0.7078  
**Command:** `QML_DEVICE=cuda python experiments/exp_098_continual_crop_year/run.py --profile publication --write-results`  
**Lesson:** Naive year-by-year fine-tune forgets; prefer joint ResidualNano (or distill) over sequential crop-year training without replay/EWC.

### EXP 099 — Masked climate SSL pretrain (D-T5)

**Models:** ResidualNanoMLP scratch · ResidualNanoSSL mask-reconstruct → fine-tune · HistGB honesty  
**Dataset:** `acyd_maize_brazil_v1` (full train / 13,566 val)  
**Claim:** SSL ≥ scratch + 0.5 pp  
**Verdict:** Honest negative — SSL 0.8143 vs scratch 0.8110 (**+0.33 pp**, below gate)  
**Command:** `QML_DEVICE=cuda python experiments/exp_099_ssl_climate_pretrain/run.py --profile publication --write-results`  
**Lesson:** Masked weather SSL helps slightly but not enough for a +0.5 pp claim; keep distill ResidualNano as serve default.

### EXP 096 — GoBug streaming ResidualNano (C-T6)

**Models:** LogisticRegression · ResidualNanoMLP joint · ResidualNanoMLP chronological streaming  
**Dataset:** `code_defects_gobug_v1` (27,172 train / 5,822 val, sha-ordered)  
**Claim:** Streaming ≥ joint − 1.0 pp PR-AUC  
**Verdict:** Accepted — streaming 0.3069 vs joint 0.2995 (**+0.75 pp**); logistic 0.3097  
**Command:** `QML_DEVICE=cuda python experiments/exp_096_gobug_streaming_nano/run.py --profile publication --write-results`  
**Lesson:** Commit-time chronological fine-tune matches/beats matched-budget joint on GoBug PR-AUC; still near logistic.

### EXP 088 — Pauli/shadow features → NarrowDeepNano (H-Q2.3)

**Models:** Logistic · NarrowDeepNano (raw 37-d) · NarrowDeepNano (64-d Pauli features) · HistGB honesty  
**Dataset:** `acyd_maize_brazil_v1` (20k train / 13,566 val)  
**Claim:** Shadow nano ≥ classical − 0.5 pp and ≥ logistic + 2.0 pp  
**Verdict:** Honest negative — shadow 0.6050 vs classical 0.7610 (−15.60 pp); vs logistic −9.11 pp  
**Command:** `QML_DEVICE=cuda python experiments/exp_088_shadow_features_nano_maize/run.py --profile publication --write-results`  
**Lesson:** Fixed 4q Pauli/shadow feature map loses ranking signal vs raw climate columns; do not claim quantum feature advantage on maize.

**Next:** Phase C multi-crop / hard drift (`exp_090` / `exp_094`).

### EXP 090 — Multi-crop joint ResidualNano (soy + maize)

**Models:** Maize-solo ResidualNano · Joint ResidualNano (38-d + crop bit) · HistGB maize honesty  
**Dataset:** `acyd_soy_brazil_v1` + `acyd_maize_brazil_v1` (full temporal splits)  
**Claim:** Joint maize val ≥ solo − 0.5 pp  
**Verdict:** Honest negative — joint 0.7938 vs solo 0.8073 (−1.34 pp); soy val 0.6758  
**Command:** `QML_DEVICE=cuda python experiments/exp_090_multicrop_joint_nano/run.py --profile publication --write-results`  
**Lesson:** Shared soy+maize trunk hurts maize ranking; keep crop-specific (maize distill) models for serve.

**Next:** Paper cycle 2 synthesis and/or hard temporal drift (`exp_094`).

### EXP 094 — Hard temporal drift ResidualNano vs HistGB

**Models:** HistGB · ResidualNanoMLP  
**Dataset:** ACYD maize hard drift (train ≤ 2016 / val 2017–2018 / test ≥ 2022)  
**Claim:** nano ≥ HistGB − 1.0 pp on hard-drift val  
**Verdict:** Accepted — HistGB 0.8246 vs nano 0.8185 (−0.61 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_094_hard_temporal_drift/run.py --profile publication --write-results`  
**Lesson:** Under stronger year shift, ResidualNano still tracks HistGB within 1 pp; keep distill serve for standard split.

### EXP 095 — CY-Bench maize US sample (ResidualNano vs HistGB)

**Models:** HistGB · ResidualNanoMLP  
**Dataset:** `cybench_maize_us_v1` (AgML sample US designed features, EUPL-1.2; yield lags excluded)  
**Claim:** nano ≥ HistGB − 1.0 pp on temporal val (2012–2015)  
**Verdict:** Honest negative — nano 0.7474 vs HistGB 0.7855 (−3.80 pp)  
**Command:** `QML_DEVICE=cuda python experiments/exp_095_cybench_maize_slice/run.py --profile publication --write-results`  
**Lesson:** External CY-Bench climate/RS features favor HistGB over ResidualNano under binary low-yield; keep ACYD distill as maize serve default.

### Phase F — Paper cycle 2 tables + arXiv bundle

**Artifacts:** `paper/tables/boosting_frontier.tex`, `sample_efficiency.tex`, `quantum_v2.tex`, `multicrop.tex`, `hard_drift.tex`  
**Registry:** `config/cycle_v2_paper.yaml` (curated from RTX 4060 `results.md`)  
**Commands:** `make cycle-v2-tables` · `make paper-build` · `make arxiv-bundle`  
**Bundle:** `dist/arxiv/quantun-ia-paper.tar.gz` (F-T4 local ready; paste `arxiv_id` / Zenodo DOI after upload)  
**Status:** closed — Cycle v2 accepts (exp_092, exp_091, exp_094) + honest negatives (incl. exp_087) in paper + arXiv tarball

### Phase E — Agro Maize Lab + uncertainty (E-T3)

**Dashboard:** `dashboard/pages/07_agro_maize_lab.py`  
**API:** maize predict returns MC-dropout `uncertainty_std`  
**Compare helper:** `src/application/agro_maize_compare.py` (published HistGB / quantum floors)

## Publication Profiles

| Profile | `n_samples` | Seeds | Command |
|---------|-------------|-------|---------|
| `publication` | 500 | 10 | `python experiments/exp_NNN/run.py` |
| `publication_large` | 1000 | 10 | `QML_PROFILE=publication_large python ...` or `make experiment-large` |

## References

- [PennyLane Docs](https://docs.pennylane.ai/)
- [QML Book (Schuld)](https://arxiv.org/abs/2103.09522)
- [Barren Plateaus Demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus/)
