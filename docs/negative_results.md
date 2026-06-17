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

## How we use negative results

1. **Applicability gates** (`src/training/protocol.py`) prevent running expensive techniques on tasks where baselines already fail.
2. **Holm-Bonferroni** correction is mandatory for all multi-comparison claims.
3. **Publication figures** (`make figures`) include all experiments — successes and failures alike.
4. Future work (Phase 4) should target regimes where these techniques might succeed (higher dimension, real data).
