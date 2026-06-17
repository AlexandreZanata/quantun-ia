# Gradient-Variance Adaptive Learning Rate (GV-ALR)

**Module:** `src/training/adaptive_lr.py`  
**Primary experiment:** [exp_015](../experiments/exp_015_adaptive_qnn/hypothesis.md)  
**Software version:** v0.9.15 (Phase 24)

GV-ALR rescales Adam's learning rate from per-step gradient variance to mitigate
barren-plateau stagnation on deep variational circuits. It is a **secondary method**
track — deferred from the primary Option C benchmark paper until a Holm-significant
holdout win is demonstrated at 6 qubits × 3 layers.

---

## Algorithm

**Input:** model parameters θ, training batch (X, y), base learning rate η₀, variance
target τ, warmup epochs W, scale bounds [s_min, s_max].

**Per epoch** (after warmup):

1. Forward pass and backward pass on the training batch.
2. Flatten all parameter gradients g into a single vector.
3. Compute gradient variance v = Var(g) (0 if |g| ≤ 1).
4. Scale factor s = clamp(√(τ / v), s_min, s_max); if v ≈ 0, use s_max.
5. Set Adam learning rate η = η₀ × s before `optimizer.step()`.
6. Log `learning_rate`, `grad_variance`, and holdout metrics to `ExperimentLogger`.

**Intuition:** when gradient variance collapses (barren plateau), increase η to escape
flat regions; when variance is high, decrease η for stability.

### Pseudocode

```
function compute_lr_scale(grad_var, var_target, min_scale, max_scale):
    if grad_var <= 0 or grad_var < 1e-12:
        return max_scale
    ratio = var_target / grad_var
    scale = sqrt(ratio)
    return clamp(scale, min_scale, max_scale)

for epoch in 0 .. epochs-1:
    loss.backward()
    if epoch >= warmup_epochs and epoch % adapt_every == 0:
        grad_var = variance(flatten(model.grads))
        scale = compute_lr_scale(grad_var, var_target, min_scale, max_scale)
        optimizer.lr = base_lr * scale
    optimizer.step()
```

Reference implementation: `compute_lr_scale()` and `train_model_adaptive()` in
`src/training/adaptive_lr.py`.

---

## Configuration

Defaults live in `AdaptiveLRConfig` and `config/experiments.yaml` under
`exp_015_adaptive_qnn.adaptive_lr`:

| Parameter | Default | Role |
|-----------|---------|------|
| `base_lr` | 0.02 | Adam base η₀ |
| `var_target` | 0.015 | Target gradient variance τ (calibrated from exp_006 at 4q) |
| `min_scale` | 0.25 | Lower bound on LR multiplier |
| `max_scale` | 4.0 | Upper bound on LR multiplier |
| `warmup_epochs` | 3 | Epochs with fixed η₀ before adaptation |
| `adapt_every` | 1 | Adapt LR every N epochs |

Diagnostic helper `step_gradient_variance()` exposes the per-step variance used in
exp_006 barren-plateau analysis.

---

## Experiment linkage

| Artifact | Path |
|----------|------|
| Hypothesis | `experiments/exp_015_adaptive_qnn/hypothesis.md` |
| Results | `experiments/exp_015_adaptive_qnn/results.md` |
| Config | `config/experiments.yaml` → `exp_015_adaptive_qnn` |
| Unit tests | `tests/unit/test_adaptive_lr.py` |
| Applicability gate | JSONL `record_type: applicability_gate` for curriculum technique |

**Models compared (publication profile, 10 seeds):**

- `quantum_6q_3l_adaptive` — GV-ALR on plateau-prone 6q×3l QNN
- `quantum_6q_3l_fixed` — fixed Adam LR control
- `quantum_4q_2l_fixed` — shallower circuit control
- `classical_matched_h*` — parameter-matched classical baseline

Run:

```bash
MLFLOW_DISABLE=1 python experiments/exp_015_adaptive_qnn/run.py
```

---

## Empirical status

Publication-profile results (circles, noise=0.2):

| Comparison | Δ (pp) | Cohen's d | Wilcoxon p | Verdict |
|------------|--------|-----------|------------|---------|
| adaptive vs fixed (6q×3l) | +5.3 | 0.78 (medium) | 0.059 | not Holm-significant |
| adaptive vs 4q fixed | +7.8 | 1.40 (large) | 0.012 | Holm-significant |

**Overall verdict:** **inconclusive** for the primary 6q claim — |d|=0.78 below
minimum detectable effect 0.89 at n=10 seeds. Documented honestly in `results.md`;
not cited as a headline win in the primary paper (Option C).

---

## Limitations

- Circles dataset only in the primary publication run — UCI/MNIST transfer untested.
- Adam only — no SPSA or natural-gradient variants.
- `var_target` hand-tuned from exp_006, not learned online.
- Adaptation uses full-batch gradient variance on the training split (not mini-batch MC).

---

## References

- McClean et al. (2018) — barren plateaus in quantum neural networks.
- Cincio et al. (2022) — cost landscape geometry and trainability.
- Grant et al. (2019) — adaptive methods for variational optimization.

See `docs/literature_review.md` for full bibliography.

---

## Citation

When referencing this method, cite the software (`CITATION.cff`) and experiment
exp_015 with the publication profile seed list. After Zenodo DOI is live, prefer
the versioned archive over commit SHA alone.
