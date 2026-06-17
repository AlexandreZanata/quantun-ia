# Hypothesis — exp_022_nano_quantum_parity

**Pre-registration:** https://osf.io/7m4pd (hybrid sandwich vs parameter-matched classical on UCI tabular)

## Question

Does the **hybrid sandwich** quantum nanomodel (`Classical → Quantum → Classical` with data re-upload) significantly outperform a **parameter-matched classical MLP** on UCI tabular tasks at an equal trainable-parameter budget?

## Expectation vs prior work

- **Different from exp_011:** exp_011 compared `quantum_angle` (basic QNN) vs matched MLP on breast cancer and found quantum **lost** (−5.6 pp, Holm-significant). Here we test the **hybrid sandwich** architecture with re-upload, which adds classical feature extraction before the variational circuit.
- **Different from exp_019:** exp_019 validates the Nano Trainer wiring only. This experiment uses the new **Nano Parity Bench** application (`qml-bench-parity`) that automatically downloads datasets and builds the matched classical baseline from the quantum model's parameter count.
- **Mechanism:** Classical pre/post layers may compensate for low-dimensional quantum readout while the quantum layer captures non-linear structure classical MLPs miss at the same param budget.

## Falsifiable claim

On `wine_binary` and `breast_cancer` with publication profile (10 seeds, 30% holdout):

1. `hybrid_sandwich` holdout accuracy **exceeds** `classical_matched_h*` by ≥ 2 pp mean difference.
2. Paired Wilcoxon test is Holm-significant at α = 0.05.

## What would prove us wrong

- Mean holdout accuracy for hybrid_sandwich ≤ matched classical on both datasets.
- Mean advantage &lt; 2 pp even if positive.
- p-value &gt; 0.05 after Holm correction (inconclusive or rejected).

## Metrics

- Holdout accuracy per seed (quantum vs matched classical)
- Parameter counts (|Δparams| ≤ 10)
- Cohen's d and magnitude
- Dataset download status (sklearn / torchvision MNIST)

## Models compared

| Role | Model | How obtained |
|------|-------|--------------|
| Quantum | `hybrid_sandwich` (4 qubits, 2 layers, re-upload) | `model_registry.build_model` |
| Classical | `classical_matched_h{N}` | `build_param_matched_classical(count_params(quantum))` |

Datasets are **downloaded automatically**: UCI via scikit-learn, MNIST via torchvision (`--download-only`).
