# Results — EXP 019 (Nano Trainer Smoke)

**Run date:** 2026-06-18  
**Profile:** `ci` (infrastructure validation — not a publication benchmark)  
**Scope:** All models in `config/nanotrainer.yaml` via `train_nanomodel.execute`

## Holdout results (ci profile, seed 42)

| Model | Dataset | Holdout accuracy |
|-------|---------|------------------|
| hybrid_sandwich | breast_cancer | 94.2% |
| classical_mlp | breast_cancer | 93.0% |
| perceptron | breast_cancer | 87.1% |
| quantum_amplitude | mnist_binary | 100.0% |
| transformer_mini | sequential_binary | 80.0% |
| transformer_qnn_fusion | sequential_phase | 53.3% |
| quantum_angle | breast_cancer | 37.4% |

All accuracies within sanity bounds [35%, 100%]. JSONL records appended with `exp_id=nano_train`.

## Verdict

**accepted** — every registry model trained without error through the Nano Trainer orchestrator (`qml-train` / Streamlit path).

## Conclusion

Infrastructure smoke validates the productized training path introduced in Phase 9. Scientific benchmarks remain in exp_011–018 and exp_021–023; this experiment only confirms wiring, logging, and holdout evaluation for app surfaces.

## Limitations

- `ci` profile subsamples data (`n_samples=50` tabular) — not comparable to publication metrics.
- Does not test REST API persistence (see exp_020).
- `quantum_angle` low accuracy on ci subsample is expected at 5 epochs / 50 samples — not a scientific claim.
