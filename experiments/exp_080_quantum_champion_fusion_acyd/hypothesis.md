# Hypothesis — EXP 080: Quantum champion fusion on ACYD (C4)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Fusing ACYD-validated winners into one shippable recipe — **frozen exp_060 C4 backbone**
+ 4-qubit re-upload QNN head trained with **head warm-start**, **depolarizing p=0.03**
(train-only), and **GV-ALR** — will:

1. Stay within **−1.0 pp** val ROC-AUC of C4 classical (exp_060, ~0.6777)
2. Beat the best single frozen-hybrid recipe (exp_065 fixed LR, **0.6771**) by **≥ +0.5 pp**

## Why I expect this

- exp_063 warm-start, exp_066 noise, exp_065 GV-ALR, and exp_062 head parity each passed
  on ACYD separately; combining them on the serve-compatible `LargeNanoHybrid` path should
  compound regularization + efficiency without leaving the C4 trunk.
- Entangle schedule (exp_064) and seasonal angle encoding (exp_068a) are **excluded**.
- Climate re-upload depth curriculum (exp_067) stays as static `reupload=True` on the hybrid
  head (curriculum API is `QuantumNetReupload`-only).

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary parity | Champion val ROC-AUC ≥ classical C4 **− 1.0 pp** |
| Primary lift | Champion val ROC-AUC ≥ best frozen hybrid (0.6771) **+ 0.5 pp** |
| Dataset | `acyd_soy_brazil_v1` temporal val 2019–2021 only |
| Architecture | `LargeNanoHybrid` / `open_hybrid` compatible with `large_nano_mlp_acyd_soy` |

## Known limitations

- Train-time noise uses `default.mixed`; eval/serve copies weights into noiseless head (p=0).
- Single seed publication (seed 42) unless extended; CI uses relaxed gates.
- Not operational planting advice — research benchmark only.
