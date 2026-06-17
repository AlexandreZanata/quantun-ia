# Ethics and Data Use Statement

**Lab:** Quantum-Inspired Micro ML Lab (`quantun-ia`)  
**Last updated:** 2026-06-17

---

## Data sources

| Dataset | Source | License / terms | Use in this repo |
|---------|--------|-----------------|------------------|
| **UCI breast cancer** | scikit-learn (`load_breast_cancer`) | Open ML / UCI ML Repository | exp_011, exp_021, exp_022, Nano Trainer |
| **UCI wine / iris** | scikit-learn | Open ML / UCI | exp_022 parity bench (binary labels) |
| **MNIST** | torchvision | [Yann LeCun / CC BY-SA](http://yann.lecun.com/exdb/mnist/) | exp_012, PCA-reduced binary subset |
| **Synthetic circles / moons** | `src/data/generators.py` | MIT (this repo) | exp_001–010, NAS, poisoning |
| **Sequential phase** | `src/data/generators.py` | MIT (this repo) | exp_014, exp_018 |

We do **not** collect personal data. All tabular sets are de-identified public benchmarks. MNIST is used only as a research benchmark (0 vs 1 subset with PCA), not for biometric identification.

---

## Responsible use

- Results report **holdout accuracy** on public benchmarks — not clinical or deployment claims.
- Quantum models are **simulator-based** (PennyLane); no hardware queue or patient data.
- Negative and inconclusive results are documented in [`negative_results.md`](negative_results.md).

---

## Pre-registration

From **exp_021** onward, publication-profile runs require an OSF pre-registration link in `hypothesis.md` (see [`research_agenda.md`](research_agenda.md)). CI `ci` profile runs are wiring-only and exempt.

---

## Citation

When reusing this codebase or MicroQML Bench exports, cite [CITATION.cff](../CITATION.cff) and the Zenodo DOI when available ([zenodo.md](zenodo.md)).
