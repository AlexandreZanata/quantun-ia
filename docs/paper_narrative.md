# Primary Paper Narrative (Option C + Four-Anchor v1.2)

**Status:** Locked for v1.2 submission (Phase 6).  
**Paper entry:** `paper/main.tex`  
**Audience:** Workshop / arXiv (cs.LG + quant-ph)

---

## Headline claim

Holdout-fair comparison of hybrid quantum–classical classifiers on synthetic and real
benchmarks, extended with a **four-anchor cross-domain panel** (physics, clinical,
software, agro-climate), multi-seed Wilcoxon tests, bootstrap CIs, and honestly
reported negative results.

---

## In-scope experiments (cite in main paper)

| ID | Role |
|----|------|
| exp_001, exp_008 | Synthetic quantum vs classical baseline |
| exp_002 | Hybrid architecture |
| exp_003, exp_009 | Entanglement ablations (negative) |
| exp_005, exp_007 | Curriculum / self-play (negative) |
| exp_011–exp_014 | Real-data suite (UCI, MNIST PCA, augmentation, sequence) |
| exp_017 | Poisoning × topology |
| exp_021 | Backend parity (reproducibility pillar) |
| exp_022 | Nano parity at matched parameters (inconclusive) |
| exp_032, exp_058 | C1 HIGGS nano + conventional sweep |
| exp_060, exp_061, exp_062 | C4 ACYD nano + conventional + QNN head |
| exp_068 | Grand comparison synthesis (C1–C4) |
| exp_069–exp_077 | C2/C3 anchors + conventional sweeps |
| exp_075 | GV-ALR efficiency (NIHR replication) |
| exp_078 | Agro Risk Lab human validation (Spearman ρ ≥ 0.85) |

---

## Deferred to follow-up papers (do not headline in v1)

| ID | Track | Doc |
|----|-------|-----|
| exp_015 | Gradient-variance adaptive LR | [method_adaptive_lr.md](method_adaptive_lr.md) |
| exp_016 | Hybrid NAS | `experiments/exp_016_hybrid_nas/` |
| exp_018 | Feature fusion | `experiments/exp_018_feature_fusion/` |
| exp_080 | Quantum champion fusion | Closed ❌ — honest negative (fusion −0.62 pp vs best hybrid) |

---

## Publication pipeline (Phase 6)

```bash
make citation-ready          # version alignment (skip-release OK pre-Zenodo)
make exp-068-publication     # regenerate grand_comparison.tex + JSON
make paper-artifacts         # latex-tables + figures + sync
make paper-build             # paper/main.pdf
make citation-ready-full     # after make release (MANIFEST checksums)
```

---

## Enforcement

Contract tests:

- `tests/contracts/test_paper_narrative.py` — LaTeX sections respect scope above
- `tests/contracts/test_results_md_uniform.py` — statistical sections for exp_011–018

---

## Related

- [research_agenda.md](research_agenda.md) — 12-month public roadmap
- [negative_results.md](negative_results.md) — documented failures
- [citation_loop.md](citation_loop.md) — DOI + arXiv closure
