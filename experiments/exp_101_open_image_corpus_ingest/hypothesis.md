# Hypothesis — EXP 101: Open image corpus ingest verification (Phase G)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** CPU synthesis (verifies RTX 4060 workstation downloads; no model training)  
**Cycle:** Research v3 · Phase G

## What I expect to happen

All three P0 packs (`cifar10`, `fashion_mnist`, `flowers102`) are on disk with
`.download_complete` markers, processed `stats.json` + `split_indices.npz` written
**before** any normalize/resize, and smoke loaders return correct tensor shapes.

## Why I expect this

- Phase G tooling already ships `scripts/download_open_images.py` and
  `scripts/build_open_image_splits.py`.
- Fashion-MNIST previously validated; CIFAR/Flowers complete the accept pack gate.

## Pre-registered gates

| Gate | Threshold |
|------|-----------|
| Packs complete | 3/3 `.download_complete` |
| Splits | `processed/v1/stats.json` + `split_indices.npz` per pack |
| Smoke load | 8 train + 8 test samples; shapes match pack contract |
| Manifest | image modality entries registered (`ready=true`) |

## Known limitations

- Caption packs (Flickr8k / pokemon-blip) deferred to G-T3 (T2I path).
- Large blobs gitignored — DVC optional later.
