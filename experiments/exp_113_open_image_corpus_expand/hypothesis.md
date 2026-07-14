# Hypothesis — EXP 113: Open image corpus expansion (Phase L)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (downloads + checksums; no model train)  
**Cycle:** Research v4 · Phase L

## What I expect to happen

At least **3 P0** Cycle-v4 image packs are ready with **split-before-resize** manifests:

1. `stl10` — torchvision STL-10 + processed splits  
2. `tiny_imagenet` — Tiny-ImageNet-200 + processed splits  
3. `coco_captions` — COCO 2017 caption micro ≤20k + `pairs.parquet`

Smoke loads (≥8 samples) succeed for class packs; COCO pairs parquet is readable.

## Why I expect this

- Roadmap Phase L gate: ≥3 P0 packs unlock Phase M efficient nano floors.  
- Prefer torchvision / COCO individual JPEGs to avoid multi-tens-of-GB zips.

## What would prove me wrong

- Fewer than 3 packs complete with splits + checksums  
- Smoke shapes fail  
- Missing license / manifest rows

## Success criteria

- **Primary (L-T7):** ≥3/3 P0 packs ready + smoke OK  
- `make check` green (ci smoke only)

## Known limitations

- AFHQ / LAION (G-T7) remain optional / gated  
- COCO micro is ≤20k train images (not full COCO)  
- Resize deferred to experiment loaders
