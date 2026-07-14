# Hypothesis — EXP 103: TinyDiT T2I floor on Flickr8k (Phase H / H-T4)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v3 · Phase G-T3 (captions) + Phase H-T4 (classical T2I floor)

## What I expect to happen

On `flickr8k_captions_v1` (G-T3), a **caption-conditioned TinyDiT** DDPM will obtain
OpenCLIP **CLIPScore** strictly above Gaussian noise **and** at least **+0.5** above
an ablated **null-caption** control (empty string embeddings) on the same prompts.

## Why I expect this

- Roadmap H1 / H-T4: classical TinyDiT T2I floor before H-Q3.5 text–quantum fusion.
- Pokemon-blip is gated/DMCA; Flickr8k (jbrownlee mirror) is the open P0 caption pack.
- Official Flickr train/dev/test lists assign splits **before** resize/normalize.

## What would prove me wrong

- `CLIP_model ≤ CLIP_noise` or `CLIP_model < CLIP_null + 0.5`
- Caption ingest incomplete / missing images after extract
- OOM on 4060

## Metrics I will measure

- [x] G-T3 ingest: pairs.parquet + official split counts
- [x] CLIPScore (OpenCLIP ViT-B/32 ×100) for model, noise, null-caption
- [x] Δ_null = CLIP_model − CLIP_null; Δ_noise = CLIP_model − CLIP_noise
- [x] Params; wall-clock; device

## Success criteria

- **Primary (H1):** `CLIP_model ≥ CLIP_noise` **and** `CLIP_model ≥ CLIP_null + 0.5`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Images resized to 32×32 after split assignment (nano floor, not SOTA FID)
- Caption embedder is bag-of-hash (not CLIP text tower inside the generator)
- CLIPScore uses OpenCLIP ViT-B/32 (`openai` weights)
