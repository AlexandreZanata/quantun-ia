# Results — EXP 114: NanoUNet-v2 efficient I2I (2026-07-14)

**Verdict:** Confirmed
**Profile:** `publication` · **Device:** `cuda`
**Pack:** `tiny_imagenet` @ 64×64
**Params:** 3,016,035

## Metrics

| Metric | Value |
|--------|-------|
| Final train noise-MSE | 0.1012 |
| Val denoise MSE | 0.102134 |
| FID-R18 (model vs val) | 345.28 |
| FID-R18 (noise null vs val) | 632.61 |
| Relative FID improvement | 0.454 |
| LPIPS-proxy (VGG) | 11.1643 |
| Peak VRAM (MB) | 4242.9 |
| Train imgs/s | 350.82 |
| Elapsed (s) | 629.6 |

## Efficiency card (M-T7)

- Params: **3,016,035**
- Peak CUDA memory: **4242.9 MB**
- Train throughput: **350.82 imgs/s**
- Card complete: **True**

## Gate (Phase M / M0)

- Relative FID ≥ 0.20 vs noise **and** (FID ≤ 138.5 vs exp_102 or efficiency card).
- Absolute FID vs exp_102 target: **False** (model=345.28, target≤138.5).
- Outcome: **Confirmed** (Δ_rel = 0.454).

## Ablation suggestion

- What if you train the same NanoUNet-v2 on STL-10@64 vs Tiny-IN only?

*Logged via ExperimentLogger · 2026-07-14T15:40:10.940744*
