# Open image packs — generation log

**Updated:** 2026-07-14  
**Scripts:** `scripts/download_open_images.py` · `scripts/download_open_captions.py` · `scripts/build_open_caption_splits.py`

## License / source matrix

| Pack | Source | License notes |
|------|--------|---------------|
| cifar10 | Toronto CIFAR / torchvision | Research use; cite Krizhevsky 2009 |
| fashion_mnist | Zalando Research / torchvision | MIT |
| flowers102 | Oxford VGG / torchvision | Research use; cite Nilsback & Zisserman 2008 |
| flickr8k | jbrownlee Datasets mirror (Hodosh et al.) | Research use for captions; G-T3 P0 |
| pokemon-blip | HuggingFace `lambda/pokemon-blip-captions` | **Unavailable** (gated / DMCA) — do not use |

## Downloads

- `cifar10` / `fashion_mnist` / `flowers102` — P0 I2I packs under `*/raw/v1/`
- `flickr8k` — G-T3 caption pack: 8091 jpg + official token/split lists; processed `pairs.parquet` 6000/1000/1000

## Protocol

- Raw blobs under `*/raw/v1/` — gitignored
- Train/val/test **split before** normalize/resize in loaders
- Flickr8k uses official `Flickr_8k.{train,dev,test}Images.txt` assignments before any resize
