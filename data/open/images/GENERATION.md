# Open image packs — generation log

**Generated:** 2026-07-14T14:37:14.253368+00:00
**Script:** `scripts/download_open_images.py`

## License / source matrix (P0)

| Pack | Source | License notes |
|------|--------|---------------|
| cifar10 | Toronto CIFAR / torchvision | Research use; cite Krizhevsky 2009 |
| fashion_mnist | Zalando Research / torchvision | MIT |
| flowers102 | Oxford VGG / torchvision | Research use; cite Nilsback & Zisserman 2008 |

## Downloads

- `cifar10` → `/data/dev/projects/webstorm/quantun-ia/data/open/images/cifar10/raw/v1` (skipped=True)
- `fashion_mnist` → `/data/dev/projects/webstorm/quantun-ia/data/open/images/fashion_mnist/raw/v1` (skipped=True)
- `flowers102` → `/data/dev/projects/webstorm/quantun-ia/data/open/images/flowers102/raw/v1` (skipped=True)

## Protocol

- Raw blobs under `*/raw/v1/` — gitignored / DVC later
- Train/val/test **split before** normalize for experiment `run.py`
- Caption packs (Flickr8k, pokemon-blip) are Phase G-T3 — separate script
