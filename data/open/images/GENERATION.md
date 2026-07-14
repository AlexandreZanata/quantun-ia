# Open image packs — generation log

**Generated:** 2026-07-14T18:47:57.730229+00:00
**Script:** `scripts/download_open_images.py`

## License / source matrix (P0)

| Pack | Source | License notes |
|------|--------|---------------|
| cifar10 | Toronto CIFAR / torchvision | Research use; cite Krizhevsky 2009 |
| fashion_mnist | Zalando Research / torchvision | MIT |
| flowers102 | Oxford VGG / torchvision | Research use; cite Nilsback & Zisserman 2008 |
| stl10 | Stanford STL-10 / torchvision | Research use; cite Coates et al. 2011 |
| tiny_imagenet | Stanford CS231n Tiny-ImageNet-200 | Research use; ImageNet subset |
| coco_captions_micro | COCO 2017 captions ≤20k images | CC BY 4.0 annotations; images COCO terms |

## Downloads

- `stl10` → `/data/dev/projects/webstorm/quantun-ia/data/open/images/stl10/raw/v1` (skipped=False)
- `tiny_imagenet` → `/data/dev/projects/webstorm/quantun-ia/data/open/images/tiny_imagenet/raw/v1` (skipped=False)

## Protocol

- Raw blobs under `*/raw/v1/` — gitignored / DVC later
- Train/val/test **split before** normalize for experiment `run.py`
- Caption packs (Flickr8k, pokemon-blip) are Phase G-T3 — separate script
