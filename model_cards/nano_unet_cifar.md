# Model card — `nano_unet_cifar`

**Task:** Unconditional / I2I-style DDPM sampling at 32×32 (CIFAR-10 classical floor).  
**Origin:** Cycle v3 Phase H / K — `exp_102` NanoUNet DDPM (accepted vs noise FID).  
**Ship:** `make ship-nano-unet-cifar` → `dist/serve_models/nano_unet_cifar/`  
**API:** `POST /api/v1/predict/image` (`mode=i2i`) · `GET /api/v1/models/image/card`  
**Dashboard:** `dashboard/pages/08_image_nano_lab.py`

## Limits

- Not a text-to-image model (TinyDiT Flickr / exp_103 rejected).
- Research demo samples only — not production image generation.

## Reproduce

```bash
source .local/env.sh
make ship-nano-unet-cifar
make exp-112-publication
```
