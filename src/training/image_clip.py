"""OpenCLIP image–text score helper for T2I gates (Phase H / H-T4)."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=2)
def _load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_score(
    images: torch.Tensor,
    captions: list[str],
    *,
    device: torch.device,
    batch_size: int = 32,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> float:
    """
    Mean OpenCLIP cosine similarity × 100 (CLIPScore-style).

    ``images`` are NCHW in [-1, 1]. Empty captions score as null text.
    """
    if images.shape[0] != len(captions):
        raise ValueError("images/captions length mismatch")
    model, preprocess, tokenizer = _load_clip(model_name, pretrained)
    model = model.to(device)

    from torchvision.transforms.functional import to_pil_image

    scores: list[float] = []
    for start in range(0, images.shape[0], batch_size):
        chunk = images[start : start + batch_size]
        caps = captions[start : start + batch_size]
        pil_batch = []
        for img in chunk:
            rgb = ((img.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
            pil_batch.append(preprocess(to_pil_image(rgb)))
        image_input = torch.stack(pil_batch, dim=0).to(device)
        text_input = tokenizer(caps).to(device)
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        sim = (image_features * text_features).sum(dim=-1) * 100.0
        scores.extend(float(s) for s in sim.detach().cpu())
    return float(sum(scores) / max(len(scores), 1))
