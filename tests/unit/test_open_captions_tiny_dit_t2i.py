"""Unit tests for caption pack loader and TinyDiT text conditioning."""

import torch

from src.classical.tiny_dit import HashCaptionEmbedder, TinyDiT
from src.data.open_captions import is_flickr8k_ready, load_flickr8k_batch


def test_hash_caption_embedder_shape():
    emb = HashCaptionEmbedder(dim=32, n_buckets=256)
    out = emb(["a dog runs", "", "two cats"])
    assert out.shape == (3, 32)


def test_tiny_dit_text_conditioning_forward():
    model = TinyDiT(dim=32, depth=2, n_heads=4, time_dim=64, text_dim=32, coupling="classical")
    emb = HashCaptionEmbedder(dim=32, n_buckets=256)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,))
    text = emb(["hello world", "null"])
    out = model(x, t, text)
    assert out.shape == x.shape


def test_flickr8k_loader_smoke():
    if not is_flickr8k_ready():
        return
    batch = load_flickr8k_batch("train", n_take=4, img_size=32, seed=0)
    assert batch["images"].shape == (4, 3, 32, 32)
    assert len(batch["captions"]) == 4
