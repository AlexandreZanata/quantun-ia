"""Fast unit smoke for image nano ship helpers (no publication train)."""

from pathlib import Path

from src.application.image_nano_ship import is_nano_unet_shipped, nano_unet_serve_dir


def test_nano_unet_serve_paths():
    root = Path(".")
    assert nano_unet_serve_dir(root).name == "nano_unet_cifar"
    # May or may not be shipped in CI — only assert boolean
    assert isinstance(is_nano_unet_shipped(root), bool)
