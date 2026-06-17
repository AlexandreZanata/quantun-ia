"""RSA key loading for JWT RS256 signing."""

from __future__ import annotations

import os
from functools import lru_cache

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def _generate_dev_keypair() -> tuple[bytes, bytes]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


@lru_cache(maxsize=1)
def load_signing_key() -> bytes:
    pem = os.environ.get("JWT_PRIVATE_KEY_PEM")
    if pem:
        return pem.encode("utf-8")
    private_pem, public_pem = _generate_dev_keypair()
    os.environ.setdefault("JWT_PUBLIC_KEY_PEM", public_pem.decode("utf-8"))
    return private_pem


@lru_cache(maxsize=1)
def load_public_key() -> bytes:
    pem = os.environ.get("JWT_PUBLIC_KEY_PEM")
    if pem:
        return pem.encode("utf-8")
    load_signing_key()
    pem = os.environ.get("JWT_PUBLIC_KEY_PEM")
    if not pem:
        raise RuntimeError("JWT public key unavailable")
    return pem.encode("utf-8")
