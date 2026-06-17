"""Unit tests for JWT auth service."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.infrastructure.auth import jwt_keys, jwt_service


@pytest.fixture(autouse=True)
def rsa_keys(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    jwt_keys.load_signing_key.cache_clear()
    jwt_keys.load_public_key.cache_clear()
    monkeypatch.setenv("JWT_PRIVATE_KEY_PEM", private_pem)
    monkeypatch.setenv("JWT_PUBLIC_KEY_PEM", public_pem)
    yield


def test_create_and_verify_access_token():
    token = jwt_service.create_access_token(tenant_id="tenant-a", user_id="user-1")
    claims = jwt_service.verify_access_token(token)
    assert claims.tenant_id == "tenant-a"
    assert claims.user_id == "user-1"


def test_verify_rejects_malformed_token():
    with pytest.raises(jwt_service.JwtAuthError, match="Invalid access token"):
        jwt_service.verify_access_token("not-a-jwt")
