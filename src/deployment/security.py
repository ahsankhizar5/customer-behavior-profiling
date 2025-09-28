"""Security utilities for deployment: AES-256 encryption and JWT management."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionManager:
    """Manage AES-256-GCM encryption for sensitive payloads."""

    def __init__(self, key: Optional[bytes] = None) -> None:
        if key is None:
            env_key = os.getenv("AES_KEY")
            if env_key:
                key = base64.urlsafe_b64decode(env_key)
        if key is None:
            key = AESGCM.generate_key(bit_length=256)
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key.")
        self.key = key
        self._aesgcm = AESGCM(self.key)

    @staticmethod
    def generate_key() -> str:
        """Generate a new AES-256 key encoded for environment storage."""

        key = AESGCM.generate_key(bit_length=256)
        return base64.urlsafe_b64encode(key).decode("utf-8")

    def encrypt(self, data: bytes, *, associated_data: Optional[bytes] = None) -> str:
        """Encrypt raw bytes and return a base64 string."""

        nonce = os.urandom(12)
        cipher = self._aesgcm.encrypt(nonce, data, associated_data)
        payload = nonce + cipher
        return base64.urlsafe_b64encode(payload).decode("utf-8")

    def decrypt(self, token: str, *, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt a payload encoded via :meth:`encrypt`."""

        data = base64.urlsafe_b64decode(token)
        nonce, cipher = data[:12], data[12:]
        return self._aesgcm.decrypt(nonce, cipher, associated_data)

    def encrypt_json(self, payload: Dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        return self.encrypt(data)

    def decrypt_json(self, token: str) -> Dict[str, Any]:
        data = self.decrypt(token)
        return json.loads(data.decode("utf-8"))


@dataclass
class JWTSettings:
    secret_key: str
    algorithm: str = "HS256"
    issuer: str = "fraud-api"
    audience: Optional[str] = None
    access_ttl: timedelta = timedelta(hours=1)


class JWTAuthManager:
    """Utility for issuing and validating JWT access tokens."""

    def __init__(self, settings: Optional[JWTSettings] = None) -> None:
        if settings is None:
            secret = os.getenv("JWT_SECRET")
            if not secret:
                raise RuntimeError("JWT_SECRET environment variable is required for JWTAuthManager.")
            settings = JWTSettings(secret_key=secret)
        self.settings = settings

    def create_token(
        self,
        subject: str,
        *,
        scopes: Optional[list[str]] = None,
        expires_delta: Optional[timedelta] = None,
        extra_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        ttl = expires_delta or self.settings.access_ttl
        payload: Dict[str, Any] = {
            "sub": subject,
            "iss": self.settings.issuer,
            "iat": int(now.timestamp()),
            "exp": int((now + ttl).timestamp()),
        }
        if self.settings.audience:
            payload["aud"] = self.settings.audience
        if scopes:
            payload["scopes"] = scopes
        if extra_claims:
            payload.update(extra_claims)
        return jwt.encode(payload, self.settings.secret_key, algorithm=self.settings.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            options = {"verify_aud": bool(self.settings.audience)}
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.algorithm],
                audience=self.settings.audience,
                issuer=self.settings.issuer,
                options=options,
            )
            return payload
        except JWTError as exc:  # pragma: no cover - defensive logging
            raise PermissionError("Invalid authentication token") from exc


__all__ = ["EncryptionManager", "JWTAuthManager", "JWTSettings"]
