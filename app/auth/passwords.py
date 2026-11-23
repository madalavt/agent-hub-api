"""Password hashing for the local provider (stdlib scrypt, no extra dependency)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

_N, _R, _P, _DKLEN = 2**14, 8, 1, 32


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        scheme, salt, digest = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    candidate = hashlib.scrypt(
        password.encode(), salt=_unb64(salt), n=_N, r=_R, p=_P, dklen=_DKLEN
    )
    return hmac.compare_digest(candidate, _unb64(digest))
