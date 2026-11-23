"""Signed session tokens issued by the API after any provider has vouched for a user.

Format: ``base64url(json payload) "." base64url(HMAC-SHA256(payload))``. The payload holds
``sub`` (user id), ``iat`` and ``exp``. Because the API issues its own tokens, sign-in providers
only decide *who* the user is; the rest of the app never changes when a provider is added.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(secret: str, body: str) -> str:
    return _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())


def issue(secret: str, subject: str, ttl_s: int) -> str:
    now = int(time.time())
    body = _b64(json.dumps({"sub": subject, "iat": now, "exp": now + ttl_s}).encode())
    return f"{body}.{_sign(secret, body)}"


def verify(secret: str, token: str) -> dict | None:
    """The payload when the signature checks out and the token has not expired, else None."""
    try:
        body, sig = token.split(".")
        if not hmac.compare_digest(sig.encode("utf-8"), _sign(secret, body).encode("utf-8")):
            return None
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("exp", 0) < time.time():
        return None
    return payload
