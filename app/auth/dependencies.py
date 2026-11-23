"""FastAPI dependency that turns a request into a ``Principal``.

Two ways in, both generic:

* ``X-API-Key: <API_KEY>``            internal services (extraction, annotations, worker)
* ``Authorization: Bearer <token>``   a signed session token issued by ``POST /auth/login``
                                      (or by an external provider's callback later)
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.access import SERVICE, Principal, principal_for_user
from app.auth import tokens
from app.config import Settings, get_settings
from app.db import get_session
from app.db_read import get_read_session
from app.models import User

# scope="function": the session commits when the endpoint returns, before the response is sent,
# so a caller never gets ids of rows that are not committed yet (e.g. add an item, then claim it).
SessionDep = Annotated[Session, Depends(get_session, scope="function")]
# Lag-tolerant reads (lists): the read replica when DB_READ_URL is set.
ReadSessionDep = Annotated[Session, Depends(get_read_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# Used when SESSION_SECRET is not configured: tokens then die with the process. Fine for a
# throwaway local run; set SESSION_SECRET anywhere that is shared or restarted.
_EPHEMERAL_SECRET = secrets.token_urlsafe(32)


def session_secret(settings: Settings) -> str:
    return settings.session_secret or _EPHEMERAL_SECRET


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


def get_principal(
    session: SessionDep,
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    if settings.api_key and x_api_key and secrets.compare_digest(
        x_api_key.encode("utf-8"), settings.api_key.encode("utf-8")
    ):
        return SERVICE
    if authorization and authorization.lower().startswith("bearer "):
        payload = tokens.verify(session_secret(settings), authorization[7:].strip())
        if payload is None:
            raise _unauthorized("session expired or invalid; sign in again")
        user = session.get(User, payload.get("sub", ""))
        if user is None or not user.active:
            raise _unauthorized("account not found or disabled")
        return principal_for_user(session, user)
    raise _unauthorized("sign in required")


PrincipalDep = Annotated[Principal, Depends(get_principal)]
