"""Sign-in providers. A provider proves who someone is and returns an ``Identity``; the app then
maps that identity to a ``User`` (``provision_user``) and issues its own session token.

Adding a provider means:
  1. a class with ``name`` and a way to produce an ``Identity``
     * password-style (username + password): implement ``verify_password`` and it is offered
       by ``POST /auth/login``;
     * redirect-style (Entra ID, OAuth/OIDC): add a callback route that builds the ``Identity``
       from the verified token and calls ``complete_sign_in`` (see ``app/users.py``);
  2. its name in ``AUTH_PROVIDERS`` (settings) so it is enabled.

Only ``local`` exists today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.passwords import verify_password
from app.config import Settings
from app.models import User


@dataclass(frozen=True)
class Identity:
    provider: str  # which provider vouched for this person
    subject: str  # the provider's stable id for them (becomes User.external_id)
    name: str
    email: str | None = None


class PasswordProvider(Protocol):
    name: str

    def verify_password(
        self, session: Session, username: str, password: str
    ) -> Identity | None: ...


class LocalPasswordProvider:
    """Users created in this database, with a scrypt password hash."""

    name = "local"

    def verify_password(self, session: Session, username: str, password: str) -> Identity | None:
        user = session.scalar(
            select(User).where(
                User.provider == self.name,
                func.lower(User.username) == username.lower(),
            )
        )
        if user is None or not verify_password(password, user.password_hash):
            return None
        return Identity(self.name, user.external_id, user.name, user.email)


_PASSWORD_PROVIDERS: dict[str, type] = {"local": LocalPasswordProvider}


def enabled_password_providers(settings: Settings) -> list[PasswordProvider]:
    providers: list[PasswordProvider] = []
    for name in settings.auth_providers:
        if name not in _PASSWORD_PROVIDERS:
            raise RuntimeError(f"unknown or non-password auth provider {name!r}")
        providers.append(_PASSWORD_PROVIDERS[name]())
    return providers


class SignInError(Exception):
    """The identity is not allowed in (unknown local user, or the account is disabled)."""


def complete_sign_in(session: Session, identity: Identity, default_role: str = "viewer") -> User:
    """Find or create the User for a verified identity and refresh their name and email.

    Local users are created by an admin, so an unknown local identity is refused. External
    identities are created on first sign-in with ``default_role`` (an admin then assigns
    customers and a role).
    """
    user = session.scalar(
        select(User).where(
            User.provider == identity.provider, User.external_id == identity.subject
        )
    )
    if user is None:
        if identity.provider == "local":
            raise SignInError("unknown user")
        user = User(
            provider=identity.provider,
            external_id=identity.subject,
            name=identity.name,
            email=identity.email,
            role=default_role,
        )
        session.add(user)
    else:
        user.name = identity.name or user.name
        user.email = identity.email or user.email
    session.flush()
    if not user.active:
        raise SignInError("account disabled")
    return user
