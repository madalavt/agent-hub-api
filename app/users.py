"""Sign-in endpoints and user administration.

* ``GET  /auth/providers``   which sign-in methods are enabled (for the login screen)
* ``POST /auth/login``       password providers: username + password -> session token
* ``POST /auth/logout``      stateless: the client discards its token
* ``GET  /v1/me``            the signed-in user and role
* ``/v1/users``              admin only: list, create, change (role, customers, password), disable

Adding an external provider (Entra ID, OAuth/OIDC) means a redirect + callback route that
builds an ``Identity`` and calls ``complete_sign_in`` then ``_session_response``; nothing else
here changes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.access import ROLES, require_admin
from app.auth import tokens
from app.auth.dependencies import (
    PrincipalDep,
    SessionDep,
    SettingsDep,
    get_principal,
    session_secret,
)
from app.auth.passwords import hash_password
from app.auth.providers import SignInError, complete_sign_in, enabled_password_providers
from app.config import Settings
from app.models import Customer, TenantRole, User, UserCustomer
from app.schemas import LoginIn, TokenOut, UserIn, UserOut, UserUpdate

auth_router = APIRouter(prefix="/auth", tags=["auth"])
router = APIRouter(prefix="/v1", tags=["users"], dependencies=[Depends(get_principal)])

_INVALID = "invalid username or password"


def _user_out(session: Session, user: User) -> UserOut:
    from app.tenants import role_permissions  # noqa: PLC0415 (import cycle)

    memberships = session.execute(
        select(UserCustomer.customer_id, UserCustomer.tenant_role_id, TenantRole.role_name)
        .outerjoin(TenantRole, TenantRole.id == UserCustomer.tenant_role_id)
        .where(UserCustomer.user_id == user.id)
    ).all()
    perms = role_permissions(session, [r for _, r, _ in memberships if r])
    return UserOut(
        id=user.id,
        provider=user.provider,
        username=user.username,
        name=user.name,
        email=user.email,
        role=user.role,
        active=user.active,
        customer_ids=sorted(cid for cid, _, _ in memberships),
        tenants=[
            {
                "customer_id": cid,
                "tenant_role_id": role_id,
                "role_name": role_name,
                "permissions": sorted(perms.get(role_id, set())),
            }
            for cid, role_id, role_name in sorted(memberships)
        ],
    )


def _session_response(session: Session, settings: Settings, user: User) -> TokenOut:
    return TokenOut(
        access_token=tokens.issue(session_secret(settings), user.id, settings.session_ttl_s),
        expires_in=settings.session_ttl_s,
        user=_user_out(session, user),
    )


# ------------------------------------------------------------------ sign-in


@auth_router.get("/providers", summary="Enabled sign-in methods")
def providers(settings: SettingsDep) -> list[dict[str, str]]:
    return [{"name": p.name, "kind": "password"} for p in enabled_password_providers(settings)]


@auth_router.post("/login", response_model=TokenOut, summary="Sign in with username and password")
def login(body: LoginIn, session: SessionDep, settings: SettingsDep) -> TokenOut:
    for provider in enabled_password_providers(settings):
        identity = provider.verify_password(session, body.username.strip(), body.password)
        if identity is None:
            continue
        try:
            user = complete_sign_in(session, identity)
        except SignInError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        user.last_active_at = datetime.now(UTC)
        return _session_response(session, settings, user)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, _INVALID)


@auth_router.post("/logout", status_code=204, summary="Sign out (the client drops its token)")
def logout() -> Response:
    return Response(status_code=204)


# ------------------------------------------------------------------ current user


@router.get("/me", response_model=UserOut, summary="The signed-in user")
def me(principal: PrincipalDep, session: SessionDep) -> UserOut:
    if principal.role == "service":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "service keys have no user profile")
    user = session.get(User, principal.subject)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return _user_out(session, user)


# ------------------------------------------------------------------ user administration


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"role must be one of {ROLES}")


def _set_customers(
    session: Session, user: User, customer_ids: list[str], *, reset_roles: bool = False
) -> None:
    wanted = set(customer_ids)
    existing = (
        set(session.scalars(select(Customer.id).where(Customer.id.in_(wanted))))
        if wanted
        else set()
    )
    missing = wanted - existing
    if missing:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"unknown customer(s): {sorted(missing)}"
        )
    from app.tenants import ensure_tenant_defaults  # noqa: PLC0415 (import cycle)

    kept = (
        {}
        if reset_roles
        else {
            m.customer_id: m.tenant_role_id
            for m in session.scalars(select(UserCustomer).where(UserCustomer.user_id == user.id))
        }
    )
    session.execute(delete(UserCustomer).where(UserCustomer.user_id == user.id))
    session.add_all(
        UserCustomer(user_id=user.id, customer_id=cid, tenant_role_id=kept.get(cid))
        for cid in existing
    )
    session.flush()
    for cid in existing:  # new members get the role their global role implies
        ensure_tenant_defaults(session, cid)


def _sync_member_roles_from_global(session: Session, user: User) -> None:
    """When an admin changes a user's global role (editor <-> viewer), align existing tenant
    memberships to the matching built-in role so privilege changes actually take effect."""
    from app.tenants import ensure_tenant_defaults  # noqa: PLC0415 (import cycle)

    memberships = list(
        session.scalars(select(UserCustomer).where(UserCustomer.user_id == user.id))
    )
    if not memberships:
        return
    for m in memberships:
        m.tenant_role_id = None
    session.flush()
    for m in memberships:
        ensure_tenant_defaults(session, m.customer_id)


@router.get("/users", response_model=list[UserOut], summary="All users (admin)")
def list_users(principal: PrincipalDep, session: SessionDep) -> list[UserOut]:
    require_admin(principal)
    users = session.scalars(select(User).order_by(User.name))
    return [_user_out(session, u) for u in users]


@router.post(
    "/users", response_model=UserOut, status_code=201, summary="Create a local user (admin)"
)
def create_user(body: UserIn, principal: PrincipalDep, session: SessionDep) -> UserOut:
    require_admin(principal)
    _check_role(body.role)
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "name cannot be blank")
    new_id = str(uuid.uuid4())
    user = User(
        id=new_id,
        provider="local",
        external_id=new_id,  # local subject is the user id
        username=body.username,
        name=name,
        email=body.email,
        role=body.role,
        password_hash=hash_password(body.password),
    )
    session.add(user)
    try:
        session.flush()
    except IntegrityError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "that username is taken") from exc
    _set_customers(session, user, body.customer_ids)
    session.flush()
    return _user_out(session, user)


@router.put("/users/{user_id}", response_model=UserOut, summary="Change a user (admin)")
def update_user(
    user_id: str, body: UserUpdate, principal: PrincipalDep, session: SessionDep
) -> UserOut:
    require_admin(principal)
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"user {user_id} not found")
    data = body.model_dump(exclude_unset=True)
    role_changed = False
    if "role" in data and data["role"] is not None:
        _check_role(data["role"])
        if user.id == principal.subject and data["role"] != "admin":
            raise HTTPException(status.HTTP_409_CONFLICT, "you cannot remove your own admin role")
        role_changed = data["role"] != user.role
        user.role = data["role"]
    if "active" in data and data["active"] is not None:
        if user.id == principal.subject and not data["active"]:
            raise HTTPException(status.HTTP_409_CONFLICT, "you cannot disable your own account")
        user.active = data["active"]
    if "name" in data and data["name"] is not None:
        name = data["name"].strip()
        if not name:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "name cannot be blank")
        user.name = name
    if "email" in data:
        user.email = data["email"]
    if data.get("password"):
        if user.provider != "local":
            raise HTTPException(
                status.HTTP_409_CONFLICT, "passwords are managed by the sign-in provider"
            )
        user.password_hash = hash_password(data["password"])
    if data.get("customer_ids") is not None:
        _set_customers(session, user, data["customer_ids"], reset_roles=role_changed)
    elif role_changed:
        _sync_member_roles_from_global(session, user)
    session.flush()
    return _user_out(session, user)


@router.delete("/users/{user_id}", status_code=204, summary="Delete a user (admin)")
def delete_user(user_id: str, principal: PrincipalDep, session: SessionDep) -> None:
    require_admin(principal)
    if user_id == principal.subject:
        raise HTTPException(status.HTTP_409_CONFLICT, "you cannot delete your own account")
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"user {user_id} not found")
    session.delete(user)
