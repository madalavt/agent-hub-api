"""Tenant users, roles and permissions (v1 ``/api/roots-auth/tenant/{tenant}/...``).

A tenant is a customer. Every tenant has the ``Update`` permission ("ability to update") and two
roles, like v1: ``ROOTS_INTERNAL_ADMIN`` (Update) and ``ANNOTATOR`` (none); more roles can be
created. A member (``user_customers`` row) has one role per tenant; roles with ``Update`` may
change the tenant's projects and trees and manage its users and roles (see ``app.access``).

Reads need access to the tenant; changes need ``Update`` there (or the global admin role).
Global admins see every tenant without being members; they are listed with
``is_platform_admin``.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import access
from app.auth.dependencies import PrincipalDep, SessionDep, get_principal
from app.auth.passwords import hash_password
from app.models import (
    Customer,
    TenantPermission,
    TenantProject,
    TenantRole,
    TenantRolePermission,
    User,
    UserCustomer,
)

UPDATE = "Update"
ADMIN_ROLE, ANNOTATOR_ROLE = "ROOTS_INTERNAL_ADMIN", "ANNOTATOR"
BUILT_IN_ROLES = (ADMIN_ROLE, ANNOTATOR_ROLE)

router = APIRouter(
    prefix="/v1/tenant/{tenant_id}", tags=["tenant roles"], dependencies=[Depends(get_principal)]
)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


# ------------------------------------------------------------------ defaults


def role_permissions(session: Session, role_ids: list[str]) -> dict[str, set[str]]:
    """Permission names of each role."""
    if not role_ids:
        return {}
    rows = session.execute(
        select(TenantRolePermission.role_id, TenantPermission.name)
        .join(TenantPermission, TenantPermission.id == TenantRolePermission.permission_id)
        .where(TenantRolePermission.role_id.in_(role_ids))
    ).all()
    out: dict[str, set[str]] = {r: set() for r in role_ids}
    for role_id, name in rows:
        out[role_id].add(name)
    return out


def ensure_tenant_defaults(session: Session, tenant_id: str, actor: str = "system") -> None:
    """v1's defaults for a tenant (idempotent): the Update permission, the two built-in roles,
    and a role for every member without one (global editors -> ROOTS_INTERNAL_ADMIN, others ->
    ANNOTATOR, so nobody's rights change)."""
    perm = session.scalar(
        select(TenantPermission).where(
            TenantPermission.tenant_id == tenant_id, TenantPermission.name == UPDATE
        )
    )
    if perm is None:
        try:
            with session.begin_nested():
                perm = TenantPermission(
                    tenant_id=tenant_id, name=UPDATE, description="ability to update", created_by=actor
                )
                session.add(perm)
                session.flush()
        except IntegrityError:
            perm = session.scalar(
                select(TenantPermission).where(
                    TenantPermission.tenant_id == tenant_id, TenantPermission.name == UPDATE
                )
            )
    roles = {
        r.role_name: r
        for r in session.scalars(select(TenantRole).where(TenantRole.tenant_id == tenant_id))
    }
    for name in BUILT_IN_ROLES:
        if name not in roles:
            try:
                with session.begin_nested():
                    role = TenantRole(tenant_id=tenant_id, role_name=name, created_by=actor)
                    session.add(role)
                    session.flush()
                    if name == ADMIN_ROLE and perm is not None:
                        session.add(TenantRolePermission(role_id=role.id, permission_id=perm.id))
                        session.flush()
                    roles[name] = role
            except IntegrityError:
                existing = session.scalar(
                    select(TenantRole).where(
                        TenantRole.tenant_id == tenant_id, TenantRole.role_name == name
                    )
                )
                if existing is not None:
                    roles[name] = existing
    unassigned = session.execute(
        select(UserCustomer, User.role)
        .join(User, User.id == UserCustomer.user_id)
        .where(UserCustomer.customer_id == tenant_id, UserCustomer.tenant_role_id.is_(None))
    ).all()
    for membership, global_role in unassigned:
        membership.tenant_role_id = roles[
            ADMIN_ROLE if global_role in ("admin", "editor") else ANNOTATOR_ROLE
        ].id
    session.flush()


def ensure_all_tenants(session: Session) -> None:
    for tenant_id in session.scalars(select(Customer.id)):
        ensure_tenant_defaults(session, tenant_id)


def sync_global_role(session: Session, user: User) -> None:
    """Keep a non-admin's global role in line with their tenant roles (editor when any of them
    has Update), for the few global actions that look at it."""
    if user.role == "admin":
        return
    rows = list(
        session.scalars(
            select(UserCustomer.tenant_role_id).where(UserCustomer.user_id == user.id)
        )
    )
    if user.role == "editor" and any(r is None for r in rows):
        return  # keep fallback editor access on uninitialized memberships
    role_ids = [r for r in rows if r]
    perms = role_permissions(session, role_ids)
    user.role = "editor" if any(UPDATE in p for p in perms.values()) else "viewer"


# ------------------------------------------------------------------ helpers


def _tenant(session: Session, principal, tenant_id: str, manage: bool = False) -> Customer:
    access.read_customer(principal, tenant_id)
    tenant = session.get(Customer, tenant_id)
    if tenant is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"tenant {tenant_id} not found")
    if manage and not principal.can_write_customer(tenant_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "managing users and roles needs the Update permission"
        )
    ensure_tenant_defaults(session, tenant_id)
    return tenant


def _role(session: Session, tenant_id: str, role_id: str | None, role_name: str | None):
    q = select(TenantRole).where(TenantRole.tenant_id == tenant_id)
    if role_id:
        q = q.where(TenantRole.id == role_id)
    elif role_name:
        q = q.where(TenantRole.role_name == role_name)
    else:
        raise HTTPException(422, "tenant_role_id or role_name is required")
    role = session.scalar(q)
    if role is None:
        raise HTTPException(404, f"role {role_id or role_name} not found in this tenant")
    return role


def _permissions_by(session: Session, tenant_id: str, ids: list | None, names: list | None):
    if ids is None and names is None:
        return None
    perms = list(
        session.scalars(select(TenantPermission).where(TenantPermission.tenant_id == tenant_id))
    )
    wanted = [p for p in perms if (ids and p.id in ids) or (names and p.name in names)]
    unknown = set(ids or []) - {p.id for p in perms} | set(names or []) - {p.name for p in perms}
    if unknown:
        raise HTTPException(422, f"unknown permission(s): {sorted(unknown)}")
    return wanted


def _set_role_permissions(session: Session, role: TenantRole, perms: list, actor: str) -> None:
    session.execute(delete(TenantRolePermission).where(TenantRolePermission.role_id == role.id))
    session.add_all(
        TenantRolePermission(role_id=role.id, permission_id=p.id, created_by=actor) for p in perms
    )


def _role_out(role: TenantRole, users: int, perms: set[str]) -> dict[str, Any]:
    return {
        "tenant_role_id": role.id,
        "tenant_id": role.tenant_id,
        "role_name": role.role_name,
        "created_at": _iso(role.created_at),
        "created_by": role.created_by,
        "updated_at": _iso(role.updated_at),
        "updated_by": role.updated_by,
        # Agent Hub: members with the role, its permissions, whether it can be deleted
        "users_count": users,
        "permissions": sorted(perms),
        "built_in": role.role_name in BUILT_IN_ROLES,
    }


def _names(user: User) -> tuple[str, str]:
    first, _, last = (user.name or "").strip().partition(" ")
    return first, last


def _member_out(
    tenant_id: str, user: User, membership: UserCustomer | None, role: TenantRole | None
) -> dict[str, Any]:
    first, last = _names(user)
    return {
        "tenant_user_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-hub/{tenant_id}/{user.id}")),
        "tenant_id": tenant_id,
        "platform_user_id": user.id,
        "display_name": user.name,
        "first_name": first,
        "last_name": last,
        "avatar_url": None,
        "created_at": _iso((membership and membership.created_at) or user.created_at),
        "created_by": None,
        "updated_at": None,
        "updated_by": None,
        "last_active_time": _iso(user.last_active_at),
        "deleted_at": None,
        "deleted_by": None,
        "role_name": role.role_name if role else None,
        "email": user.email,
        # Agent Hub
        "tenant_role_id": role.id if role else None,
        "username": user.username,
        "active": user.active,
        "is_platform_admin": user.role == "admin",
    }


# ------------------------------------------------------------------ v1 reads


@router.get("/roles", summary="Roles of the tenant (with member counts)")
def list_roles(tenant_id: str, principal: PrincipalDep, session: SessionDep) -> list[dict]:
    _tenant(session, principal, tenant_id)
    roles = list(
        session.scalars(
            select(TenantRole)
            .where(TenantRole.tenant_id == tenant_id)
            .order_by(TenantRole.role_name)
        )
    )
    counts = dict(
        session.execute(
            select(UserCustomer.tenant_role_id, func.count())
            .where(UserCustomer.customer_id == tenant_id)
            .group_by(UserCustomer.tenant_role_id)
        ).all()
    )
    perms = role_permissions(session, [r.id for r in roles])
    return [_role_out(r, counts.get(r.id, 0), perms.get(r.id, set())) for r in roles]


@router.get("/role", summary="Roles with their permissions (one row per role and permission)")
def roles_with_permissions(
    tenant_id: str, principal: PrincipalDep, session: SessionDep
) -> list[dict]:
    _tenant(session, principal, tenant_id)
    roles = list(session.scalars(select(TenantRole).where(TenantRole.tenant_id == tenant_id)))
    perms = {
        p.id: p
        for p in session.scalars(
            select(TenantPermission).where(TenantPermission.tenant_id == tenant_id)
        )
    }
    links = session.execute(
        select(TenantRolePermission.role_id, TenantRolePermission.permission_id).where(
            TenantRolePermission.role_id.in_([r.id for r in roles])
        )
    ).all()
    by_role: dict[str, list] = {}
    for role_id, perm_id in links:
        by_role.setdefault(role_id, []).append(perms[perm_id])
    out = []
    for r in sorted(roles, key=lambda r: r.role_name):
        for p in by_role.get(r.id) or [None]:
            out.append(
                {
                    "tenant_id": tenant_id,
                    "tenant_role_id": r.id,
                    "role_name": r.role_name,
                    "tenant_permission_id": p.id if p else None,
                    "tenant_permission_name": p.name if p else None,
                    "permission_description": p.description if p else None,
                    "created_at": _iso(r.created_at),
                    "created_by": r.created_by,
                    "updated_at": _iso(r.updated_at),
                    "updated_by": r.updated_by,
                }
            )
    return out


@router.get("/permission", summary="Permissions roles can have")
def list_permissions(tenant_id: str, principal: PrincipalDep, session: SessionDep) -> dict:
    _tenant(session, principal, tenant_id)
    perms = session.scalars(
        select(TenantPermission)
        .where(TenantPermission.tenant_id == tenant_id)
        .order_by(TenantPermission.name)
    )
    return {
        "data": [
            {
                "tenant_permission_id": p.id,
                "tenant_permission_name": p.name,
                "permission_description": p.description,
                "permission_group": p.group,
                "created_at": _iso(p.created_at),
                "created_by": p.created_by,
                "updated_at": _iso(p.updated_at),
                "updated_by": p.updated_by,
                "tenant_id": p.tenant_id,
            }
            for p in perms
        ],
        "status": 200,
        "statusText": "OK",
    }


@router.get("/users/roles", summary="Members of the tenant with their roles")
def list_members(tenant_id: str, principal: PrincipalDep, session: SessionDep) -> list[dict]:
    _tenant(session, principal, tenant_id)
    rows = session.execute(
        select(User, UserCustomer, TenantRole)
        .join(UserCustomer, UserCustomer.user_id == User.id)
        .outerjoin(TenantRole, TenantRole.id == UserCustomer.tenant_role_id)
        .where(UserCustomer.customer_id == tenant_id)
    ).all()
    out = [_member_out(tenant_id, u, m, r) for u, m, r in rows]
    members = {u.id for u, _, _ in rows}
    admins = session.scalars(select(User).where(User.role == "admin", User.active.is_(True)))
    out += [_member_out(tenant_id, u, None, None) for u in admins if u.id not in members]
    return sorted(out, key=lambda m: (m["display_name"] or "").lower())


def _project_out(p: TenantProject, members: int) -> dict[str, Any]:
    """v1 roots-auth project, plus the project's work item queue (v1 keeps it server-side)."""
    return {
        "project_id": p.id,
        "tenant_id": p.customer_id,
        "display_name": p.name,
        "description": p.description,
        "created_by": p.created_by,
        "created_at": _iso(p.created_at),
        "document_project_id": None,  # v1: projects are not tied to a document project
        "membersCount": members,
        "queue_id": p.queue_id,
    }


def _members(session: Session, tenant_id: str) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(UserCustomer)
            .where(UserCustomer.customer_id == tenant_id)
        )
        or 0
    )


@router.get("/projects", summary="Projects of the tenant (with member counts and queue)")
def list_tenant_projects(
    tenant_id: str, principal: PrincipalDep, session: SessionDep
) -> list[dict]:
    _tenant(session, principal, tenant_id)
    members = _members(session, tenant_id)
    projects = session.scalars(
        select(TenantProject)
        .where(TenantProject.customer_id == tenant_id)
        .order_by(TenantProject.name)
    )
    return [_project_out(p, members) for p in projects]


@router.get("/projects/{project_id}", summary="One project of the tenant")
def get_tenant_project(
    tenant_id: str, project_id: str, principal: PrincipalDep, session: SessionDep
) -> dict:
    _tenant(session, principal, tenant_id)
    return _project_out(
        _tenant_project(session, tenant_id, project_id), _members(session, tenant_id)
    )


# ------------------------------------------------------------------ changes


class RoleIn(BaseModel):
    role_name: str = Field(min_length=1, max_length=100)
    permission_ids: list[str] | None = None
    permissions: list[str] | None = None  # names


class RoleUpdate(BaseModel):
    role_name: str | None = Field(default=None, min_length=1, max_length=100)
    permission_ids: list[str] | None = None
    permissions: list[str] | None = None


class ProjectIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    queue_id: str | None = None


class ProjectUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    queue_id: str | None = None


def _queue_id(value: str | None) -> str | None:
    """Work item manager queue ids: 32 lowercase hex chars (dashes and case ignored)."""
    text = (value or "").replace("-", "").strip().lower()
    if not text:
        return None
    if len(text) > 32:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "queue_id is too long")
    return text


def _tenant_project(session: Session, tenant_id: str, project_id: str) -> TenantProject:
    p = session.get(TenantProject, project_id.lower())
    if p is None or p.customer_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"project {project_id} not found")
    return p


@router.post("/projects", status_code=201, summary="Create a project (with its queue id)")
def create_tenant_project(
    tenant_id: str, body: ProjectIn, principal: PrincipalDep, session: SessionDep
) -> dict:
    _tenant(session, principal, tenant_id, manage=True)
    name = body.display_name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "display_name cannot be blank")
    p = TenantProject(
        customer_id=tenant_id,
        name=name,
        description=body.description,
        queue_id=_queue_id(body.queue_id),
        created_by=principal.subject,
    )
    session.add(p)
    session.flush()
    return _project_out(p, _members(session, tenant_id))


@router.put("/projects/{project_id}", summary="Rename a project or change its queue")
def update_tenant_project(
    tenant_id: str,
    project_id: str,
    body: ProjectUpdate,
    principal: PrincipalDep,
    session: SessionDep,
) -> dict:
    _tenant(session, principal, tenant_id, manage=True)
    p = _tenant_project(session, tenant_id, project_id)
    if body.display_name is not None:
        name = body.display_name.strip()
        if not name:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "display_name cannot be blank")
        p.name = name
    if "description" in body.model_fields_set:
        p.description = body.description
    if "queue_id" in body.model_fields_set:
        p.queue_id = _queue_id(body.queue_id)
    return _project_out(p, _members(session, tenant_id))


@router.delete("/projects/{project_id}", status_code=204, summary="Delete a project")
def delete_tenant_project(
    tenant_id: str, project_id: str, principal: PrincipalDep, session: SessionDep
):
    """Deletes the project only: its queue and work items stay in the work item manager."""
    _tenant(session, principal, tenant_id, manage=True)
    session.delete(_tenant_project(session, tenant_id, project_id))


@router.post("/roles", status_code=201, summary="Create a role")
def create_role(
    tenant_id: str, body: RoleIn, principal: PrincipalDep, session: SessionDep
) -> dict:
    _tenant(session, principal, tenant_id, manage=True)
    name = body.role_name.strip().upper().replace(" ", "_")
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "role_name cannot be blank")
    role = TenantRole(tenant_id=tenant_id, role_name=name, created_by=principal.name)
    session.add(role)
    try:
        session.flush()
    except IntegrityError as exc:
        raise HTTPException(409, f"role {name} already exists") from exc
    perms = _permissions_by(session, tenant_id, body.permission_ids, body.permissions) or []
    _set_role_permissions(session, role, perms, principal.name)
    session.flush()
    return _role_out(role, 0, {p.name for p in perms})


@router.put("/roles/{role_id}", summary="Rename a role or change its permissions")
def update_role(
    tenant_id: str, role_id: str, body: RoleUpdate, principal: PrincipalDep, session: SessionDep
) -> dict:
    _tenant(session, principal, tenant_id, manage=True)
    role = _role(session, tenant_id, role_id, None)
    if body.role_name is not None:
        new_name = body.role_name.strip().upper().replace(" ", "_")
        if not new_name:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "role_name cannot be blank")
        if role.role_name in BUILT_IN_ROLES and new_name != role.role_name:
            raise HTTPException(409, f"{role.role_name} is a built-in role and keeps its name")
        role.role_name = new_name
    perms = _permissions_by(session, tenant_id, body.permission_ids, body.permissions)
    if perms is not None:
        perm_names = {p.name for p in perms}
        if role.role_name == ADMIN_ROLE and UPDATE not in perm_names:
            raise HTTPException(409, f"{ADMIN_ROLE} must keep the {UPDATE} permission")
        if principal.role != "admin" and UPDATE not in perm_names:
            caller_membership = session.get(UserCustomer, (principal.subject, tenant_id))
            if caller_membership is not None and caller_membership.tenant_role_id == role.id:
                raise HTTPException(409, "you cannot remove your own Update permission")
        _set_role_permissions(session, role, perms, principal.name)
    role.updated_at, role.updated_by = _now(), principal.name
    try:
        session.flush()
    except IntegrityError as exc:
        raise HTTPException(409, f"role {role.role_name} already exists") from exc
    for user in session.scalars(
        select(User).join(UserCustomer).where(UserCustomer.tenant_role_id == role.id)
    ):
        sync_global_role(session, user)
    users = session.scalar(
        select(func.count())
        .select_from(UserCustomer)
        .where(UserCustomer.tenant_role_id == role.id)
    )
    return _role_out(role, users or 0, role_permissions(session, [role.id])[role.id])


@router.delete("/roles/{role_id}", status_code=204, summary="Delete a role nobody has")
def delete_role(tenant_id: str, role_id: str, principal: PrincipalDep, session: SessionDep):
    _tenant(session, principal, tenant_id, manage=True)
    role = _role(session, tenant_id, role_id, None)
    if role.role_name in BUILT_IN_ROLES:
        raise HTTPException(409, f"{role.role_name} is a built-in role")
    in_use = session.scalar(
        select(func.count())
        .select_from(UserCustomer)
        .where(UserCustomer.tenant_role_id == role.id)
    )
    if in_use:
        raise HTTPException(409, f"{in_use} user(s) have this role; give them another role first")
    session.delete(role)


class InviteIn(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    display_name: str | None = Field(default=None, max_length=255)
    first_name: str | None = Field(default=None, max_length=120)
    last_name: str | None = Field(default=None, max_length=120)
    tenant_role_id: str | None = None
    role_name: str | None = None
    username: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=1024)


class MemberUpdate(BaseModel):
    tenant_role_id: str | None = None
    role_name: str | None = None


@router.post("/users", status_code=201, summary="Invite a user to the tenant")
def invite_user(
    tenant_id: str, body: InviteIn, principal: PrincipalDep, session: SessionDep
) -> dict:
    """Adds an existing user (matched by email or username) to the tenant, or creates a local
    user. A generated password is returned once (``temporary_password``) when none is given."""
    _tenant(session, principal, tenant_id, manage=True)
    role = _role(session, tenant_id, body.tenant_role_id, body.role_name)
    email = body.email.strip()
    username = (body.username or email).strip()
    if not email or not username:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "email and username are required")
    user = session.scalar(
        select(User).where(
            (func.lower(User.email) == email.lower())
            | (func.lower(User.username) == username.lower())
        )
    )
    temporary = None
    if user is None:
        name = (
            body.display_name
            or " ".join(x for x in (body.first_name, body.last_name) if x)
            or email.split("@")[0]
        ).strip() or username
        password = body.password or (temporary := secrets.token_urlsafe(12))
        new_id = str(uuid.uuid4())
        user = User(
            id=new_id,
            provider="local",
            external_id=new_id,
            username=username,
            name=name,
            email=email,
            role="viewer",
            password_hash=hash_password(password),
        )
        session.add(user)
        try:
            session.flush()
        except IntegrityError as exc:
            raise HTTPException(409, "a user with that username already exists") from exc
    elif user.role == "admin":
        raise HTTPException(409, "platform admins already have every tenant")
    if (
        user.id == principal.subject
        and principal.role != "admin"
        and UPDATE not in role_permissions(session, [role.id])[role.id]
    ):
        raise HTTPException(409, "you cannot remove your own Update permission")
    membership = session.get(UserCustomer, (user.id, tenant_id))
    if membership is None:
        membership = UserCustomer(user_id=user.id, customer_id=tenant_id)
        session.add(membership)
    membership.tenant_role_id = role.id
    session.flush()
    sync_global_role(session, user)
    session.flush()
    out = _member_out(tenant_id, user, membership, role)
    if temporary:
        out["temporary_password"] = temporary
    return out


@router.put("/users/{user_id}", summary="Change a member's role")
def update_member(
    tenant_id: str,
    user_id: str,
    principal: PrincipalDep,
    session: SessionDep,
    body: Annotated[MemberUpdate, Body()],
) -> dict:
    _tenant(session, principal, tenant_id, manage=True)
    membership = session.get(UserCustomer, (user_id, tenant_id))
    user = session.get(User, user_id)
    if membership is None or user is None:
        raise HTTPException(404, f"user {user_id} is not a member of this tenant")
    role = _role(session, tenant_id, body.tenant_role_id, body.role_name)
    if (
        user_id == principal.subject
        and UPDATE not in role_permissions(session, [role.id])[role.id]
    ):
        raise HTTPException(409, "you cannot remove your own Update permission")
    membership.tenant_role_id = role.id
    session.flush()
    sync_global_role(session, user)
    session.flush()
    return _member_out(tenant_id, user, membership, role)


@router.delete("/users/{user_id}", status_code=204, summary="Remove a member from the tenant")
def remove_member(tenant_id: str, user_id: str, principal: PrincipalDep, session: SessionDep):
    _tenant(session, principal, tenant_id, manage=True)
    if user_id == principal.subject:
        raise HTTPException(409, "you cannot remove yourself")
    membership = session.get(UserCustomer, (user_id, tenant_id))
    if membership is None:
        raise HTTPException(404, f"user {user_id} is not a member of this tenant")
    session.delete(membership)
    session.flush()
    user = session.get(User, user_id)
    if user is not None:
        sync_global_role(session, user)
