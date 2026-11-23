"""Who may do what. One place for the role rules; routes call these helpers.

Roles (stored on ``users.role``):

* ``admin``  every customer, customer/tree create and delete, ML models, user management.
* ``editor`` assigned customers: edit their projects and the trees those customers use.
* ``viewer`` assigned customers: read only.
* ``service`` internal callers (extraction worker, annotations) with the service API key:
  read everything, write nothing.

Tenant roles (v1 roots-auth, ``app.tenants``): a member's role in a customer decides whether
they may change it: roles with the ``Update`` permission may (ROOTS_INTERNAL_ADMIN), others
may not (ANNOTATOR). A membership without a tenant role falls back to the global role.

Decision trees belong to an org's library (``trees.customer_id``), like v1: every member of the
org (anyone with access to one of its projects) can open all of its trees; members whose tenant
role has Update can change them. A tree another org's configuration uses is visible to that org
too. Trees with no org and used by no configuration are admin-only. Reads the user may not
see return 404 (so existence is not revealed); actions
the user may see but not perform return 403.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Project, Tree, User, UserCustomer

ROLES = ("admin", "editor", "viewer")


@dataclass(frozen=True)
class Principal:
    subject: str  # user id, or "service"
    name: str
    role: str  # admin | editor | viewer | service
    customer_ids: frozenset[str] | None  # None = every customer
    # Customers the user may change (tenant roles with Update); None = decided by ``role``.
    writable_customer_ids: frozenset[str] | None = None

    @property
    def reads_all(self) -> bool:
        return self.customer_ids is None

    def can_read_customer(self, customer_id: str) -> bool:
        return self.customer_ids is None or customer_id in self.customer_ids

    def can_write_customer(self, customer_id: str) -> bool:
        if self.role == "admin":
            return True
        if self.role == "service":
            return False
        if self.writable_customer_ids is not None:
            return customer_id in self.writable_customer_ids
        return self.role == "editor" and self.can_read_customer(customer_id)

    @property
    def writes_somewhere(self) -> bool:
        if self.writable_customer_ids is not None:
            return bool(self.writable_customer_ids)
        return self.role in ("admin", "editor")


SERVICE = Principal(subject="service", name="service", role="service", customer_ids=None)


def principal_for_user(session: Session, user: User) -> Principal:
    if user.role == "admin":
        return Principal(user.id, user.name, "admin", None)
    from app.tenants import UPDATE, role_permissions  # noqa: PLC0415 (import cycle)

    rows = session.execute(
        select(UserCustomer.customer_id, UserCustomer.tenant_role_id).where(
            UserCustomer.user_id == user.id
        )
    ).all()
    perms = role_permissions(session, [r for _, r in rows if r])
    writable = frozenset(
        cid
        for cid, role_id in rows
        if (UPDATE in perms.get(role_id, set()) if role_id else user.role == "editor")
    )
    return Principal(user.id, user.name, user.role, frozenset(c for c, _ in rows), writable)


def _not_found(what: str, id_: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"{what} {id_} not found")


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_403_FORBIDDEN, detail)


def require_admin(p: Principal) -> None:
    if p.role != "admin":
        raise _forbidden("this action needs the admin role")


def require_editor(p: Principal) -> None:
    if not p.writes_somewhere:
        raise _forbidden("this action needs the editor or admin role")


def read_customer(p: Principal, customer_id: str) -> None:
    if not p.can_read_customer(customer_id):
        raise _not_found("customer", customer_id)


def write_customer(p: Principal, customer_id: str) -> None:
    read_customer(p, customer_id)
    if not p.can_write_customer(customer_id):
        raise _forbidden("your role cannot change this customer's projects")


def visible_tree_ids(p: Principal):
    """Subquery of tree ids the principal may see (None = all trees)."""
    if p.reads_all:
        return None
    customers = p.customer_ids or []
    owned = select(Tree.id).where(Tree.customer_id.in_(customers))
    used = select(Project.tree_id).where(
        Project.customer_id.in_(customers), Project.tree_id.is_not(None)
    )
    return owned.union(used)


def tree_customer_ids(session: Session, tree_id: str) -> set[str]:
    """Orgs a tree belongs to: its owner and every org whose configuration uses it."""
    owner = session.scalar(select(Tree.customer_id).where(Tree.id == tree_id))
    users = session.scalars(select(Project.customer_id).where(Project.tree_id == tree_id))
    return {c for c in (owner, *users) if c}


def read_tree(session: Session, p: Principal, tree_id: str) -> None:
    if p.reads_all:
        return
    if not any(p.can_read_customer(c) for c in tree_customer_ids(session, tree_id)):
        raise _not_found("tree", tree_id)


def write_tree(session: Session, p: Principal, tree_id: str) -> None:
    read_tree(session, p, tree_id)
    if p.role == "admin":
        return
    if not any(p.can_write_customer(c) for c in tree_customer_ids(session, tree_id)):
        raise _forbidden("your role cannot edit this decision tree")


def check_tree_assignable(session: Session, p: Principal, tree_id: str) -> None:
    """Editors may use only trees of their orgs' libraries (owned or already used)."""
    if p.role == "admin":
        return
    try:
        read_tree(session, p, tree_id)
    except HTTPException as exc:
        raise _forbidden("you can only use decision trees of your orgs' libraries") from exc


def project_customer_id(session: Session, project_id: str) -> str | None:
    return session.scalar(select(Project.customer_id).where(Project.id == project_id))


def tree_exists(session: Session, tree_id: str) -> bool:
    return session.get(Tree, tree_id) is not None
