"""Decision tree configuration schema, modeled on Platform-DecisionTree-API.

    customers ─< projects >─ trees ─┬─< nodes ─< fields (child_fields via parent_id)
                                    └─< branches (parent_node -> child_node, condition)
    ml_models  (referenced by nodes; may name a fallback model)

A *project* plays the role of the platform's *configuration*: it ties a customer to the tree that
runs on that customer's documents. Trees are reusable across projects.

ML model ``parameters`` never hold secrets. Credentials live in the extraction service's
``config/credentials.yaml``; a model only says which provider/model it wants.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    select,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    projects: Mapped[list[Project]] = relationship(
        back_populates="customer", cascade="all, delete-orphan", order_by="Project.created_at"
    )


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (Index("uq_projects_platform_id", "platform_id", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Integer project id, like platform v1's, for automation projects whose config holds one
    # (PLATFORM_DOC_PROJECT_ID). Assigned on insert (Postgres: sequence projects_platform_id_seq,
    # also the column default, so rows inserted straight into the database get one).
    platform_id: Mapped[int | None] = mapped_column(Integer)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    tree_id: Mapped[str | None] = mapped_column(ForeignKey("trees.id", ondelete="SET NULL"))
    # Platform configuration flags (split_package, ocr_page_limit, ...), passed to extraction.
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    customer: Mapped[Customer] = relationship(back_populates="projects")
    tree: Mapped[Tree | None] = relationship()


class TenantProject(Base):
    """A tenant's project, like v1's roots-auth project: what the UI calls a project (work items,
    documents). Each project has its own work item queue (``queue_id``, in the work item
    manager). A project is not tied to a decision tree: automations upload a work item's file
    to a tree configuration (``Project``, by its integer id) and link the package to the item.
    """

    __tablename__ = "tenant_projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    # The work item manager queue of this project (32 hex chars); None until one is assigned.
    queue_id: Mapped[str | None] = mapped_column(String(32), index=True)
    created_by: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MlModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    # Platform model class: gpt, azure_openai_reasoning, finetuned_streaming, vertex_ai, ...
    type: Mapped[str] = mapped_column(String(100))
    # model_type, model_name, max_completion_tokens, fallback_mlmodel_id, ... (no secrets)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    standard: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Tree(Base):
    __tablename__ = "trees"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # The org whose decision tree library holds the tree (every member can open it, like v1).
    # A tree is not tied to a project: automations pick the configuration they upload to.
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    business_context: Mapped[str | None] = mapped_column(Text)
    starting_node_id: Mapped[str | None] = mapped_column(String(36))
    # Tree-level parameters, e.g. {"postprocessing": [{"name": "compute_math"}]}
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # UI canvas positions: {"positions": {node_id: {"x": .., "y": ..}}}
    layout: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=0)  # bumped on every save
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    nodes: Mapped[list[Node]] = relationship(
        back_populates="tree", cascade="all, delete-orphan", order_by="Node.sort_order"
    )
    branches: Mapped[list[Branch]] = relationship(
        back_populates="tree", cascade="all, delete-orphan", order_by="Branch.sort_order"
    )


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tree_id: Mapped[str] = mapped_column(ForeignKey("trees.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    # gpt_node (prompt -> JSON fields) or code_node (parameters.function picks the executor)
    type: Mapped[str] = mapped_column(String(50))
    description: Mapped[str | None] = mapped_column(Text)
    mlmodel_id: Mapped[str | None] = mapped_column(ForeignKey("ml_models.id", ondelete="SET NULL"))
    # prompt_template {system, human, ai}, function, table_key, prompt_format, domain_rules, ...
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    standard: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    tree: Mapped[Tree] = relationship(back_populates="nodes")
    mlmodel: Mapped[MlModel | None] = relationship()
    # Flat list of every field on the node; table columns point at their table via parent_id.
    fields: Mapped[list[Field]] = relationship(
        back_populates="node", cascade="all, delete-orphan", order_by="Field.sort_order"
    )


class Field(Base):
    __tablename__ = "fields"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(36))  # table field id
    name: Mapped[str] = mapped_column(String(255))
    customer_key: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    # string | number | currency | date | boolean | table
    field_type: Mapped[str] = mapped_column(String(50), default="string")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    standard: Mapped[bool] = mapped_column(Boolean, default=True)

    node: Mapped[Node] = relationship(back_populates="fields")


class Branch(Base):
    __tablename__ = "branches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tree_id: Mapped[str] = mapped_column(ForeignKey("trees.id", ondelete="CASCADE"), index=True)
    parent_node_id: Mapped[str] = mapped_column(String(36))
    child_node_id: Mapped[str] = mapped_column(String(36))
    name: Mapped[str | None] = mapped_column(String(255))
    # Empty = always. Otherwise e.g.  document_type == "Mortgage Change"   (see app/conditions.py)
    condition: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    tree: Mapped[Tree] = relationship(back_populates="branches")


# ------------------------------------------------------------------ users and access


class User(Base):
    """A person who can sign in. ``provider`` says which sign-in method vouches for them
    (``local`` = username and password here; ``entra`` / ``oauth`` later) and ``external_id``
    is that provider's stable subject. Local users use their own id as ``external_id``."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_users_provider_subject"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    provider: Mapped[str] = mapped_column(String(50), default="local")
    external_id: Mapped[str] = mapped_column(String(255))
    username: Mapped[str | None] = mapped_column(String(255), unique=True)  # local sign-in name
    password_hash: Mapped[str | None] = mapped_column(String(255))  # local provider only
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    # admin: every customer. editor: assigned customers, can change them. viewer: read-only.
    role: Mapped[str] = mapped_column(String(20), default="viewer")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class UserCustomer(Base):
    """Which customers (tenants) a (non-admin) user belongs to, and their role there. Admins
    do not need rows here."""

    __tablename__ = "user_customers"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), primary_key=True
    )
    # The user's role in this tenant (v1 tenant roles); None = the user's global role decides.
    tenant_role_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenant_roles.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=_now)


class TenantPermission(Base):
    """What a tenant role may do (v1: one permission, "Update")."""

    __tablename__ = "tenant_permissions"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_tenant_permission_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    group: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[str | None] = mapped_column(String(255))


class TenantRole(Base):
    """A role in one tenant (v1: ROOTS_INTERNAL_ADMIN, ANNOTATOR, and custom roles)."""

    __tablename__ = "tenant_roles"
    __table_args__ = (UniqueConstraint("tenant_id", "role_name", name="uq_tenant_role_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), index=True
    )
    role_name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[str | None] = mapped_column(String(255))


class TenantRolePermission(Base):
    __tablename__ = "tenant_role_permissions"

    role_id: Mapped[str] = mapped_column(
        ForeignKey("tenant_roles.id", ondelete="CASCADE"), primary_key=True
    )
    permission_id: Mapped[str] = mapped_column(
        ForeignKey("tenant_permissions.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[str | None] = mapped_column(String(255))


def _v1_project_id(project: Project) -> int | None:
    """The v1 doc processing project id an imported configuration carries, if any."""
    raw = str((project.parameters or {}).get("doc_processing_project_id") or "").strip()
    return int(raw) if raw.isascii() and raw.isdigit() and len(raw) <= 9 else None


@event.listens_for(Project, "before_insert")
def _assign_platform_id(_mapper, connection, project: Project) -> None:
    """Integer project id: the v1 id of an imported configuration when it is free (so automation
    configs keep working unchanged), else the next number (Postgres sequence, which the column
    default shares; MAX + 1 on SQLite, used by tests)."""
    if project.platform_id is None:
        v1 = _v1_project_id(project)
        taken = v1 is not None and connection.scalar(
            select(Project.id).where(Project.platform_id == v1)
        )
        if v1 is not None and not taken:
            project.platform_id = v1
    postgres = connection.dialect.name == "postgresql"
    if project.platform_id is not None:
        if postgres:  # keep the sequence ahead of explicit ids
            connection.execute(
                text(
                    "SELECT setval('projects_platform_id_seq', GREATEST("
                    "(SELECT last_value FROM projects_platform_id_seq), :v))"
                ),
                {"v": project.platform_id},
            )
        return
    if postgres:
        project.platform_id = connection.scalar(text("SELECT nextval('projects_platform_id_seq')"))
    else:
        current = connection.scalar(select(func.max(Project.platform_id)))
        project.platform_id = (current or 0) + 1
