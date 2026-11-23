from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

FIELD_TYPES = ("string", "number", "currency", "date", "boolean", "table")
NODE_TYPES = ("gpt_node", "code_node")
CODE_FUNCTIONS = (
    "tables.chunking.process",
    "verifiers.verify_table.process",
    "verifiers.verify_business.process",
)


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ customers / projects


class CustomerIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class CustomerOut(_ORM):
    id: str
    name: str
    description: str | None
    created_at: datetime
    project_count: int = 0


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    tree_id: str | None = None
    parameters: dict[str, Any] = {}


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    tree_id: str | None = None
    parameters: dict[str, Any] | None = None


class ProjectOut(_ORM):
    id: str
    # Integer project id (like platform v1's) that automation projects can use instead of id.
    platform_id: int | None = None
    customer_id: str
    name: str
    description: str | None
    tree_id: str | None
    tree_name: str | None = None
    parameters: dict[str, Any]
    created_at: datetime


# ------------------------------------------------------------------ ml models


class MlModelIn(BaseModel):
    name: str = Field(min_length=1)
    type: str = "gpt"
    parameters: dict[str, Any] = {}
    standard: bool = True


class MlModelOut(_ORM):
    id: str
    name: str
    type: str
    parameters: dict[str, Any]
    standard: bool


# ------------------------------------------------------------------ tree graph


class FieldGraph(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1)
    # Platform keys may contain - / . & (e.g. "reference_/_loan_-_acord_125"); projects read
    # them verbatim, so only whitespace and quotes are refused. Keys used in branch conditions
    # must still be identifiers (letters, digits, underscore).
    customer_key: str = Field(min_length=1, max_length=255, pattern=r"^[^\s\"'`]+$")
    description: str = ""
    field_type: str = "string"
    threshold: float = 0.5
    child_fields: list[FieldGraph] = []


class NodeGraph(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1)
    type: str = "gpt_node"
    description: str | None = None
    mlmodel_id: str | None = None
    parameters: dict[str, Any] = {}
    fields: list[FieldGraph] = []


class BranchGraph(BaseModel):
    id: str | None = None
    parent_node_id: str
    child_node_id: str
    name: str | None = None
    condition: str | None = None


class TreeGraphIn(BaseModel):
    name: str = Field(min_length=1)
    # Owning org (decision tree library). Set on create; only an admin can move a tree later.
    customer_id: str | None = None
    description: str | None = None
    business_context: str | None = None
    starting_node_id: str | None = None
    parameters: dict[str, Any] = {}
    layout: dict[str, Any] = {}
    nodes: list[NodeGraph] = []
    branches: list[BranchGraph] = []


class TreeGraphOut(TreeGraphIn):
    id: str
    version: int
    created_at: datetime
    updated_at: datetime
    projects: list[dict[str, str]] = []


class TreeSummary(BaseModel):
    id: str
    customer_id: str | None = None
    name: str
    description: str | None
    node_count: int
    version: int
    updated_at: datetime
    projects: list[dict[str, str]] = []


class ProjectConfig(BaseModel):
    """Everything the extraction service needs to run a project's documents."""

    project: ProjectOut
    customer: CustomerOut
    tree: TreeGraphOut
    mlmodels: dict[str, MlModelOut]


# ------------------------------------------------------------------ auth and users


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=1024)


class UserOut(_ORM):
    id: str
    provider: str
    username: str | None
    name: str
    email: str | None
    role: str
    active: bool
    customer_ids: list[str] = []
    # Role and permissions per tenant (v1 tenant roles)
    tenants: list[dict] = []


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class UserIn(BaseModel):
    username: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._@-]+$")
    name: str = Field(min_length=1, max_length=255)
    email: str | None = None
    role: str = "viewer"
    password: str = Field(min_length=8, max_length=1024)
    customer_ids: list[str] = []


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    email: str | None = None
    role: str | None = None
    active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=1024)
    customer_ids: list[str] | None = None
