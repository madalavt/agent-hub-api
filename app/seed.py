"""Startup seeding: the ML model catalog, then example customers/projects/trees.

Each ``seed/*.json`` file describes one project::

    {"customer": "...", "project": "...", "project_description": "...", "tree_name": "...",
     "export": <Platform-DecisionTree-API export>,
     "branches": [{"parent": "Node A", "child": "Node B", "condition": "..."}],   # optional
     "starting_node": "Node A",                                                     # optional
     "export_from": "other_seed.json",   # optional: reuse that seed's export/branches/start
     "customer_id": "...", "project_id": "...",   # optional fixed ids (same in every env)
     "customer_description": "..."}                                                 # optional

A seed is applied once: it is skipped when the customer already has a project with that name.
Seeding runs on every start (SEED_ON_START, on by default), so the seeded customers and
projects (e.g. the **Demo** organization with Mortgage Change and Loss Runs) exist in every
environment. Fixed ids let other services (the work item manager's Demo queues) link to them.
Exports must not contain secrets (the importer strips them anyway).
"""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from pathlib import Path

from sqlalchemy import func, select

from app import importer
from app.auth.passwords import hash_password
from app.catalog import ensure_catalog
from app.db import session_scope
from app.models import Customer, Project, TenantProject, User, UserCustomer

logger = logging.getLogger(__name__)


def seed(
    seed_dir: Path | None,
    local_models: list[str] | None,
    admin_password: str | None = None,
) -> None:
    with session_scope() as session:
        added = ensure_catalog(session, local_models)
        if added:
            logger.info("Added catalog models: %s", ", ".join(added))
    if not seed_dir or not seed_dir.is_dir():
        seed_users(admin_password)
        return
    for path in sorted(seed_dir.glob("*.json")):
        try:
            _apply(path)
        except Exception:
            logger.exception("Seed %s failed", path.name)
    seed_users(admin_password)


def seed_users(admin_password: str | None) -> None:
    """First run only (no users yet): an admin plus one editor and one viewer on the demo
    customers. The admin password comes from SEED_ADMIN_PASSWORD; any password not set there
    is generated and logged once, so change them after the first sign-in."""
    with session_scope() as session:
        if session.scalar(select(func.count()).select_from(User)):
            return
        generated: dict[str, str] = {}

        def make(username: str, name: str, role: str, password: str | None) -> User:
            if not password:
                password = secrets.token_urlsafe(12)
                generated[username] = password
            user = User(
                username=username,
                name=name,
                role=role,
                provider="local",
                password_hash=hash_password(password),
            )
            new_id = str(uuid.uuid4())
            user.id = new_id
            user.external_id = new_id
            session.add(user)
            session.flush()
            return user

        make("admin", "Admin", "admin", admin_password)
        editor = make("editor", "Editor (Selective)", "editor", None)
        viewer = make("viewer", "Viewer (Demo)", "viewer", None)
        for user, customer_name in ((editor, "Selective"), (viewer, "Demo")):
            customer = session.scalar(select(Customer).where(Customer.name == customer_name))
            if customer:
                session.add(UserCustomer(user_id=user.id, customer_id=customer.id))
        if generated:
            logger.warning(
                "Seeded sign-in passwords (change them after first sign-in): %s",
                ", ".join(f"{u}={p}" for u, p in generated.items()),
            )
        logger.info("Seeded users: admin, editor, viewer")


# Queues of the Demo projects (fixed ids, created by services/work-items seed_default_queues).
DEMO_QUEUE_IDS = {
    "de300000-0000-4000-8000-0000000000a1": "de3000000000400080000000000000c1",
    "de300000-0000-4000-8000-0000000000a2": "de3000000000400080000000000000c2",
}


def _apply(path: Path) -> None:
    data = json.loads(path.read_text())
    if data.get("export_from"):
        base = json.loads((path.parent / data["export_from"]).read_text())
        for key in ("export", "branches", "starting_node"):
            data.setdefault(key, base.get(key))
    with session_scope() as session:
        customer = session.scalar(select(Customer).where(Customer.name == data["customer"]))
        if customer and session.scalar(
            select(Project.id).where(
                Project.customer_id == customer.id, Project.name == data["project"]
            )
        ):
            return
        tree, report = importer.import_tree(
            session,
            data["export"],
            name=data.get("tree_name"),
            branches=data.get("branches"),
            starting_node=data.get("starting_node"),
        )
        if customer is None:
            customer = Customer(
                name=data["customer"], description=data.get("customer_description")
            )
            if data.get("customer_id") and session.get(Customer, data["customer_id"]) is None:
                customer.id = data["customer_id"]
            session.add(customer)
            session.flush()
        tree.customer_id = customer.id  # the org's decision tree library
        project_id = data.get("project_id")
        if project_id and session.get(Project, project_id) is not None:
            project_id = None  # taken (e.g. renamed and re-seeded): use a fresh id
        session.add(
            Project(
                **({"id": project_id} if project_id else {}),
                customer_id=customer.id,
                name=data["project"],
                description=data.get("project_description"),
                tree_id=tree.id,
                parameters=report.get("configuration", {}),
            )
        )
        session.flush()
        # The org's project for this process (v1 roots-auth project), with the same id, and the
        # Demo queues' fixed ids (services/work-items seed).
        project = session.scalars(
            select(Project).where(
                Project.customer_id == customer.id, Project.name == data["project"]
            )
        ).first()
        if project and session.get(TenantProject, project.id) is None:
            session.add(
                TenantProject(
                    id=project.id,
                    customer_id=customer.id,
                    name=data["project"],
                    description=data.get("project_description"),
                    queue_id=DEMO_QUEUE_IDS.get(project.id),
                    created_by="system",
                )
            )
        logger.info(
            "Seeded %s / %s (tree %s): %s",
            data["customer"],
            data["project"],
            tree.name,
            {k: v for k, v in report.items() if k != "configuration"},
        )
