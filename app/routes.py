from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app import access, graph, importer
from app.auth.dependencies import (
    PrincipalDep,
    ReadSessionDep,
    SessionDep,
    SettingsDep,
    get_principal,
)
from app.config import Settings
from app.mlmodels import ModelParamsError, validate_model_params
from app.models import Customer, MlModel, Node, Project, TenantProject, Tree
from app.schemas import (
    CODE_FUNCTIONS,
    FIELD_TYPES,
    NODE_TYPES,
    CustomerIn,
    CustomerOut,
    MlModelIn,
    MlModelOut,
    ProjectConfig,
    ProjectIn,
    ProjectOut,
    ProjectUpdate,
    TreeGraphIn,
    TreeGraphOut,
    TreeSummary,
)

# Every endpoint needs a signed-in user or the service key (see app/auth/dependencies.py).
# Role checks per endpoint live in app/access.py.
router = APIRouter(prefix="/v1", dependencies=[Depends(get_principal)])


def _404(what: str, id_: str):
    return HTTPException(status.HTTP_404_NOT_FOUND, f"{what} {id_} not found")


def _get(session: Session, model, id_: str, what: str):
    obj = session.get(model, id_)
    if obj is None:
        raise _404(what, id_)
    return obj


def _tree_or_404(session: Session, tree_id: str) -> Tree:
    tree = graph.load(session, tree_id)
    if tree is None:
        raise _404("tree", tree_id)
    return tree


def _validation_error(exc: graph.TreeValidationError) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, {"errors": exc.errors})


# ------------------------------------------------------------------ customers


def _customer_out(session: Session, c: Customer) -> CustomerOut:
    count = session.scalar(select(func.count()).where(Project.customer_id == c.id)) or 0
    return CustomerOut(
        id=c.id,
        name=c.name,
        description=c.description,
        created_at=c.created_at,
        project_count=count,
    )


@router.get("/customers", response_model=list[CustomerOut], tags=["customers"])
def list_customers(principal: PrincipalDep, session: SessionDep):
    query = select(Customer).order_by(Customer.name)
    if not principal.reads_all:
        query = query.where(Customer.id.in_(principal.customer_ids or []))
    return [_customer_out(session, c) for c in session.scalars(query)]


@router.post("/customers", response_model=CustomerOut, status_code=201, tags=["customers"])
def create_customer(body: CustomerIn, principal: PrincipalDep, session: SessionDep):
    access.require_admin(principal)
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "name cannot be blank")
    customer = Customer(name=name, description=body.description)
    session.add(customer)
    try:
        session.flush()
    except IntegrityError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "a customer with that name exists") from exc
    from app.tenants import ensure_tenant_defaults  # noqa: PLC0415

    ensure_tenant_defaults(session, customer.id, actor=principal.name)
    return _customer_out(session, customer)


@router.get("/customers/{customer_id}", response_model=CustomerOut, tags=["customers"])
def get_customer(customer_id: str, principal: PrincipalDep, session: SessionDep):
    access.read_customer(principal, customer_id)
    return _customer_out(session, _get(session, Customer, customer_id, "customer"))


@router.put("/customers/{customer_id}", response_model=CustomerOut, tags=["customers"])
def update_customer(
    customer_id: str, body: CustomerIn, principal: PrincipalDep, session: SessionDep
):
    access.require_admin(principal)
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "name cannot be blank")
    customer = _get(session, Customer, customer_id, "customer")
    customer.name, customer.description = name, body.description
    try:
        session.flush()
    except IntegrityError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "a customer with that name exists") from exc
    return _customer_out(session, customer)


@router.delete("/customers/{customer_id}", status_code=204, tags=["customers"])
def delete_customer(customer_id: str, principal: PrincipalDep, session: SessionDep):
    access.require_admin(principal)
    session.delete(_get(session, Customer, customer_id, "customer"))


# ------------------------------------------------------------------ projects


def _project_out(session: Session, p: Project) -> ProjectOut:
    tree_name = (
        session.scalar(select(Tree.name).where(Tree.id == p.tree_id)) if p.tree_id else None
    )
    out = ProjectOut.model_validate(p)
    out.tree_name = tree_name
    return out


def _check_tree(session: Session, tree_id: str | None) -> None:
    if tree_id and session.get(Tree, tree_id) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"tree {tree_id} not found")


@router.get(
    "/customers/{customer_id}/projects", response_model=list[ProjectOut], tags=["projects"]
)
def list_projects(customer_id: str, principal: PrincipalDep, session: SessionDep):
    access.read_customer(principal, customer_id)
    _get(session, Customer, customer_id, "customer")
    rows = session.scalars(
        select(Project).where(Project.customer_id == customer_id).order_by(Project.created_at)
    )
    return [_project_out(session, p) for p in rows]


@router.post(
    "/customers/{customer_id}/projects",
    response_model=ProjectOut,
    status_code=201,
    tags=["projects"],
)
def create_project(
    customer_id: str, body: ProjectIn, principal: PrincipalDep, session: SessionDep
):
    access.write_customer(principal, customer_id)
    _get(session, Customer, customer_id, "customer")
    _check_tree(session, body.tree_id)
    if body.tree_id:
        access.check_tree_assignable(session, principal, body.tree_id)
    project = Project(customer_id=customer_id, **body.model_dump())
    session.add(project)
    session.flush()
    return _project_out(session, project)


def _project_for(session: Session, principal, project_id: str, write: bool) -> Project:
    """A project by its id or by its integer platform id (what v1 automation configs hold)."""
    if project_id.isascii() and project_id.isdigit():
        num = int(project_id)
        if num > 2_147_483_647:
            raise _404("project", project_id)
        project = session.scalar(select(Project).where(Project.platform_id == num))
        if project is None:
            raise _404("project", project_id)
    else:
        project = _get(session, Project, project_id, "project")
    if write:
        access.write_customer(principal, project.customer_id)
    else:
        access.read_customer(principal, project.customer_id)
    return project


@router.get(
    "/projects",
    response_model=list[ProjectOut],
    tags=["projects"],
    summary="Projects the caller may see (all customers for admins and services)",
)
def list_visible_projects(principal: PrincipalDep, session: SessionDep):
    query = select(Project).order_by(Project.name)
    if not principal.reads_all:
        query = query.where(Project.customer_id.in_(principal.customer_ids or []))
    return [_project_out(session, p) for p in session.scalars(query)]


@router.get("/projects/{project_id}", response_model=ProjectOut, tags=["projects"])
def get_project(project_id: str, principal: PrincipalDep, session: SessionDep):
    return _project_out(session, _project_for(session, principal, project_id, write=False))


@router.patch("/projects/{project_id}", response_model=ProjectOut, tags=["projects"])
def update_project(
    project_id: str, body: ProjectUpdate, principal: PrincipalDep, session: SessionDep
):
    project = _project_for(session, principal, project_id, write=True)
    data = body.model_dump(exclude_unset=True)
    if "name" in data:
        if not data["name"] or not data["name"].strip():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "name cannot be blank")
        data["name"] = data["name"].strip()
    if "parameters" in data and data["parameters"] is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "parameters cannot be null")
    if "tree_id" in data:
        _check_tree(session, data["tree_id"])
        if data["tree_id"] and data["tree_id"] != project.tree_id:
            access.check_tree_assignable(session, principal, data["tree_id"])
    for k, v in data.items():
        setattr(project, k, v)
    session.flush()
    return _project_out(session, project)


@router.delete("/projects/{project_id}", status_code=204, tags=["projects"])
def delete_project(project_id: str, principal: PrincipalDep, session: SessionDep):
    # Projects are normally deleted in the database; through the API only admins may.
    access.require_admin(principal)
    session.delete(_project_for(session, principal, project_id, write=True))


@router.get(
    "/projects/{project_id}/config",
    response_model=ProjectConfig,
    tags=["projects"],
    summary="Project + customer + full tree + models, for the extraction service",
)
def get_project_config(project_id: str, principal: PrincipalDep, session: SessionDep):
    project = _project_for(session, principal, project_id, write=False)
    if not project.tree_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "project has no decision tree assigned")
    tree = _tree_or_404(session, project.tree_id)
    return ProjectConfig(
        project=_project_out(session, project),
        customer=_customer_out(session, project.customer),
        tree=graph.serialize(session, tree),
        mlmodels={
            m.id: MlModelOut.model_validate(m) for m in graph.referenced_models(session, tree)
        },
    )


# ------------------------------------------------------------------ trees


@router.get("/trees", response_model=list[TreeSummary], tags=["trees"])
def list_trees(
    principal: PrincipalDep,
    session: ReadSessionDep,
    customer_id: Annotated[str | None, Query(description="One org's library")] = None,
):
    query = select(Tree).order_by(Tree.name)
    if customer_id:
        access.read_customer(principal, customer_id)
        used = select(Project.tree_id).where(Project.customer_id == customer_id)
        query = query.where((Tree.customer_id == customer_id) | Tree.id.in_(used))
    visible = access.visible_tree_ids(principal)
    if visible is not None:
        query = query.where(Tree.id.in_(visible))
    out = []
    for t in session.scalars(query):
        node_count = session.scalar(select(func.count()).where(Node.tree_id == t.id)) or 0
        projects_q = select(Project.id, Project.name, Project.customer_id).where(
            Project.tree_id == t.id
        )
        if not principal.reads_all:
            projects_q = projects_q.where(Project.customer_id.in_(principal.customer_ids or []))
        projects = session.execute(projects_q).all()
        out.append(
            TreeSummary(
                id=t.id,
                customer_id=t.customer_id,
                name=t.name,
                description=t.description,
                node_count=node_count,
                version=t.version,
                updated_at=t.updated_at,
                projects=[
                    {"id": p.id, "name": p.name, "customer_id": p.customer_id} for p in projects
                ],
            )
        )
    return out


@router.post("/trees", response_model=TreeGraphOut, status_code=201, tags=["trees"])
def create_tree(body: TreeGraphIn, principal: PrincipalDep, session: SessionDep):
    # In an org's library: members with Update on the org. Without an org: admins only.
    if body.customer_id:
        _get(session, Customer, body.customer_id, "customer")
        access.write_customer(principal, body.customer_id)
    else:
        access.require_admin(principal)
    tree = Tree(name=body.name, customer_id=body.customer_id)
    session.add(tree)
    session.flush()
    try:
        graph.save(session, tree, body)
    except graph.TreeValidationError as exc:
        raise _validation_error(exc) from exc
    return graph.serialize(session, _tree_or_404(session, tree.id))


@router.get("/trees/{tree_id}", response_model=TreeGraphOut, tags=["trees"])
def get_tree(tree_id: str, principal: PrincipalDep, session: SessionDep):
    access.read_tree(session, principal, tree_id)
    return graph.serialize(session, _tree_or_404(session, tree_id), principal=principal)


@router.put(
    "/trees/{tree_id}",
    response_model=TreeGraphOut,
    tags=["trees"],
    summary="Replace the whole tree (nodes, fields, branches, layout)",
)
def update_tree(tree_id: str, body: TreeGraphIn, principal: PrincipalDep, session: SessionDep):
    access.write_tree(session, principal, tree_id)
    tree = _tree_or_404(session, tree_id)
    if body.customer_id and body.customer_id != tree.customer_id:
        # Moving a tree to another org's library is an admin action; others keep the owner.
        if principal.role == "admin":
            _get(session, Customer, body.customer_id, "customer")
            tree.customer_id = body.customer_id
    try:
        graph.save(session, tree, body)
    except graph.TreeValidationError as exc:
        raise _validation_error(exc) from exc
    session.expire_all()
    return graph.serialize(session, _tree_or_404(session, tree_id), principal=principal)


@router.post("/trees/validate", tags=["trees"], summary="Validate a tree without saving it")
def validate_tree(
    body: TreeGraphIn, principal: PrincipalDep, session: SessionDep
) -> dict[str, Any]:
    access.require_editor(principal)
    try:
        graph.validate(body, session)
    except graph.TreeValidationError as exc:
        return {"valid": False, "errors": exc.errors}
    return {"valid": True, "errors": []}


@router.post(
    "/trees/{tree_id}/duplicate", response_model=TreeGraphOut, status_code=201, tags=["trees"]
)
def duplicate_tree(
    tree_id: str,
    principal: PrincipalDep,
    session: SessionDep,
    customer_id: Annotated[str | None, Query(description="Org library for the copy")] = None,
):
    import uuid

    access.read_tree(session, principal, tree_id)
    src = graph.serialize(session, _tree_or_404(session, tree_id))
    id_map = {n.id: str(uuid.uuid4()) for n in src.nodes}
    data = TreeGraphIn.model_validate(src.model_dump())
    data.name = f"{src.name} (copy)"
    data.customer_id = customer_id or src.customer_id  # create_tree checks write access
    for n in data.nodes:
        n.id = id_map[n.id]
    for b in data.branches:
        b.id = None
        b.parent_node_id, b.child_node_id = id_map[b.parent_node_id], id_map[b.child_node_id]
    data.starting_node_id = id_map.get(data.starting_node_id or "", None)
    positions = (data.layout or {}).get("positions") or {}
    if not isinstance(positions, dict):
        positions = {}
    data.layout = {"positions": {id_map[k]: v for k, v in positions.items() if k in id_map}}
    return create_tree(data, principal, session)


@router.delete("/trees/{tree_id}", status_code=204, tags=["trees"])
def delete_tree(tree_id: str, principal: PrincipalDep, session: SessionDep):
    tree = _tree_or_404(session, tree_id)
    if tree.customer_id and principal.role != "admin":
        access.write_customer(principal, tree.customer_id)
    else:
        access.require_admin(principal)
    in_use = session.scalar(select(func.count()).where(Project.tree_id == tree_id))
    if in_use:
        raise HTTPException(status.HTTP_409_CONFLICT, f"tree is used by {in_use} project(s)")
    session.delete(tree)


@router.post(
    "/trees/import",
    status_code=201,
    tags=["trees"],
    summary="Import a Platform-DecisionTree-API configuration export",
)
def import_tree(
    principal: PrincipalDep,
    session: SessionDep,
    payload: Annotated[dict[str, Any], Body()],
    name: Annotated[str | None, Query()] = None,
    customer_id: Annotated[str | None, Query(description="Org library for the tree")] = None,
) -> dict[str, Any]:
    access.require_admin(principal)
    if customer_id:
        _get(session, Customer, customer_id, "customer")
    try:
        tree, report = importer.import_tree(
            session,
            payload.get("export", payload),
            name=name,
            branches=payload.get("branches"),
            starting_node=payload.get("starting_node"),
        )
    except graph.TreeValidationError as exc:
        raise _validation_error(exc) from exc
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    tree.customer_id = customer_id
    return {"tree_id": tree.id, "name": tree.name, "customer_id": customer_id, **report}


# ------------------------------------------------------------------ nodes (v1 igpt shape)

_DISPLAY_TYPES = {
    "string": "Text",
    "number": "Number",
    "currency": "Currency",
    "date": "Date",
    "boolean": "Checkbox",
    "table": "Table",
}
_JSON_TYPES = {"number": "number", "currency": "number", "boolean": "boolean", "table": "array"}


def _node_or_404(session: Session, principal, node_id: str) -> Node:
    node = session.get(Node, node_id)
    if node is None:
        raise _404("node", node_id)
    access.read_tree(session, principal, node.tree_id)  # 404 when the tree isn't visible
    return node


def _field_display(f) -> dict[str, Any]:
    return {
        "options": None,
        "sort_order": f.sort_order,
        "date_format": None,
        "maximum_value": None,
        "minimum_value": None,
        "number_of_rows": None,
        "placeholder_text": None,
    }


def _field_type_v1(name: str) -> dict[str, Any]:
    return {**_field_type(name), "display_type": _DISPLAY_TYPES.get(name, "Text").lower()}


def _v1_field(f, children: list | None = None) -> dict[str, Any]:
    """A decision-tree field in v1's igpt shape (configuration, project field library)."""
    return {
        "id": f.id,
        "categories": "",
        "field_type": _field_type_v1(f.field_type),
        "description": f.description or "",
        "name": f.name,
        "standard": f.standard,
        "threshold": f.threshold,
        "customer_key": f.customer_key,
        "field_display": _field_display(f),
        "library_field": {
            "template": {
                "id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"agent-hub/field-default/{f.field_type}")
                ),
                "name": f.field_type.capitalize(),
            }
        },
        "child_fields": [_v1_field(c) for c in children or []],
    }


def _property(f, children) -> dict[str, Any]:
    out = {
        "type": _JSON_TYPES.get(f.field_type, "string"),
        "order": f.sort_order,
        "title": f.name,
        "customer_key": f.customer_key,
        "display_type": _DISPLAY_TYPES.get(f.field_type, "Text"),
        "field_display": _field_display(f),
        "child_fields": [_v1_field(c) for c in children],
    }
    if f.field_type == "table":
        ordered = sorted(children, key=lambda c: c.sort_order)
        out["column_order"] = {c.customer_key: i for i, c in enumerate(ordered)}
    return out


def _display_template(node) -> dict[str, Any]:
    children: dict[str, list] = {}
    for f in node.fields:
        if f.parent_id:
            children.setdefault(f.parent_id, []).append(f)
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-hub/display-template/{node.id}")),
        "template": {
            "type": "",
            "title": node.name,
            "description": node.description or "",
            "disabled": [],
            "required": [],
            "properties": {
                f.id: _property(f, children.get(f.id, []))
                for f in node.fields
                if f.parent_id is None
            },
        },
    }


@router.get("/nodes/{node_id}", tags=["trees"], summary="A decision-tree node with its fields")
def get_node(node_id: str, principal: PrincipalDep, session: SessionDep) -> dict[str, Any]:
    node = _node_or_404(session, principal, node_id)
    siblings = session.execute(
        select(Node.id, Node.name).where(Node.tree_id == node.tree_id).order_by(Node.sort_order)
    ).all()
    projects_q = select(Project.id, Project.name).where(Project.tree_id == node.tree_id)
    if not principal.reads_all:
        projects_q = projects_q.where(Project.customer_id.in_(principal.customer_ids or []))
    return {
        "node": {
            "id": node.id,
            "name": node.name,
            "standard": node.standard,
            "type": node.type,
            "fields": [
                {
                    "id": f.id,
                    "categories": "",
                    "description": f.description or "",
                    "name": f.name,
                    "standard": f.standard,
                    "threshold": f.threshold,
                    "customer_key": f.customer_key,
                }
                for f in node.fields
                if f.parent_id is None
            ],
            "tree": {
                "extraction_nodes": [{"id": i, "name": n} for i, n in siblings],
                "configurations": [
                    {"id": i, "name": n, "enable_pss": False}
                    for i, n in session.execute(projects_q).all()
                ],
            },
        }
    }


@router.get(
    "/nodes/{node_id}/display-template",
    tags=["trees"],
    summary="How a node's fields are shown: titles and order (generated from the fields)",
)
def get_node_display_template(
    node_id: str, principal: PrincipalDep, session: SessionDep
) -> dict[str, Any]:
    node = _node_or_404(session, principal, node_id)
    return {
        "node": {
            "id": node.id,
            "name": node.name,
            "standard": node.standard,
            "type": node.type,
            "display_template": _display_template(node),
        }
    }


# ------------------------------------------------------------------ igpt (v1 UI shapes)
# The UI reaches these through /api/roots-igpt/... like v1: customer/{id}, configuration/{id},
# mlmodels, field-default-template, project-field-library/{projectId}. A v1 "configuration" is
# an Agent Hub project (a customer's process with its decision tree). Secrets (api_key) are
# never returned.

_CONFIG_DEFAULTS = {
    "split_package": False,
    "ocr_page_limit": None,
    "include_hidden_sheets": False,
    "split_for_igpt": False,
}


def _configuration(p: Project) -> dict[str, Any]:
    params = {
        k: v for k, v in (p.parameters or {}).items() if k.lower() not in importer.SECRET_KEYS
    }
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description or "",
        "parameters": params or None,
        **{k: params.get(k, d) for k, d in _CONFIG_DEFAULTS.items()},
        "doc_processing_project_id": str(
            params.get("doc_processing_project_id") or p.platform_id or p.id
        ),
        "enable_pss": bool(params.get("enable_pss", False)),
        "tree_id": p.tree_id,
    }


def _field_type(name: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-hub/field-type/{name}")),
        "name": name,
        "customer_field_type": None,
    }


@router.get("/igpt/customers/{customer_id}", tags=["igpt"], summary="Customer + configurations")
def igpt_customer(customer_id: str, principal: PrincipalDep, session: SessionDep):
    access.read_customer(principal, customer_id)
    c = _get(session, Customer, customer_id, "customer")
    projects = session.scalars(
        select(Project).where(Project.customer_id == c.id).order_by(Project.name)
    )
    return {
        "customer": {
            "id": c.id,
            "name": c.name,
            "configurations": [_configuration(p) for p in projects],
        }
    }


@router.get(
    "/igpt/configurations/{project_id}", tags=["igpt"], summary="Configuration + its decision tree"
)
def igpt_configuration(project_id: str, principal: PrincipalDep, session: SessionDep):
    p = _project_for(session, principal, project_id, write=False)
    out = _configuration(p)
    if p.tree_id:
        out["tree"] = _v1_tree(session, _tree_or_404(session, p.tree_id))
    else:
        out["tree"] = None
    return {"configuration": out}


def _v1_tree(session: Session, tree: Tree) -> dict[str, Any]:
    """The decision tree as v1's configuration returns it (layout, nodes, branches)."""
    raw_positions = (tree.layout or {}).get("positions")
    positions = raw_positions if isinstance(raw_positions, dict) else {}
    models = {m.id: m for m in session.scalars(select(MlModel))}
    nodes = sorted(tree.nodes, key=lambda n: n.sort_order)

    def mlmodel(n) -> dict[str, Any] | None:
        m = models.get(n.mlmodel_id) if n.mlmodel_id else None
        if m is None:
            return None
        return {
            "id": m.id,
            "name": m.name,
            "parameters": m.parameters,
            "standard": m.standard,
            "type": m.type,
        }

    def node_out(n) -> dict[str, Any]:
        children: dict[str, list] = {}
        for f in n.fields:
            if f.parent_id:
                children.setdefault(f.parent_id, []).append(f)
        top = sorted((f for f in n.fields if f.parent_id is None), key=lambda f: f.sort_order)
        return {
            "id": n.id,
            "name": n.name,
            "description": n.description,
            "parameters": n.parameters,
            "standard": n.standard,
            "type": n.type,
            "mlmodel_id": n.mlmodel_id,
            "mlmodel": mlmodel(n),
            "fields": [_v1_field(f, children.get(f.id, [])) for f in top],
            "display_template": _display_template(n),
        }

    branches = sorted(tree.branches, key=lambda b: b.sort_order)
    return {
        "id": tree.id,
        "name": tree.name,
        "starting_node": {"id": tree.starting_node_id} if tree.starting_node_id else None,
        "layout": _v1_layout(tree, nodes, branches, positions),
        "nodes": [node_out(n) for n in nodes],
        "branches": [
            {
                "id": b.id,
                "name": b.name,
                "standard": True,
                "condition": b.condition,
                "parent_node": {"id": b.parent_node_id},
                "child_node": {"id": b.child_node_id},
                "parent_node_id": b.parent_node_id,
                "child_node_id": b.child_node_id,
            }
            for b in branches
        ],
    }


def _unconditional(condition: str | None) -> bool:
    return (condition or "").strip().lower() in ("", "true", "always")


def _v1_layout(tree: Tree, nodes: list, branches: list, positions: dict) -> dict[str, Any]:
    """v1's canvas: a Start node joined to the starting node, the tree's nodes, and an
    Extraction End node after every leaf. A node's first or unconditional branch leaves from
    its right handle ("b"); further conditional branches leave from the bottom ("c")."""

    def synthetic(key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-hub/tree/{tree.id}/{key}"))

    def pos(key: str, i: int) -> dict[str, Any]:
        p = positions.get(key)
        if isinstance(p, dict) and "x" in p and "y" in p:
            return p
        return {"x": 300 * (i + 1), "y": 0}

    by_id = {n.id: n for n in nodes}
    order = {n.id: i for i, n in enumerate(nodes)}
    layout_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    start = tree.starting_node_id if tree.starting_node_id in by_id else None
    if start:
        sp = pos(start, order[start])
        start_id = synthetic("start")
        layout_nodes.append(
            {
                "id": start_id,
                "data": {"name": "Start"},
                "type": "start_node",
                "position": positions.get("start") or {"x": sp["x"] - 420, "y": sp["y"]},
            }
        )
        edges.append(_v1_edge(f"{start_id}-{start}", start_id, start, "Connection", "b"))
    layout_nodes += [
        {"id": n.id, "data": {"name": n.name}, "type": n.type, "position": pos(n.id, i)}
        for i, n in enumerate(nodes)
    ]
    used: set[str] = set()
    for b in branches:
        plain = _unconditional(b.condition)
        handle = "b" if plain or b.parent_node_id not in used else "c"
        used.add(b.parent_node_id)
        edges.append(
            _v1_edge(
                b.id,
                b.parent_node_id,
                b.child_node_id,
                "Connection" if plain else "Condition",
                handle,
            )
        )
    parents = {b.parent_node_id for b in branches}
    for i, n in enumerate(nodes):
        if n.id in parents:
            continue
        end_id = synthetic(f"end/{n.id}")
        np_ = pos(n.id, i)
        layout_nodes.append(
            {
                "id": end_id,
                "data": {"name": "Extraction End"},
                "type": "end_node",
                "position": positions.get(f"end:{n.id}") or {"x": np_["x"] + 270, "y": np_["y"]},
            }
        )
        edges.append(_v1_edge(f"{n.id}-{end_id}", n.id, end_id, "Connection", "b"))
    return {"nodes": layout_nodes, "edges": edges}


def _v1_edge(edge_id: str, source: str, target: str, logic: str, handle: str) -> dict[str, Any]:
    return {
        "id": edge_id,
        "type": "addWorkflowEdge",
        "source": source,
        "target": target,
        "logicType": logic,
        "sourceHandle": handle,
        "targetHandle": "d",
    }


@router.get("/igpt/mlmodels", tags=["igpt"], summary="ML models (id, name, type)")
def igpt_mlmodels(session: SessionDep):
    models = session.scalars(select(MlModel).order_by(MlModel.name))
    return {
        "mlmodels": [
            {"id": m.id, "name": m.name, "standard": m.standard, "type": m.type} for m in models
        ]
    }


@router.get("/igpt/field-default-template", tags=["igpt"], summary="Default field per field type")
def igpt_field_default_template():
    return {
        "fieldDefaultTemplates": [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-hub/field-default/{t}")),
                "categories": "",
                "field_type": _field_type(t),
                "description": "",
                "name": t.capitalize(),
                "standard": True,
                "threshold": 0.5,
            }
            for t in FIELD_TYPES
        ]
    }


def _library_template(field: dict[str, Any]) -> dict[str, Any]:
    """v1: the library field a project field was made from (the field type's default)."""
    t = field["field_type"]["name"]
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-hub/field-default/{t}")),
        "categories": "",
        "field_type": _field_type(t),
        "description": "",
        "name": t.capitalize(),
        "standard": True,
        "threshold": 0.5,
    }


@router.get(
    "/igpt/project-field-library/{project_id}",
    tags=["igpt"],
    summary="Fields the project's decision tree extracts (one per customer key)",
)
def igpt_project_field_library(project_id: str, principal: PrincipalDep, session: SessionDep):
    """v1 calls this with the page's project (a tenant project): the fields of every decision
    tree in the org's library. A tree configuration id gives that configuration's tree."""
    work = session.get(TenantProject, project_id.lower())
    if work is not None:
        access.read_customer(principal, work.customer_id)
        owner_id = work.id
        used = select(Project.tree_id).where(Project.customer_id == work.customer_id)
        trees = session.scalars(
            select(Tree)
            .where((Tree.customer_id == work.customer_id) | Tree.id.in_(used))
            .options(selectinload(Tree.nodes).selectinload(Node.fields))
            .order_by(Tree.name)
        ).all()
    else:
        p = _project_for(session, principal, project_id, write=False)
        owner_id = p.id
        trees = [_tree_or_404(session, p.tree_id)] if p.tree_id else []
    library: dict[str, dict[str, Any]] = {}
    for tree in trees:
        for node in tree.nodes:
            children: dict[str, list] = {}
            for f in node.fields:
                if f.parent_id:
                    children.setdefault(f.parent_id, []).append(f)
            for f in node.fields:
                if f.parent_id or f.customer_key in library:
                    continue
                field = _v1_field(f, children.get(f.id, []))
                field["project_id"] = owner_id
                for c in field["child_fields"]:
                    c["project_id"] = owner_id
                    c["template"] = _library_template(c)
                field["template"] = _library_template(field)
                library[f.customer_key] = field
    return {"projectFieldLibrary": list(library.values())}


# ------------------------------------------------------------------ ml models


@router.get("/mlmodels", response_model=list[MlModelOut], tags=["mlmodels"])
def list_models(session: ReadSessionDep):
    return list(session.scalars(select(MlModel).order_by(MlModel.name)))


def _check_model(body: MlModelIn, settings: Settings) -> dict[str, Any]:
    try:
        return validate_model_params(body.parameters, settings.supported_model_types)
    except ModelParamsError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@router.post("/mlmodels", response_model=MlModelOut, status_code=201, tags=["mlmodels"])
def create_model(
    body: MlModelIn, principal: PrincipalDep, session: SessionDep, settings: SettingsDep
):
    access.require_admin(principal)
    params = _check_model(body, settings)
    model = MlModel(name=body.name, type=body.type, parameters=params, standard=body.standard)
    session.add(model)
    session.flush()
    return model


@router.put("/mlmodels/{model_id}", response_model=MlModelOut, tags=["mlmodels"])
def update_model(
    model_id: str,
    body: MlModelIn,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
):
    access.require_admin(principal)
    model = _get(session, MlModel, model_id, "ML model")
    params = _check_model(body, settings)
    model.name, model.type, model.parameters, model.standard = (
        body.name,
        body.type,
        params,
        body.standard,
    )
    return model


@router.delete("/mlmodels/{model_id}", status_code=204, tags=["mlmodels"])
def delete_model(model_id: str, principal: PrincipalDep, session: SessionDep):
    access.require_admin(principal)
    model = _get(session, MlModel, model_id, "ML model")
    in_use = session.scalar(select(func.count()).where(Node.mlmodel_id == model_id))
    if in_use:
        raise HTTPException(status.HTTP_409_CONFLICT, f"model is used by {in_use} node(s)")
    session.delete(model)


@router.get("/meta", tags=["meta"], summary="Allowed node types, functions and field types")
def meta(settings: SettingsDep) -> dict[str, Any]:
    return {
        "supported_model_types": settings.supported_model_types,
        "node_types": list(NODE_TYPES),
        "code_functions": list(CODE_FUNCTIONS),
        "field_types": list(FIELD_TYPES),
        "prompt_placeholders": {
            "gpt_node": ["{text}", "{extraction_fields}"],
            "tables.chunking.process": ["{text}", "{extraction_fields}"],
            "verifiers.verify_table.process": [
                "{text}",
                "{annot}",
                "{n_rows}",
                "{extraction_fields}",
            ],
            "verifiers.verify_business.process": [
                "{text}",
                "{relevant_rows}",
                "{violations}",
                "{extraction_fields}",
            ],
        },
        "business_rules": [
            "check_paid_components",
            "check_loss_paid",
            "check_reserve_components",
            "check_total_incurred",
            "check_date_order",
            "check_financial_totals",
        ],
        "condition_examples": [
            'document_type == "Mortgage Change"',
            'document_type in ["Check", "Claim"]',
            'policy_is_correct == "No" or claims_is_correct == "No"',
        ],
    }
