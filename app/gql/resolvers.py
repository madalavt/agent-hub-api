"""GraphQL resolvers (Ariadne). Business rules are shared with REST via ``app.graph``,
``app.importer`` and ``app.mlmodels``, so both APIs validate trees and models identically.

Each request gets one SQLAlchemy session (``info.context["session"]``), committed by the HTTP
endpoint after execution. A failing mutation rolls the session back before re-raising, so it
never leaves partial writes behind.
"""

from __future__ import annotations

import functools
import uuid
from datetime import datetime
from typing import Any

from ariadne import MutationType, ObjectType, QueryType, ScalarType
from graphql import GraphQLError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import graph, importer
from app.mlmodels import ModelParamsError, validate_model_params
from app.models import Branch, Customer, Field, MlModel, Node, Project, Tree
from app.routes import meta as rest_meta
from app.schemas import (
    BranchGraph,
    FieldGraph,
    MlModelOut,
    NodeGraph,
    TreeGraphIn,
)

query = QueryType()
mutation = MutationType()
json_scalar = ScalarType("JSON")


@json_scalar.serializer
def _serialize_json(value: Any) -> Any:
    return value


def _session(info) -> Session:
    return info.context["session"]


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _not_found(what: str, id_: str) -> GraphQLError:
    return GraphQLError(f"{what} {id_} not found", extensions={"code": "NOT_FOUND"})


def _invalid(message: str, errors: list[str] | None = None) -> GraphQLError:
    return GraphQLError(message, extensions={"code": "VALIDATION_ERROR", "errors": errors or []})


def mutating(fn):
    """Roll back the request's session when a mutation fails."""

    @functools.wraps(fn)
    def wrapper(obj, info, **kwargs):
        session = _session(info)
        try:
            result = fn(obj, info, **kwargs)
            session.flush()
            return result
        except graph.TreeValidationError as exc:
            session.rollback()
            raise _invalid("tree validation failed", exc.errors) from exc
        except (ModelParamsError, ValueError, KeyError, TypeError) as exc:
            session.rollback()
            raise _invalid(str(exc)) from exc
        except IntegrityError as exc:
            session.rollback()
            raise _invalid("conflicts with an existing record") from exc
        except Exception:
            session.rollback()
            raise

    return wrapper


def _load_tree(session: Session, tree_id: str) -> Tree:
    tree = graph.load(session, tree_id)
    if tree is None:
        raise _not_found("tree", tree_id)
    return tree


def _get(session: Session, model, id_: str, what: str):
    obj = session.get(model, id_)
    if obj is None:
        raise _not_found(what, id_)
    return obj


# ------------------------------------------------------------------ object types

customer_type = ObjectType("Customer")
project_type = ObjectType("Project")
tree_type = ObjectType("Tree")
node_type = ObjectType("Node")
branch_type = ObjectType("Branch")
field_type = ObjectType("Field")
mlmodel_type = ObjectType("MlModel")


@customer_type.field("created_at")
def _customer_created(obj: Customer, info):
    return _iso(obj.created_at)


@customer_type.field("projects")
@customer_type.field("configurations")
def _customer_projects(obj: Customer, info):
    return obj.projects


@project_type.field("created_on")
def _project_created(obj: Project, info):
    return _iso(obj.created_at)


@project_type.field("parameters")
def _project_params(obj: Project, info):
    return obj.parameters or {}


@tree_type.field("starting_node")
def _tree_start(obj: Tree, info):
    return _session(info).get(Node, obj.starting_node_id) if obj.starting_node_id else None


@tree_type.field("fields")
def _tree_fields(obj: Tree, info):
    return [f for n in obj.nodes for f in n.fields if f.parent_id is None]


@tree_type.field("projects")
@tree_type.field("configurations")
def _tree_projects(obj: Tree, info):
    return _session(info).scalars(select(Project).where(Project.tree_id == obj.id)).all()


@tree_type.field("created_at")
def _tree_created(obj: Tree, info):
    return _iso(obj.created_at)


@tree_type.field("updated_at")
def _tree_updated(obj: Tree, info):
    return _iso(obj.updated_at)


@tree_type.field("parameters")
def _tree_params(obj: Tree, info):
    return obj.parameters or {}


@tree_type.field("layout")
def _tree_layout(obj: Tree, info):
    return obj.layout or {}


@node_type.field("fields")
def _node_fields(obj: Node, info):
    return [f for f in obj.fields if f.parent_id is None]


@node_type.field("is_start")
def _node_is_start(obj: Node, info):
    return obj.tree.starting_node_id == obj.id


@node_type.field("parameters")
def _node_params(obj: Node, info):
    return obj.parameters or {}


@branch_type.field("parent_node")
def _branch_parent(obj: Branch, info):
    return _session(info).get(Node, obj.parent_node_id)


@branch_type.field("child_node")
def _branch_child(obj: Branch, info):
    return _session(info).get(Node, obj.child_node_id)


_DISPLAY = {
    "string": "text",
    "number": "number",
    "currency": "currency",
    "date": "date",
    "boolean": "checkbox",
    "table": "multiselect",
}


@field_type.field("field_type")
def _field_type(obj: Field, info):
    return {"name": obj.field_type, "display_type": _DISPLAY.get(obj.field_type, "text")}


@field_type.field("child_fields")
def _field_children(obj: Field, info):
    return sorted(
        (f for f in obj.node.fields if f.parent_id == obj.id), key=lambda f: f.sort_order
    )


@field_type.field("parent")
def _field_parent(obj: Field, info):
    return _session(info).get(Field, obj.parent_id) if obj.parent_id else None


@mlmodel_type.field("parameters")
def _model_params(obj: MlModel, info):
    return obj.parameters or {}


# ------------------------------------------------------------------ queries


@query.field("customer")
def q_customer(_, info, id):
    return _session(info).get(Customer, id)


@query.field("customers")
def q_customers(_, info):
    return _session(info).scalars(select(Customer).order_by(Customer.name)).all()


@query.field("project")
@query.field("configuration")
def q_project(_, info, id):
    return _session(info).get(Project, id)


@query.field("projects")
@query.field("configurations")
def q_projects(_, info, customer_id=None):
    stmt = select(Project).order_by(Project.created_at)
    if customer_id:
        stmt = stmt.where(Project.customer_id == customer_id)
    return _session(info).scalars(stmt).all()


@query.field("tree")
def q_tree(_, info, id):
    return graph.load(_session(info), id)


@query.field("trees")
def q_trees(_, info):
    return _session(info).scalars(select(Tree).order_by(Tree.name)).all()


@query.field("node")
def q_node(_, info, id):
    return _session(info).get(Node, id)


@query.field("nodes")
def q_nodes(_, info, tree_id=None):
    stmt = select(Node).order_by(Node.tree_id, Node.sort_order)
    if tree_id:
        stmt = stmt.where(Node.tree_id == tree_id)
    return _session(info).scalars(stmt).all()


@query.field("branch")
def q_branch(_, info, id):
    return _session(info).get(Branch, id)


@query.field("branches")
def q_branches(_, info, tree_id=None):
    stmt = select(Branch).order_by(Branch.tree_id, Branch.sort_order)
    if tree_id:
        stmt = stmt.where(Branch.tree_id == tree_id)
    return _session(info).scalars(stmt).all()


@query.field("field")
def q_field(_, info, id):
    return _session(info).get(Field, id)


@query.field("mlmodel")
def q_mlmodel(_, info, id):
    return _session(info).get(MlModel, id)


@query.field("mlmodels")
def q_mlmodels(_, info):
    return _session(info).scalars(select(MlModel).order_by(MlModel.name)).all()


@query.field("projectConfig")
def q_project_config(_, info, project_id):
    session = _session(info)
    project = session.get(Project, project_id)
    if project is None:
        return None
    if not project.tree_id:
        raise _invalid("project has no decision tree assigned")
    tree = graph.load(session, project.tree_id)
    return {
        "project": {
            "id": project.id,
            "name": project.name,
            "customer_id": project.customer_id,
            "tree_id": project.tree_id,
            "parameters": project.parameters or {},
        },
        "customer": {"id": project.customer.id, "name": project.customer.name},
        "tree": graph.serialize(session, tree).model_dump(mode="json"),
        "mlmodels": {
            m.id: MlModelOut.model_validate(m).model_dump()
            for m in graph.referenced_models(session, tree)
        },
    }


@query.field("meta")
def q_meta(_, info):
    return rest_meta(info.context["settings"])


# ------------------------------------------------------------------ mutations: customers


@mutation.field("createCustomer")
@mutating
def m_create_customer(_, info, input):
    name = (input.get("name") or "").strip()
    if not name:
        raise ValueError("name cannot be blank")
    session = _session(info)
    customer = Customer(name=name, description=input.get("description"))
    session.add(customer)
    session.flush()
    from app.tenants import ensure_tenant_defaults  # noqa: PLC0415

    ensure_tenant_defaults(session, customer.id)
    return customer


@mutation.field("updateCustomer")
@mutating
def m_update_customer(_, info, id, input):
    customer = _get(_session(info), Customer, id, "customer")
    if input.get("name") is not None:
        name = input["name"].strip()
        if not name:
            raise ValueError("name cannot be blank")
        customer.name = name
    if "description" in input:
        customer.description = input["description"]
    return customer


@mutation.field("deleteCustomer")
@mutating
def m_delete_customer(_, info, id):
    _session(info).delete(_get(_session(info), Customer, id, "customer"))
    return {"id": id}


# ------------------------------------------------------------------ mutations: projects


def _check_tree_id(session: Session, tree_id: str | None) -> None:
    if tree_id and session.get(Tree, tree_id) is None:
        raise ValueError(f"tree {tree_id} not found")


@mutation.field("createProject")
@mutating
def m_create_project(_, info, input):
    session = _session(info)
    _get(session, Customer, input["customer_id"], "customer")
    _check_tree_id(session, input.get("tree_id"))
    project = Project(
        customer_id=input["customer_id"],
        name=input["name"],
        description=input.get("description"),
        tree_id=input.get("tree_id"),
        parameters=input.get("parameters") or {},
    )
    session.add(project)
    return project


@mutation.field("updateProject")
@mutating
def m_update_project(_, info, id, input):
    session = _session(info)
    project = _get(session, Project, id, "project")
    if "tree_id" in input:
        _check_tree_id(session, input["tree_id"])
    for key in ("name", "description", "tree_id", "parameters"):
        if key in input:
            val = input[key]
            if key == "name":
                if not val or not str(val).strip():
                    raise ValueError("name cannot be blank")
                val = str(val).strip()
            elif key == "parameters" and val is None:
                raise ValueError("parameters cannot be null")
            setattr(project, key, val)
    return project


@mutation.field("deleteProject")
@mutating
def m_delete_project(_, info, id):
    _session(info).delete(_get(_session(info), Project, id, "project"))
    return {"id": id}


# ------------------------------------------------------------------ mutations: ML models


@mutation.field("createMlModel")
@mutating
def m_create_model(_, info, input):
    params = validate_model_params(
        input.get("parameters"), info.context["settings"].supported_model_types
    )
    model = MlModel(
        name=input["name"],
        type=input["type"],
        parameters=params,
        standard=input.get("standard", True),
    )
    _session(info).add(model)
    return model


@mutation.field("updateMlModel")
@mutating
def m_update_model(_, info, id, input):
    model = _get(_session(info), MlModel, id, "ML model")
    if "parameters" in input:
        model.parameters = validate_model_params(
            input["parameters"], info.context["settings"].supported_model_types
        )
    for key in ("name", "type", "standard"):
        if input.get(key) is not None:
            setattr(model, key, input[key])
    return model


@mutation.field("deleteMlModel")
@mutating
def m_delete_model(_, info, id):
    session = _session(info)
    model = _get(session, MlModel, id, "ML model")
    used = session.scalars(select(Node.name).where(Node.mlmodel_id == id)).all()
    if used:
        raise ValueError(f"model is used by node(s): {', '.join(used)}")
    session.delete(model)
    return {"id": id}


# ------------------------------------------------------------------ mutations: trees


def _field_graph(raw: dict) -> FieldGraph:
    return FieldGraph(
        name=raw["name"],
        customer_key=raw["customer_key"],
        description=raw.get("description") or "",
        field_type=raw.get("field_type") or "string",
        threshold=raw.get("threshold") if raw.get("threshold") is not None else 0.5,
        child_fields=[_field_graph(c) for c in raw.get("child_fields") or []],
    )


def _node_graph(raw: dict) -> NodeGraph:
    return NodeGraph(
        id=raw.get("id") or str(uuid.uuid4()),
        name=raw["name"],
        type=raw["type"],
        description=raw.get("description"),
        mlmodel_id=raw.get("mlmodel_id"),
        parameters=raw.get("parameters") or {},
        fields=[_field_graph(f) for f in raw.get("fields") or []],
    )


def _branch_graph(raw: dict) -> BranchGraph:
    return BranchGraph(
        parent_node_id=raw["parent_node_id"],
        child_node_id=raw["child_node_id"],
        name=raw.get("name"),
        condition=raw.get("condition"),
    )


def _tree_input(raw: dict, current: TreeGraphIn | None = None) -> TreeGraphIn:
    """Build a full tree from create/update input; update keeps fields that were not sent."""
    base = (
        current.model_dump()
        if current
        else {
            "name": "",
            "description": None,
            "business_context": None,
            "starting_node_id": None,
            "parameters": {},
            "layout": {},
            "nodes": [],
            "branches": [],
        }
    )
    for key in ("name", "parameters", "layout"):
        if key in raw and raw[key] is not None:
            base[key] = raw[key]
    for key in ("description", "business_context", "starting_node_id"):
        if key in raw:
            base[key] = raw[key]
    g = TreeGraphIn.model_validate(base)
    if raw.get("nodes") is not None:
        g.nodes = [_node_graph(n) for n in raw["nodes"]]
        starts = [
            g.nodes[i].id for i, n in enumerate(raw["nodes"]) if n.get("start_node")
        ]
        if starts:
            g.starting_node_id = starts[0]
        elif not g.nodes:
            g.starting_node_id = None
        elif (
            not raw.get("starting_node_id")
            and not any(n.id == g.starting_node_id for n in g.nodes)
        ):
            g.starting_node_id = g.nodes[0].id
    if raw.get("branches") is not None:
        g.branches = [_branch_graph(b) for b in raw["branches"]]
    return g


def _current(session: Session, tree: Tree) -> TreeGraphIn:
    return TreeGraphIn.model_validate(graph.serialize(session, tree).model_dump())


def _save(session: Session, tree: Tree, g: TreeGraphIn) -> Tree:
    graph.save(session, tree, g)
    session.flush()
    session.expire_all()
    return graph.load(session, tree.id)


@mutation.field("createTree")
@mutating
def m_create_tree(_, info, input):
    session = _session(info)
    g = _tree_input(input)
    tree = Tree(name=g.name)
    session.add(tree)
    session.flush()
    return _save(session, tree, g)


@mutation.field("updateTree")
@mutating
def m_update_tree(_, info, id, input):
    session = _session(info)
    tree = _load_tree(session, id)
    return _save(session, tree, _tree_input(input, _current(session, tree)))


@mutation.field("deleteTree")
@mutating
def m_delete_tree(_, info, id):
    session = _session(info)
    tree = _get(session, Tree, id, "tree")
    if session.scalars(select(Project.id).where(Project.tree_id == id)).first():
        raise ValueError("tree is used by a project")
    session.delete(tree)
    return {"id": id}


@mutation.field("duplicateTree")
@mutating
def m_duplicate_tree(_, info, id):
    session = _session(info)
    tree = _load_tree(session, id)
    src = _current(session, tree)
    id_map = {n.id: str(uuid.uuid4()) for n in src.nodes}
    src.name = f"{src.name} (copy)"
    for n in src.nodes:
        n.id = id_map[n.id]
    for b in src.branches:
        b.id = None
        b.parent_node_id, b.child_node_id = id_map[b.parent_node_id], id_map[b.child_node_id]
    src.starting_node_id = id_map.get(src.starting_node_id or "")
    positions = (src.layout or {}).get("positions") or {}
    if not isinstance(positions, dict):
        positions = {}
    src.layout = {"positions": {id_map[k]: v for k, v in positions.items() if k in id_map}}
    copy = Tree(name=src.name, customer_id=tree.customer_id)
    session.add(copy)
    session.flush()
    return _save(session, copy, src)


@mutation.field("validateTree")
def m_validate_tree(_, info, input):
    try:
        graph.validate(_tree_input(input), _session(info))
    except graph.TreeValidationError as exc:
        return {"valid": False, "errors": exc.errors}
    except ValidationError as exc:
        return {"valid": False, "errors": [e["msg"] for e in exc.errors()]}
    return {"valid": True, "errors": []}


@mutation.field("importTree")
@mutating
def m_import_tree(_, info, export, name=None, branches=None, starting_node=None):
    tree, report = importer.import_tree(
        _session(info), export, name=name, branches=branches, starting_node=starting_node
    )
    return {"tree": graph.load(_session(info), tree.id), **report}


# ------------------------------------------------------------------ mutations: nodes


def _node_tree(session: Session, node_id: str) -> tuple[Tree, TreeGraphIn]:
    node = _get(session, Node, node_id, "node")
    tree = graph.load(session, node.tree_id)
    return tree, _current(session, tree)


@mutation.field("createNode")
@mutating
def m_create_node(_, info, tree_id, input):
    session = _session(info)
    tree = _load_tree(session, tree_id)
    g = _current(session, tree)
    node = _node_graph(input)
    g.nodes.append(node)
    if input.get("start_node") or not g.starting_node_id:
        g.starting_node_id = node.id
    _save(session, tree, g)
    return session.get(Node, node.id)


@mutation.field("updateNode")
@mutating
def m_update_node(_, info, id, input):
    session = _session(info)
    tree, g = _node_tree(session, id)
    target = next(n for n in g.nodes if n.id == id)
    for key in ("name", "type", "description", "mlmodel_id", "parameters"):
        if key in input:
            val = input[key]
            if key in ("name", "type") and (val is None or not str(val).strip()):
                raise ValueError(f"{key} cannot be blank")
            setattr(target, key, val if key != "parameters" else (val or {}))
    if input.get("fields") is not None:
        target.fields = [_field_graph(f) for f in input["fields"]]
    if input.get("start_node"):
        g.starting_node_id = id
        g.branches = [b for b in g.branches if b.child_node_id != id]
    _save(session, tree, g)
    return session.get(Node, id)


@mutation.field("deleteNode")
@mutating
def m_delete_node(_, info, id):
    session = _session(info)
    tree, g = _node_tree(session, id)
    g.nodes = [n for n in g.nodes if n.id != id]
    g.branches = [b for b in g.branches if id not in (b.parent_node_id, b.child_node_id)]
    if g.starting_node_id == id:
        child_ids = {b.child_node_id for b in g.branches}
        roots = [n.id for n in g.nodes if n.id not in child_ids]
        g.starting_node_id = roots[0] if roots else (g.nodes[0].id if g.nodes else None)
    _save(session, tree, g)
    return {"id": id}


# ------------------------------------------------------------------ mutations: branches


@mutation.field("createBranch")
@mutating
def m_create_branch(_, info, tree_id, input):
    session = _session(info)
    tree = _load_tree(session, tree_id)
    g = _current(session, tree)
    branch = _branch_graph(input)
    branch.id = str(uuid.uuid4())
    g.branches.append(branch)
    _save(session, tree, g)
    return session.get(Branch, branch.id)


def _branch_tree(session: Session, branch_id: str) -> tuple[Tree, TreeGraphIn]:
    branch = _get(session, Branch, branch_id, "branch")
    tree = graph.load(session, branch.tree_id)
    return tree, _current(session, tree)


@mutation.field("updateBranch")
@mutating
def m_update_branch(_, info, id, input):
    session = _session(info)
    tree, g = _branch_tree(session, id)
    target = next(b for b in g.branches if b.id == id)
    for key in ("parent_node_id", "child_node_id", "name", "condition"):
        if key in input and (input[key] is not None or key in ("name", "condition")):
            setattr(target, key, input[key])
    _save(session, tree, g)
    return session.get(Branch, id)


@mutation.field("deleteBranch")
@mutating
def m_delete_branch(_, info, id):
    session = _session(info)
    tree, g = _branch_tree(session, id)
    g.branches = [b for b in g.branches if b.id != id]
    _save(session, tree, g)
    return {"id": id}


OBJECT_TYPES = [
    query,
    mutation,
    json_scalar,
    customer_type,
    project_type,
    tree_type,
    node_type,
    branch_type,
    field_type,
    mlmodel_type,
]
