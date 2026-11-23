"""Read, validate and save whole decision trees.

The UI edits a tree as one document (nodes + fields + branches + canvas layout) and saves it with
a single PUT. Saving replaces the tree's nodes and branches; ids sent by the client are kept so
branches and layout positions keep pointing at the same nodes.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app import conditions
from app.models import Branch, Field, MlModel, Node, Project, Tree
from app.schemas import (
    CODE_FUNCTIONS,
    FIELD_TYPES,
    NODE_TYPES,
    BranchGraph,
    FieldGraph,
    NodeGraph,
    TreeGraphIn,
    TreeGraphOut,
)

CHUNK_FUNCTION = "tables.chunking.process"
# Agent Hub's earlier per-node chunk keys -> v1 chunk_params keys.
_LEGACY_CHUNK_KEYS = {"size_to_chunk": "pdf_size_to_chunk", "chunk_size": "pdf_chunk_size"}


def normalize_chunking(node: NodeGraph) -> None:
    """Page chunking is a code node's job, like v1: ``tables.chunking.process`` reads
    ``parameters.chunk_params`` {pdf_size_to_chunk, pdf_chunk_size}. Older Agent Hub keys
    (size_to_chunk / chunk_size) on such a node are moved into chunk_params."""
    params = node.parameters
    if params.get("function") != CHUNK_FUNCTION:
        return
    legacy = {k: params.pop(k) for k in list(_LEGACY_CHUNK_KEYS) if k in params}
    if not legacy:
        return
    raw_cp = params.get("chunk_params")
    if raw_cp is not None and not isinstance(raw_cp, dict):
        return  # let validate() report the invalid chunk_params cleanly
    cp = dict(raw_cp or {})
    for old, new in _LEGACY_CHUNK_KEYS.items():
        if legacy.get(old) not in (None, ""):
            cp.setdefault(new, legacy[old])
    params["chunk_params"] = cp


class TreeValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def validate(graph: TreeGraphIn, session: Session, *, tree_id: str | None = None) -> None:
    errors: list[str] = []
    for node in graph.nodes:
        normalize_chunking(node)
    node_ids = [n.id for n in graph.nodes if n.id]
    if len(node_ids) != len(set(node_ids)):
        errors.append("node ids must be unique")
    branch_ids = [b.id for b in graph.branches if b.id]
    if len(branch_ids) != len(set(branch_ids)):
        errors.append("branch ids must be unique")
    if node_ids:
        q = select(Node.id).where(Node.id.in_(node_ids))
        if tree_id:
            q = q.where(Node.tree_id != tree_id)
        for clash in sorted(session.scalars(q)):
            errors.append(f"node id {clash} belongs to another tree")
    if branch_ids:
        q = select(Branch.id).where(Branch.id.in_(branch_ids))
        if tree_id:
            q = q.where(Branch.tree_id != tree_id)
        for clash in sorted(session.scalars(q)):
            errors.append(f"branch id {clash} belongs to another tree")
    names = [n.name for n in graph.nodes]
    if len(names) != len(set(names)):
        errors.append("node names must be unique within a tree")
    ids = set(node_ids)

    model_ids = {n.mlmodel_id for n in graph.nodes if n.mlmodel_id}
    if model_ids:
        found = set(session.scalars(select(MlModel.id).where(MlModel.id.in_(model_ids))))
        for missing in sorted(model_ids - found):
            errors.append(f"unknown ML model {missing}")

    key_nodes: dict[str, list] = defaultdict(list)
    for node in graph.nodes:
        if node.type not in NODE_TYPES:
            errors.append(f"node {node.name!r}: type must be one of {NODE_TYPES}")
        fn = node.parameters.get("function")
        if node.type == "code_node" and fn not in CODE_FUNCTIONS:
            errors.append(
                f"node {node.name!r}: code_node function must be one of {CODE_FUNCTIONS}"
            )
        if node.type != "code_node" and fn:
            errors.append(f"node {node.name!r}: only code_node takes a function")
        if fn in ("verifiers.verify_table.process", "verifiers.verify_business.process") and (
            not node.parameters.get("table_key")
        ):
            errors.append(f"node {node.name!r}: verifier needs parameters.table_key")
        chunk_keys = sorted(
            k for k in ("chunk_params", *_LEGACY_CHUNK_KEYS) if k in node.parameters
        )
        if chunk_keys and fn != CHUNK_FUNCTION:
            errors.append(
                f"node {node.name!r}: page chunking ({', '.join(chunk_keys)}) runs only in a "
                f"code_node with function {CHUNK_FUNCTION}"
            )
        cp = node.parameters.get("chunk_params")
        if cp is not None:
            if not isinstance(cp, dict) or any(
                k not in ("pdf_size_to_chunk", "pdf_chunk_size")
                or isinstance(v, bool)
                or not isinstance(v, int)
                or v < 1
                for k, v in cp.items()
            ):
                errors.append(
                    f"node {node.name!r}: chunk_params takes positive integers "
                    "pdf_size_to_chunk and pdf_chunk_size"
                )
        if fn == "tables.chunking.process" and not any(
            f.field_type == "table" for f in node.fields
        ):
            errors.append(f"node {node.name!r}: table node needs a field of type 'table'")
        for f in node.fields:
            _validate_field(node.name, f, errors, top_level=True)
            key_nodes[f.customer_key].append(node)

    parents: dict[str, str] = {}
    children = defaultdict(list)
    for b in graph.branches:
        if b.parent_node_id not in ids or b.child_node_id not in ids:
            errors.append(f"branch {b.parent_node_id}->{b.child_node_id} references unknown node")
            continue
        if b.parent_node_id == b.child_node_id:
            errors.append("a branch cannot point a node at itself")
        if b.child_node_id in parents:
            errors.append(f"node {b.child_node_id} has more than one parent branch")
        parents[b.child_node_id] = b.parent_node_id
        children[b.parent_node_id].append(b.child_node_id)
        try:
            conditions.parse(b.condition)
        except conditions.ConditionError as exc:
            errors.append(f"branch condition: {exc}")

    # A customer_key must identify one value per document: two nodes may share a key only when
    # they sit on different branches (e.g. "placeholder" on Other / Check / Claim), never when
    # one runs after the other on the same path.
    def ancestors(nid: str) -> set[str]:
        seen: set[str] = set()
        while nid in parents and parents[nid] not in seen:
            nid = parents[nid]
            seen.add(nid)
        return seen

    for key, nodes_with_key in key_nodes.items():
        for i, a in enumerate(nodes_with_key):
            for b in nodes_with_key[i + 1 :]:
                same_node = a is b or (a.id is not None and a.id == b.id)
                same_path = (
                    a.id is not None
                    and b.id is not None
                    and (a.id in ancestors(b.id) or b.id in ancestors(a.id))
                )
                if same_node or same_path:
                    errors.append(
                        f"customer_key {key!r} is defined on both {a.name!r} and {b.name!r}"
                        + ("" if same_node else ", which run on the same path")
                    )

    if graph.starting_node_id:
        if graph.starting_node_id not in ids:
            errors.append("starting_node_id is not a node of this tree")
        elif graph.starting_node_id in parents:
            errors.append("the starting node cannot have a parent branch")
    elif graph.nodes:
        errors.append("choose a starting node")

    # cycle check (each node has at most one parent, so walk up from every node)
    for nid in ids:
        seen, cur = set(), nid
        while cur in parents:
            if cur in seen:
                errors.append("branches form a cycle")
                break
            seen.add(cur)
            cur = parents[cur]
        else:
            continue
        break

    if errors:
        raise TreeValidationError(errors)


def _validate_field(node: str, f: FieldGraph, errors: list[str], top_level: bool) -> None:
    if f.field_type not in FIELD_TYPES:
        errors.append(f"{node}/{f.name}: field_type must be one of {FIELD_TYPES}")
    if f.field_type == "table":
        if not top_level:
            errors.append(f"{node}/{f.name}: tables cannot be nested")
        if not f.child_fields:
            errors.append(f"{node}/{f.name}: table needs at least one column")
        keys = [c.customer_key for c in f.child_fields]
        if len(keys) != len(set(keys)):
            errors.append(f"{node}/{f.name}: column customer_keys must be unique")
        for c in f.child_fields:
            _validate_field(node, c, errors, top_level=False)
    elif f.child_fields:
        errors.append(f"{node}/{f.name}: only table fields have columns")


def _new_id(value: str | None) -> str:
    return value or str(uuid.uuid4())


def save(session: Session, tree: Tree, graph: TreeGraphIn) -> Tree:
    """Replace ``tree``'s content with ``graph`` (which must already be validated)."""
    # Assign ids to new nodes first so branches can refer to them.
    for node in graph.nodes:
        node.id = _new_id(node.id)
    validate(graph, session, tree_id=tree.id)

    tree.name = graph.name
    tree.description = graph.description
    tree.business_context = graph.business_context
    tree.starting_node_id = graph.starting_node_id
    tree.parameters = graph.parameters
    tree.layout = graph.layout
    tree.version = (tree.version or 0) + 1

    # Delete then flush before re-inserting so client-kept ids don't collide.
    tree.nodes.clear()
    tree.branches.clear()
    session.flush()

    for i, ng in enumerate(graph.nodes):
        node = Node(
            id=ng.id,
            name=ng.name,
            type=ng.type,
            description=ng.description,
            mlmodel_id=ng.mlmodel_id,
            parameters=ng.parameters,
            sort_order=i,
        )
        node.fields = list(_build_fields(ng.id, ng.fields, parent_id=None))
        tree.nodes.append(node)
    for i, bg in enumerate(graph.branches):
        tree.branches.append(
            Branch(
                id=_new_id(bg.id),
                parent_node_id=bg.parent_node_id,
                child_node_id=bg.child_node_id,
                name=bg.name,
                condition=(bg.condition or "").strip() or None,
                sort_order=i,
            )
        )
    session.flush()
    return tree


def _build_fields(node_id: str, fields: list[FieldGraph], parent_id: str | None):
    for i, fg in enumerate(fields):
        fid = str(uuid.uuid4())  # field ids are regenerated; customer_key is the stable handle
        yield Field(
            id=fid,
            node_id=node_id,
            parent_id=parent_id,
            name=fg.name,
            customer_key=fg.customer_key,
            description=fg.description,
            field_type=fg.field_type,
            threshold=fg.threshold,
            sort_order=i,
        )
        yield from _build_fields(node_id, fg.child_fields, parent_id=fid)


def load(session: Session, tree_id: str) -> Tree | None:
    return session.scalar(
        select(Tree)
        .where(Tree.id == tree_id)
        .options(selectinload(Tree.nodes).selectinload(Node.fields), selectinload(Tree.branches))
    )


def serialize(session: Session, tree: Tree, *, principal=None) -> TreeGraphOut:
    projects_q = select(Project.id, Project.name, Project.customer_id).where(
        Project.tree_id == tree.id
    )
    if principal is not None and not principal.reads_all:
        projects_q = projects_q.where(Project.customer_id.in_(principal.customer_ids or []))
    projects = session.execute(projects_q).all()
    return TreeGraphOut(
        id=tree.id,
        customer_id=tree.customer_id,
        name=tree.name,
        description=tree.description,
        business_context=tree.business_context,
        starting_node_id=tree.starting_node_id,
        parameters=tree.parameters or {},
        layout=tree.layout or {},
        version=tree.version,
        created_at=tree.created_at,
        updated_at=tree.updated_at,
        nodes=[_node_out(n) for n in tree.nodes],
        branches=[
            BranchGraph(
                id=b.id,
                parent_node_id=b.parent_node_id,
                child_node_id=b.child_node_id,
                name=b.name,
                condition=b.condition,
            )
            for b in tree.branches
        ],
        projects=[{"id": p.id, "name": p.name, "customer_id": p.customer_id} for p in projects],
    )


def _node_out(node: Node) -> NodeGraph:
    by_parent: dict[str | None, list[Field]] = defaultdict(list)
    for f in node.fields:
        by_parent[f.parent_id].append(f)

    def build(parent: str | None) -> list[FieldGraph]:
        return [
            FieldGraph(
                id=f.id,
                name=f.name,
                customer_key=f.customer_key,
                description=f.description,
                field_type=f.field_type,
                threshold=f.threshold,
                child_fields=build(f.id),
            )
            for f in sorted(by_parent[parent], key=lambda x: x.sort_order)
        ]

    return NodeGraph(
        id=node.id,
        name=node.name,
        type=node.type,
        description=node.description,
        mlmodel_id=node.mlmodel_id,
        parameters=node.parameters or {},
        fields=build(None),
    )


def referenced_models(session: Session, tree: Tree) -> list[MlModel]:
    """Models used by the tree's nodes plus their fallback chain."""
    pending = {n.mlmodel_id for n in tree.nodes if n.mlmodel_id}
    seen: dict[str, MlModel] = {}
    while pending:
        batch = session.scalars(select(MlModel).where(MlModel.id.in_(pending))).all()
        pending = set()
        for m in batch:
            seen[m.id] = m
            fb = (m.parameters or {}).get("fallback_mlmodel_id")
            if fb and fb not in seen:
                pending.add(fb)
    return list(seen.values())
