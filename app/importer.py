"""Import a Platform-DecisionTree-API export.

Accepted shapes:

* ``{"data": {"configuration": {..., "tree": {"nodes": [...]}}}}``  (configuration query)
* ``{"data": {"tree": {"starting_node": {...}, "nodes": [...]?, "configurations": [...]}}}``
  (tree query; a single-node tree may only carry ``starting_node``)
* the same without the ``data`` wrapper, a bare configuration (has ``tree``), or a bare tree.

Rules:

* ML models are upserted by id. Secret-looking parameters (``api_key``, ``token``, ...) are
  dropped and reported; credentials belong in the extraction service's credentials.yaml.
* Models whose ``model_type`` is not supported here (e.g. ``finetuned_streaming``,
  ``vertex_ai``) are not imported; their nodes use the catalog's GPT-4.1 instead (reported).
* Configuration ``api_key`` values are never read.
* Field types come from the library template (Currency, Number, Date, Text, Table) when present,
  since the platform stores most table columns as ``string`` with the real type on the template.
* Branches and the starting node are imported when present. Tree-query exports often list
  branches without ``parent_node`` / ``child_node``; their endpoints are then read from
  ``layout.edges`` (edge id = branch id, source/target = node ids). Callers may also add
  ``branches`` / ``starting_node`` (by node name) for exports that lack them.
* A ``code_node`` whose function the extraction engine cannot run (e.g. ``sov.process``) is
  imported as a ``gpt_node`` with the same prompt; the original function and its arguments
  are kept under ``parameters.original_function`` / ``original_function_args``, and a warning
  says which node to reconfigure (e.g. as ``tables.chunking.process`` with table columns).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app import graph
from app.catalog import DEFAULT_GPT_MODEL, catalog_id
from app.config import get_settings
from app.models import MlModel, Tree
from app.schemas import CODE_FUNCTIONS, BranchGraph, FieldGraph, NodeGraph, TreeGraphIn

SECRET_KEYS = {"api_key", "apikey", "key", "secret", "token", "password", "access_key"}
CONFIG_FLAGS = (
    "split_package",
    "ocr_page_limit",
    "include_hidden_sheets",
    "split_for_igpt",
    "doc_processing_project_id",
)

_TEMPLATE_TYPES = {
    "currency": "currency",
    "number": "number",
    "date": "date",
    "text": "string",
    "table": "table",
    "boolean": "boolean",
    "checkbox": "boolean",
}
_DATA_TYPES = {"number": "number", "date": "date", "table": "table", "string": "string"}


def _unwrap(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (configuration dict, tree dict)."""
    if not isinstance(payload, dict):
        raise ValueError("not a decision tree export: expected a JSON object")
    if "data" in payload and isinstance(payload["data"], dict):
        payload = payload["data"]
    if "configuration" in payload:
        payload = payload["configuration"]
        if not isinstance(payload, dict):
            raise ValueError("not a decision tree export: configuration must be an object")
    if "tree" in payload:
        config, tree = payload, payload["tree"]
        if not isinstance(tree, dict):
            raise ValueError("not a decision tree export: tree must be an object")
    elif "nodes" in payload or "starting_node" in payload:
        config, tree = {}, payload
    else:
        raise ValueError(
            "not a decision tree export: expected data.configuration.tree or data.tree"
        )
    # Tree-query exports list configurations on the tree instead of wrapping it.
    configs = tree.get("configurations")
    if not config.get("name") and isinstance(configs, list) and configs and isinstance(configs[0], dict):
        config = {**configs[0], **{k: v for k, v in config.items() if k != "tree"}}
    return config, tree


def _raw_nodes(tree: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = list(tree.get("nodes") or [])
    start = tree.get("starting_node")
    if start and start.get("name") and not any(n.get("id") == start.get("id") for n in nodes):
        nodes.insert(0, start)
    return nodes


def _field_type(f: dict[str, Any]) -> str:
    template = ((f.get("library_field") or {}).get("template") or {}).get("name")
    if template and template.lower() in _TEMPLATE_TYPES:
        return _TEMPLATE_TYPES[template.lower()]
    name = ((f.get("field_type") or {}).get("name") or "string").lower()
    return _DATA_TYPES.get(name, "string")


def _sort_key(f: dict[str, Any]) -> float:
    fd = f.get("field_display")
    raw = fd.get("sort_order") if isinstance(fd, dict) else 0
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _fields(raw: list[dict[str, Any]]) -> list[FieldGraph]:
    ordered = sorted((f for f in raw if isinstance(f, dict)), key=_sort_key)
    return [
        FieldGraph(
            name=f.get("name") or f["customer_key"],
            customer_key=f["customer_key"],
            description=f.get("description") or "",
            field_type=_field_type(f),
            threshold=f.get("threshold") if f.get("threshold") is not None else 0.5,
            child_fields=_fields(f.get("child_fields") or []),
        )
        for f in ordered
    ]


def _clean_params(params: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    params = dict(params or {})
    dropped = [k for k in params if k.lower() in SECRET_KEYS]
    for k in dropped:
        params.pop(k)
    return params, dropped


def _model_for(session: Session, model: dict[str, Any], report: dict[str, Any]) -> str | None:
    supported = set(get_settings().supported_model_types)
    params, dropped = _clean_params(model.get("parameters"))
    if dropped:
        report["dropped_secrets"].append({"model": model.get("name"), "keys": dropped})
    mtype = params.get("model_type") or model.get("type")
    if mtype not in supported:
        fallback = session.get(MlModel, catalog_id(DEFAULT_GPT_MODEL))
        report["remapped_models"].append(
            {
                "model": model.get("name"),
                "model_type": mtype,
                "replaced_with": fallback.name if fallback else None,
            }
        )
        return fallback.id if fallback else None
    existing = session.get(MlModel, model["id"]) if model.get("id") else None
    if existing is None:
        existing = MlModel(id=model.get("id"), name=model.get("name") or "model")
        session.add(existing)
    existing.name = model.get("name") or existing.name
    existing.type = model.get("type") or "gpt"
    existing.parameters = params
    existing.standard = bool(model.get("standard", True))
    session.flush()
    if existing.name not in report["models"]:
        report["models"].append(existing.name)
    return existing.id


def _drop_dangling_fallbacks(session: Session, model_ids: set[str]) -> None:
    for mid in model_ids:
        m = session.get(MlModel, mid)
        fb = (m.parameters or {}).get("fallback_mlmodel_id") if m else None
        if fb and session.get(MlModel, fb) is None:
            m.parameters = {**m.parameters, "fallback_mlmodel_id": None}


def import_tree(
    session: Session,
    payload: dict[str, Any],
    *,
    name: str | None = None,
    branches: list[dict[str, str]] | None = None,
    starting_node: str | None = None,
) -> tuple[Tree, dict[str, Any]]:
    config, raw_tree = _unwrap(payload)
    report: dict[str, Any] = {
        "models": [],
        "remapped_models": [],
        "dropped_secrets": [],
        "warnings": [],
    }

    nodes: list[NodeGraph] = []
    id_map: dict[str, str] = {}  # export node id -> new id (imports never reuse node ids)
    model_ids: set[str] = set()
    for raw in _raw_nodes(raw_tree):
        new_id = str(uuid.uuid4())
        if raw.get("id"):
            id_map[raw["id"]] = new_id
        model_id = _model_for(session, raw["mlmodel"], report) if raw.get("mlmodel") else None
        if model_id:
            model_ids.add(model_id)
        node_type = raw.get("type") or "gpt_node"
        params = dict(raw.get("parameters") or {})
        if node_type == "code_node" and params.get("function") not in CODE_FUNCTIONS:
            fn = params.pop("function", None)
            params["original_function"] = fn
            if "function_args" in params:
                params["original_function_args"] = params.pop("function_args")
            for ck in ("chunk_params", "size_to_chunk", "chunk_size"):
                if ck in params:
                    params[f"original_{ck}"] = params.pop(ck)
            node_type = "gpt_node"
            report["warnings"].append(
                f"node {raw['name']!r}: function {fn!r} is not supported; imported as a gpt_node "
                f"(reconfigure it, e.g. as tables.chunking.process with table columns)"
            )
        nodes.append(
            NodeGraph(
                id=new_id,
                name=raw["name"],
                type=node_type,
                description=raw.get("description"),
                mlmodel_id=model_id,
                parameters=params,
                fields=_fields(raw.get("fields") or []),
            )
        )
    _drop_dangling_fallbacks(session, model_ids)

    by_name = {n.name: n.id for n in nodes}
    edges = {
        e.get("id"): e
        for e in ((raw_tree.get("layout") or {}).get("edges") or [])
        if isinstance(e, dict)
    }
    branch_graphs: list[BranchGraph] = []
    for b in raw_tree.get("branches") or []:
        edge = edges.get(b.get("id")) or {}
        parent = (
            (b.get("parent_node") or {}).get("id") or b.get("parent_node_id") or edge.get("source")
        )
        child = (
            (b.get("child_node") or {}).get("id") or b.get("child_node_id") or edge.get("target")
        )
        if not parent or not child:
            report["warnings"].append(
                f"branch {b.get('name') or b.get('id')!r} has no parent/child node; skipped"
            )
            continue
        branch_graphs.append(
            BranchGraph(
                parent_node_id=id_map.get(parent, parent),
                child_node_id=id_map.get(child, child),
                name=b.get("name"),
                condition=b.get("condition"),
            )
        )
    for b in branches or []:
        try:
            branch_graphs.append(
                BranchGraph(
                    parent_node_id=by_name[b["parent"]],
                    child_node_id=by_name[b["child"]],
                    condition=b.get("condition"),
                    name=b.get("name"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"branch refers to unknown node name {exc}") from exc

    start = (raw_tree.get("starting_node") or {}).get("id")
    start = id_map.get(start, start) if start else None
    if starting_node:
        if starting_node not in by_name:
            raise ValueError(f"starting_node {starting_node!r} not found")
        start = by_name[starting_node]
    if not start and nodes:
        children = {b.child_node_id for b in branch_graphs}
        roots = [n.id for n in nodes if n.id not in children]
        start = roots[0] if len(roots) == 1 else None
        if not start:
            report["warnings"].append(
                "no branches/starting node in the export; connect the nodes in the tree editor"
            )
            start = nodes[0].id

    report["configuration"] = {k: config[k] for k in CONFIG_FLAGS if k in config}
    tree_params = dict(config.get("parameters") or {})

    g = TreeGraphIn(
        name=name or config.get("name") or raw_tree.get("name") or "Imported tree",
        description=config.get("description"),
        business_context=raw_tree.get("business_context") or None,
        starting_node_id=start,
        parameters=tree_params,
        layout=_layout(raw_tree, id_map),
        nodes=nodes,
        branches=branch_graphs,
    )
    tree = Tree(name=g.name)
    session.add(tree)
    session.flush()
    graph.save(session, tree, g)
    return tree, report


def _layout(raw_tree: dict[str, Any], id_map: dict[str, str]) -> dict[str, Any]:
    """v1 canvas positions (layout.nodes) keyed by the new node ids; v1's Start node is kept
    under ``start``. Extraction End nodes are drawn by the editor, so they are not stored."""
    positions: dict[str, dict[str, int]] = {}
    for n in (raw_tree.get("layout") or {}).get("nodes") or []:
        pos = n.get("position") if isinstance(n, dict) else None
        if not isinstance(pos, dict) or not all(
            isinstance(pos.get(a), int | float) for a in ("x", "y")
        ):
            continue
        xy = {"x": round(pos["x"]), "y": round(pos["y"])}
        if n.get("id") in id_map:
            positions[id_map[n["id"]]] = xy
        elif n.get("type") == "start_node":
            positions["start"] = xy
    return {"positions": positions} if positions else {}
