from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app import conditions
from app import db as db_module
from app.config import Settings, get_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/dt.db",
        seed_admin_password="admin-test-pw",
        session_secret="test-secret",
    )
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    from app.main import app

    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        token = c.post(
            "/auth/login", json={"username": "admin", "password": "admin-test-pw"}
        ).json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c
    app.dependency_overrides.clear()


def _project(client, customer, project):
    customers = {c["name"]: c for c in client.get("/v1/customers").json()}
    projects = client.get(f"/v1/customers/{customers[customer]['id']}/projects").json()
    return next(p for p in projects if p["name"] == project)


def test_seed_creates_example_projects(client):
    names = sorted(c["name"] for c in client.get("/v1/customers").json())
    assert names == ["Demo", "Frontline Insurance", "Selective"]

    p = _project(client, "Selective", "Loss Runs")
    config = client.get(f"/v1/projects/{p['id']}/config").json()
    tree = config["tree"]
    names = {n["id"]: n["name"] for n in tree["nodes"]}
    assert names[tree["starting_node_id"]] == "Document Type"
    edges = {
        (names[b["parent_node_id"]], names[b["child_node_id"]]): b["condition"]
        for b in tree["branches"]
    }
    assert edges[("Document Type", "Loss Runs Policy Table")] == 'document_type == "Loss Runs"'
    claims = next(n for n in tree["nodes"] if n["name"] == "Loss Runs Claims Table")
    table = claims["fields"][0]
    assert table["field_type"] == "table" and len(table["child_fields"]) == 29
    # Baseten / Gemini models are not supported: those nodes run on the catalog GPT-4.1
    models = {m["id"]: m for m in config["mlmodels"].values()}
    assert models[claims["mlmodel_id"]]["name"] == "GPT-4.1"
    for m in models.values():
        assert m["parameters"]["model_type"] in ("azure_openai", "openai", "ollama")
        assert "api_key" not in m["parameters"]


def test_seed_mortgage_change_project(client):
    p = _project(client, "Frontline Insurance", "Mortgage Change")
    assert p["tree_name"] == "Frontline - Mortgage Change"
    assert p["parameters"]["doc_processing_project_id"] == "1514"
    assert "api_key" not in p["parameters"]
    config = client.get(f"/v1/projects/{p['id']}/config").json()
    tree = config["tree"]
    assert [n["name"] for n in tree["nodes"]] == ["Classifier"]
    node = tree["nodes"][0]
    assert tree["starting_node_id"] == node["id"]
    assert node["fields"][0]["customer_key"] == "document_type"
    assert "Mortgage Change" in node["parameters"]["prompt_template"]["system"]
    model = config["mlmodels"][node["mlmodel_id"]]
    assert model["parameters"]["model_name"] == "gpt-5.6-terra"
    assert model["parameters"]["fallback_mlmodel_id"] is None  # Gemini fallback not imported


def test_projects_have_integer_platform_ids(client):
    """Automation projects configure a v1-style integer project id (PLATFORM_DOC_PROJECT_ID)."""
    frontline = _project(client, "Frontline Insurance", "Mortgage Change")
    assert frontline["platform_id"] == 1514  # the imported v1 configuration's id
    assert client.get("/v1/projects/1514").json()["id"] == frontline["id"]
    ids = [p["platform_id"] for p in client.get("/v1/projects").json()]
    assert all(isinstance(i, int) for i in ids) and len(set(ids)) == len(ids)

    customer = client.get("/v1/customers").json()[0]["id"]
    a = client.post(f"/v1/customers/{customer}/projects", json={"name": "A"}).json()
    b = client.post(f"/v1/customers/{customer}/projects", json={"name": "B"}).json()
    assert b["platform_id"] == a["platform_id"] + 1 and a["platform_id"] > max(ids)
    assert client.get(f"/v1/projects/{b['platform_id']}").json()["id"] == b["id"]
    # An imported v1 id that is already taken is not reused.
    c = client.post(
        f"/v1/customers/{customer}/projects",
        json={"name": "C", "parameters": {"doc_processing_project_id": "1514"}},
    ).json()
    assert c["platform_id"] == b["platform_id"] + 1
    assert client.get("/v1/projects/999999").status_code == 404


def test_model_catalog(client):
    models = {m["name"]: m for m in client.get("/v1/mlmodels").json()}
    assert {"GPT-4.1", "GPT-4.1 mini", "GPT-5 mini (reasoning)"} <= set(models)
    assert models["Ollama · qwen3:8b"]["parameters"]["model_type"] == "ollama"
    assert client.get("/v1/meta").json()["supported_model_types"] == [
        "azure_openai",
        "openai",
        "ollama",
    ]


def test_unsupported_model_type_rejected(client):
    r = client.post(
        "/v1/mlmodels",
        json={
            "name": "Gemini",
            "type": "vertex_ai",
            "parameters": {"model_type": "vertex_ai", "model_name": "gemini"},
        },
    )
    assert r.status_code == 422 and "model_type" in r.text
    r = client.post(
        "/v1/mlmodels",
        json={
            "name": "Local",
            "type": "ollama",
            "parameters": {"model_type": "ollama", "model_name": "llama3.2:3b"},
        },
    )
    assert r.status_code == 201


def _simple_tree(model_id=None):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    return {
        "name": "Invoices",
        "starting_node_id": a,
        "layout": {"positions": {a: {"x": 0, "y": 0}}},
        "nodes": [
            {
                "id": a,
                "name": "Doc Type",
                "type": "gpt_node",
                "mlmodel_id": model_id,
                "parameters": {
                    "prompt_template": {
                        "system": "Classify.\n{extraction_fields}",
                        "human": "{text}",
                    }
                },
                "fields": [{"name": "Document Type", "customer_key": "document_type"}],
            },
            {
                "id": b,
                "name": "Lines",
                "type": "code_node",
                "parameters": {
                    "function": "tables.chunking.process",
                    "prompt_format": "list",
                    "prompt_template": {"system": "{extraction_fields}", "human": "{text}"},
                },
                "fields": [
                    {
                        "name": "Lines",
                        "customer_key": "lines",
                        "field_type": "table",
                        "child_fields": [
                            {"name": "Amount", "customer_key": "amount", "field_type": "currency"},
                            {"name": "Date", "customer_key": "date", "field_type": "date"},
                        ],
                    }
                ],
            },
        ],
        "branches": [
            {"parent_node_id": a, "child_node_id": b, "condition": 'document_type == "Invoice"'}
        ],
    }


def test_tree_crud_round_trip(client):
    model = client.post(
        "/v1/mlmodels",
        json={
            "name": "Azure 4.1",
            "type": "gpt",
            "parameters": {"model_type": "azure_openai", "model_name": "gpt-4.1"},
        },
    ).json()
    created = client.post("/v1/trees", json=_simple_tree(model["id"]))
    assert created.status_code == 201, created.text
    tree = created.json()
    assert tree["version"] == 1 and len(tree["nodes"]) == 2
    assert tree["nodes"][1]["fields"][0]["child_fields"][0]["customer_key"] == "amount"

    tree["name"] = "Invoices v2"
    tree["nodes"][0]["parameters"]["prompt_template"]["system"] = "Changed {extraction_fields}"
    updated = client.put(f"/v1/trees/{tree['id']}", json=tree)
    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["version"] == 2 and body["name"] == "Invoices v2"
    assert body["nodes"][0]["id"] == tree["nodes"][0]["id"]  # ids survive a save
    assert body["nodes"][0]["parameters"]["prompt_template"]["system"].startswith("Changed")

    customer = client.post("/v1/customers", json={"name": "Acme"}).json()
    project = client.post(
        f"/v1/customers/{customer['id']}/projects", json={"name": "AP", "tree_id": tree["id"]}
    ).json()
    config = client.get(f"/v1/projects/{project['id']}/config").json()
    assert list(config["mlmodels"]) == [model["id"]]

    assert client.delete(f"/v1/trees/{tree['id']}").status_code == 409  # in use
    assert client.delete(f"/v1/mlmodels/{model['id']}").status_code == 409

    dup = client.post(f"/v1/trees/{tree['id']}/duplicate").json()
    assert dup["name"] == "Invoices v2 (copy)" and dup["nodes"][0]["id"] != tree["nodes"][0]["id"]


def test_tree_validation_errors(client):
    bad = _simple_tree()
    bad["branches"][0]["condition"] = "document_type = Invoice"
    bad["nodes"][1]["parameters"]["function"] = "nope"
    bad["nodes"][1]["fields"][0]["child_fields"] = []
    r = client.post("/v1/trees", json=bad)
    assert r.status_code == 422
    errors = " | ".join(r.json()["detail"]["errors"])
    assert "condition" in errors and "function" in errors and "column" in errors

    cyc = _simple_tree()
    a, b = cyc["nodes"][0]["id"], cyc["nodes"][1]["id"]
    cyc["branches"].append({"parent_node_id": b, "child_node_id": a})
    r = client.post("/v1/trees/validate", json=cyc).json()
    assert not r["valid"]


def test_chunking_belongs_to_the_table_code_node(client):
    # Older Agent Hub keys on the chunking code node become v1's chunk_params.
    tree = _simple_tree()
    tree["nodes"][1]["parameters"].update(size_to_chunk=10, chunk_size=3)
    created = client.post("/v1/trees", json=tree)
    assert created.status_code == 201, created.text
    params = created.json()["nodes"][1]["parameters"]
    assert params["chunk_params"] == {"pdf_size_to_chunk": 10, "pdf_chunk_size": 3}
    assert "size_to_chunk" not in params and "chunk_size" not in params

    # A gpt_node cannot chunk; chunk_params must be positive integers.
    bad = _simple_tree()
    bad["nodes"][0]["parameters"]["chunk_params"] = {"pdf_chunk_size": 2}
    bad["nodes"][1]["parameters"]["chunk_params"] = {"pdf_chunk_size": 0}
    r = client.post("/v1/trees", json=bad)
    assert r.status_code == 422
    errors = " | ".join(r.json()["detail"]["errors"])
    assert "runs only in a code_node" in errors and "positive integers" in errors


def test_v1_configuration_layout_has_start_and_end_nodes(client):
    project = client.get(f"/v1/igpt/configurations/{DEMO_MORTGAGE}").json()["configuration"]
    tree = project["tree"]
    nodes = {n["id"]: n for n in tree["layout"]["nodes"]}
    kinds = [n["type"] for n in nodes.values()]
    assert kinds.count("start_node") == 1
    start = next(n for n in nodes.values() if n["type"] == "start_node")
    assert start["data"]["name"] == "Start" and start["id"] != tree["starting_node"]["id"]
    first = next(e for e in tree["layout"]["edges"] if e["source"] == start["id"])
    assert first["target"] == tree["starting_node"]["id"] and first["logicType"] == "Connection"
    # One Extraction End per leaf (Other, Check, Claim, Returned Mail, Risk).
    ends = [n for n in nodes.values() if n["type"] == "end_node"]
    assert len(ends) == 5 and {n["data"]["name"] for n in ends} == {"Extraction End"}
    by_name = {n["name"]: n["id"] for n in tree["nodes"]}
    out = [e for e in tree["layout"]["edges"] if e["source"] == by_name["Classifier"]]
    assert all(e["logicType"] == "Condition" for e in out)
    assert [e["sourceHandle"] for e in out].count("b") == 1  # first branch right, rest bottom
    risk = next(n for n in tree["nodes"] if n["name"] == "Risk")
    assert risk["type"] == "code_node" and risk["parameters"]["chunk_params"]


def test_import_keeps_v1_canvas_positions(client):
    export = {
        "tree": {
            "name": "Positions",
            "starting_node": {"id": "a"},
            "nodes": [
                {
                    "id": "a",
                    "name": "A",
                    "type": "gpt_node",
                    "parameters": {"prompt_template": {"system": "{extraction_fields}"}},
                    "fields": [{"name": "A", "customer_key": "a"}],
                }
            ],
            "branches": [],
            "layout": {
                "nodes": [
                    {"id": "s", "type": "start_node", "position": {"x": -420, "y": 285}},
                    {"id": "a", "type": "gpt_node", "position": {"x": 15, "y": 285}},
                    {"id": "e", "type": "end_node", "position": {"x": 400, "y": 285}},
                ]
            },
        }
    }
    r = client.post("/v1/trees/import", json=export)
    assert r.status_code == 201, r.text
    tree = client.get(f"/v1/trees/{r.json()['tree_id']}").json()
    positions = tree["layout"]["positions"]
    assert positions["start"] == {"x": -420, "y": 285}
    assert positions[tree["nodes"][0]["id"]] == {"x": 15, "y": 285}
    assert len(positions) == 2


def test_models_reject_secrets(client):
    r = client.post("/v1/mlmodels", json={"name": "x", "parameters": {"api_key": "sk-123"}})
    assert r.status_code == 422


DEMO_CUSTOMER = "de300000-0000-4000-8000-000000000001"
DEMO_MORTGAGE = "de300000-0000-4000-8000-0000000000a1"
DEMO_LOSS_RUNS = "de300000-0000-4000-8000-0000000000a2"


def test_seed_demo_organization_with_fixed_ids(client):
    projects = {p["name"]: p for p in client.get(f"/v1/customers/{DEMO_CUSTOMER}/projects").json()}
    assert projects["Mortgage Change"]["id"] == DEMO_MORTGAGE
    assert projects["Loss Runs"]["id"] == DEMO_LOSS_RUNS
    assert projects["Loss Runs"]["tree_name"] == "Demo - Loss Runs"
    assert projects["Mortgage Change"]["parameters"]["doc_processing_project_id"] == "1444"

    config = client.get(f"/v1/projects/{DEMO_MORTGAGE}/config").json()
    tree = config["tree"]
    names = {n["id"]: n["name"] for n in tree["nodes"]}
    assert names[tree["starting_node_id"]] == "Classifier"
    edges = {
        (names[b["parent_node_id"]], names[b["child_node_id"]]): b["condition"]
        for b in tree["branches"]
    }
    assert edges[("Classifier", "General Fields")] == "document_type == 'Mortgage Change'"
    assert edges[("General Fields", "Risk")] == "True"
    assert edges[("Returned Mail Subdoctype", "Returned Mail")] == "True"

    risk = next(n for n in tree["nodes"] if n["name"] == "Risk")
    assert risk["parameters"]["function"] == "tables.chunking.process"
    assert risk["parameters"]["chunk_params"] == {"pdf_chunk_size": 2, "pdf_size_to_chunk": 4}
    table = risk["fields"][0]
    assert table["customer_key"] == "risk" and table["field_type"] == "table"
    cols = {c["customer_key"]: c for c in table["child_fields"]}
    assert {"policy_num1", "newloannum", "loanNumber", "insuredName", "mortgageRole"} <= set(cols)
    assert cols["policy_num1"]["threshold"] == 0.5
    # confidence models keep logprobs; Gemini (vertex_ai) nodes run on GPT-4.1
    models = config["mlmodels"]
    assert models[risk["mlmodel_id"]]["parameters"]["logprobs"] is True
    classifier = next(n for n in tree["nodes"] if n["name"] == "Classifier")
    assert models[classifier["mlmodel_id"]]["name"] == "GPT-4.1"
    # "placeholder" on Other / Check / Claim: same key on mutually exclusive branches is fine
    assert sum(f["customer_key"] == "placeholder" for n in tree["nodes"] for f in n["fields"]) == 3

    # applied once: a restart does not duplicate the Demo projects
    from app.seed import seed

    seed(Settings().seed_dir, None)
    again = client.get(f"/v1/customers/{DEMO_CUSTOMER}/projects").json()
    assert sorted(p["name"] for p in again) == ["Loss Runs", "Mortgage Change"]


def test_shared_key_on_one_path_is_rejected(client):
    tree = client.get("/v1/trees/" + _project(client, "Demo", "Mortgage Change")["tree_id"]).json()
    risk = next(n for n in tree["nodes"] if n["name"] == "Risk")
    general = next(n for n in tree["nodes"] if n["name"] == "General Fields")
    risk["fields"].append({**general["fields"][0], "id": None})  # Risk runs after General Fields
    body = {k: tree[k] for k in ("name", "starting_node_id", "nodes", "branches")}
    r = client.post("/v1/trees/validate", json=body)
    assert any("which run on the same path" in e for e in r.json()["errors"]), r.text


def test_import_tree_export_reads_branches_from_layout_edges(client):
    node = lambda nid, name, ntype="gpt_node", **params: {  # noqa: E731
        "id": nid,
        "name": name,
        "type": ntype,
        "parameters": {
            **params,
            "prompt_template": {"system": "{extraction_fields}", "human": "{text}"},
        },
        "fields": [{"name": name, "customer_key": f"{nid}_-_key/x", "description": ""}],
    }
    export = {
        "data": {
            "tree": {
                "name": "Platform tree",
                "starting_node": {"id": "a", "name": "A"},
                "nodes": [
                    node("a", "A"),
                    node("b", "B", "code_node", function="sov.process", function_args={"n": 1}),
                ],
                "branches": [{"id": "e1", "condition": "document_type == 'SOV'", "name": "SOV"}],
                "layout": {"edges": [{"id": "e1", "source": "a", "target": "b"}]},
            }
        }
    }
    r = client.post("/v1/trees/import", json=export)
    assert r.status_code == 201, r.text
    assert any("sov.process" in w for w in r.json()["warnings"])
    tree = client.get(f"/v1/trees/{r.json()['tree_id']}").json()
    b = next(n for n in tree["nodes"] if n["name"] == "B")
    assert b["type"] == "gpt_node"
    assert b["parameters"]["original_function"] == "sov.process"
    assert b["parameters"]["original_function_args"] == {"n": 1}
    assert len(tree["branches"]) == 1
    assert tree["branches"][0]["condition"] == "document_type == 'SOV'"


def test_import_endpoint_strips_secrets(client):
    export = {
        "data": {
            "configuration": {
                "name": "Imp",
                "tree": {
                    "nodes": [
                        {
                            "id": "n1",
                            "name": "Only",
                            "type": "gpt_node",
                            "parameters": {
                                "prompt_template": {
                                    "system": "{extraction_fields}",
                                    "human": "{text}",
                                }
                            },
                            "mlmodel": {
                                "id": "m-imp",
                                "name": "Baseten",
                                "type": "finetuned_streaming",
                                "parameters": {"api_key": "SECRET", "model_name": "q"},
                            },
                            "fields": [
                                {
                                    "name": "A",
                                    "customer_key": "a",
                                    "field_type": {"name": "string"},
                                }
                            ],
                        }
                    ]
                },
            }
        }
    }
    r = client.post("/v1/trees/import", json=export)
    assert r.status_code == 201, r.text
    assert r.json()["dropped_secrets"] == [{"model": "Baseten", "keys": ["api_key"]}]
    assert r.json()["remapped_models"] == [
        {"model": "Baseten", "model_type": "finetuned_streaming", "replaced_with": "GPT-4.1"}
    ]
    again = client.post("/v1/trees/import", json=export)  # same node ids: must not collide
    assert again.status_code == 201


@pytest.mark.parametrize(
    "cond, ctx, expected",
    [
        (None, {}, True),
        ("", {}, True),
        ('document_type == "Loss Runs"', {"document_type": " loss runs "}, True),
        ('document_type == "Loss Runs"', {"document_type": "OTHERS"}, False),
        ('document_type != "OTHERS"', {"document_type": "Loss Runs"}, True),
        ('document_type in ["A", "Loss Runs"]', {"document_type": "Loss Runs"}, True),
        ('document_type not in ["A"]', {}, True),
        ('a == "1" and b == "2" or c == "3"', {"c": "3"}, True),
        ('a == "1" and b == "2"', {"a": "1"}, False),
        ('x == "has and inside"', {"x": "has and inside"}, True),
        ("n == 5", {"n": 5}, True),
        ("document_type == 'Loss Runs'", {"document_type": "loss runs"}, True),
        ("document_type in ['ACORD', 'Loss Runs']", {"document_type": "ACORD"}, True),
        ("x == 'has \" and \" inside' or y == 'z'", {"y": "z"}, True),
        ("True", {}, True),
    ],
)
def test_conditions(cond, ctx, expected):
    assert conditions.evaluate(cond, ctx) is expected


@pytest.mark.parametrize("bad", ["document_type = x", "== 'x'", 'a in "x"', "a == [1]"])
def test_condition_parse_errors(bad):
    with pytest.raises(conditions.ConditionError):
        conditions.parse(bad)


def test_health(client, monkeypatch):
    live = client.get("/health/live").json()
    assert live["status"] == "ok" and live["service"] == "agent-hub-api"
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["checks"]["database"]["details"]["trees"] >= 2

    from app import health

    def broken():
        raise RuntimeError("db gone")

    monkeypatch.setattr(health, "session_scope", broken)
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "down"
    assert "db gone" in r.json()["checks"]["database"]["error"]
