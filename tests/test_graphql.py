from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app.config import Settings, get_settings


@pytest.fixture
def gql(tmp_path, monkeypatch):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/g.db", api_key=SERVICE_KEY)
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.gql.get_settings", lambda: settings)
    from app.main import app

    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:

        def run(query: str, variables: dict | None = None, headers: dict | None = None):
            r = client.post(
                "/graphql",
                json={"query": query, "variables": variables or {}},
                headers=headers or {"X-API-Key": SERVICE_KEY},
            )
            return r.json()

        run.client = client
        run.settings = settings
        yield run
    app.dependency_overrides.clear()


SERVICE_KEY = "svc-test-key"  # GraphQL is for internal services: POST needs this key

TREE_QUERY = """
query($id: ID!) {
  tree(id: $id) {
    id name version
    starting_node { id name }
    nodes { id name type is_start mlmodel { name parameters }
            fields { customer_key field_type { name display_type } child_fields { customer_key } } }
    branches { condition parent_node { name } child_node { name } }
    projects { name customer { name } }
  }
}
"""


def test_platform_style_tree_query(gql):
    data = gql("{ customers { name configurations { name tree { id name } } } }")["data"]
    selective = next(c for c in data["customers"] if c["name"] == "Selective")
    tree_id = selective["configurations"][0]["tree"]["id"]

    tree = gql(TREE_QUERY, {"id": tree_id})["data"]["tree"]
    assert tree["starting_node"]["name"] == "Document Type"
    edges = {
        (b["parent_node"]["name"], b["child_node"]["name"]): b["condition"]
        for b in tree["branches"]
    }
    assert edges[("Document Type", "Loss Runs Policy Table")] == 'document_type == "Loss Runs"'
    claims = next(n for n in tree["nodes"] if n["name"] == "Loss Runs Claims Table")
    table = claims["fields"][0]
    assert table["field_type"] == {"name": "table", "display_type": "multiselect"}
    assert len(table["child_fields"]) == 29
    assert claims["mlmodel"]["name"] == "GPT-4.1"
    assert tree["projects"][0]["customer"]["name"] == "Selective"


def test_crud_and_tree_mutations(gql):
    cust = gql('mutation { createCustomer(input: {name: "Acme"}) { id name } }')
    customer_id = cust["data"]["createCustomer"]["id"]

    model = gql(
        """mutation($p: JSON!) { createMlModel(input: {name: "Local", type: "ollama",
                  parameters: $p}) { id } }""",
        {"p": {"model_type": "ollama", "model_name": "llama3.2:3b"}},
    )
    model_id = model["data"]["createMlModel"]["id"]

    node_id = str(uuid.uuid4())
    created = gql(
        """
      mutation($input: TreeCreateInput!) { createTree(input: $input) { id version nodes { id is_start } } }
    """,
        {
            "input": {
                "name": "Invoices",
                "branches": [],
                "nodes": [
                    {
                        "id": node_id,
                        "name": "Classifier",
                        "type": "gpt_node",
                        "mlmodel_id": model_id,
                        "start_node": True,
                        "parameters": {
                            "prompt_template": {"system": "{extraction_fields}", "human": "{text}"}
                        },
                        "fields": [{"name": "Document Type", "customer_key": "document_type"}],
                    }
                ],
            }
        },
    )
    tree = created["data"]["createTree"]
    assert tree["version"] == 1 and tree["nodes"] == [{"id": node_id, "is_start": True}]

    # add a table node + a branch, edit a node, rename the tree
    table_node = gql(
        """
      mutation($t: ID!, $input: NodeInput!) { createNode(tree_id: $t, input: $input) { id name } }
    """,
        {
            "t": tree["id"],
            "input": {
                "name": "Lines",
                "type": "code_node",
                "parameters": {
                    "function": "tables.chunking.process",
                    "prompt_template": {"system": "{extraction_fields}", "human": "{text}"},
                },
                "fields": [
                    {
                        "name": "Lines",
                        "customer_key": "lines",
                        "field_type": "table",
                        "child_fields": [
                            {"name": "Amount", "customer_key": "amount", "field_type": "currency"}
                        ],
                    }
                ],
            },
        },
    )["data"]["createNode"]
    branch = gql(
        """
      mutation($t: ID!, $input: BranchInput!) { createBranch(tree_id: $t, input: $input) {
        id condition parent_node { name } child_node { name } } }
    """,
        {
            "t": tree["id"],
            "input": {
                "parent_node_id": node_id,
                "child_node_id": table_node["id"],
                "condition": 'document_type == "Invoice"',
            },
        },
    )
    b = branch["data"]["createBranch"]
    assert b["parent_node"]["name"] == "Classifier" and b["child_node"]["name"] == "Lines"

    upd = gql(
        """mutation($id: ID!) { updateBranch(id: $id, input: {condition: ""}) { condition } }""",
        {"id": b["id"]},
    )
    assert upd["data"]["updateBranch"]["condition"] in (None, "")

    node = gql(
        """mutation($id: ID!) { updateNode(id: $id, input: {name: "Doc Type"}) { name } }""",
        {"id": node_id},
    )
    assert node["data"]["updateNode"]["name"] == "Doc Type"

    renamed = gql(
        """mutation($id: ID!) { updateTree(id: $id, input: {name: "Invoices v2"}) {
                       name version nodes { name } branches { id } } }""",
        {"id": tree["id"]},
    )
    t = renamed["data"]["updateTree"]
    assert t["name"] == "Invoices v2" and len(t["nodes"]) == 2 and len(t["branches"]) == 1

    proj = gql(
        """mutation($c: ID!, $t: ID!) { createProject(input: {customer_id: $c, name: "AP",
                  tree_id: $t}) { id tree { name } customer { name } } }""",
        {"c": customer_id, "t": tree["id"]},
    )
    assert proj["data"]["createProject"]["tree"]["name"] == "Invoices v2"
    config = gql(
        """query($p: ID!) { projectConfig(project_id: $p) }""",
        {"p": proj["data"]["createProject"]["id"]},
    )["data"]["projectConfig"]
    assert list(config["mlmodels"]) == [model_id]

    # deleting a tree in use fails and changes nothing
    err = gql("""mutation($id: ID!) { deleteTree(id: $id) { id } }""", {"id": tree["id"]})
    assert err["errors"][0]["extensions"]["code"] == "VALIDATION_ERROR"

    dup = gql(
        """mutation($id: ID!) { duplicateTree(id: $id) { name nodes { id } } }""",
        {"id": tree["id"]},
    )["data"]["duplicateTree"]
    assert dup["name"] == "Invoices v2 (copy)" and node_id not in {n["id"] for n in dup["nodes"]}

    gone = gql("""mutation($id: ID!) { deleteBranch(id: $id) { id } }""", {"id": b["id"]})
    assert gone["data"]["deleteBranch"]["id"] == b["id"]
    assert (
        gql("""mutation($id: ID!) { deleteCustomer(id: $id) { id } }""", {"id": customer_id})[
            "data"
        ]["deleteCustomer"]["id"]
        == customer_id
    )


def test_validation_errors_and_rollback(gql):
    before = len(gql("{ trees { id } }")["data"]["trees"])
    bad = gql(
        """mutation($input: TreeCreateInput!) { createTree(input: $input) { id } }""",
        {
            "input": {
                "name": "Broken",
                "branches": [],
                "nodes": [
                    {
                        "id": str(uuid.uuid4()),
                        "name": "T",
                        "type": "code_node",
                        "parameters": {"function": "nope"},
                        "fields": [],
                    }
                ],
            }
        },
    )
    err = bad["errors"][0]
    assert err["extensions"]["code"] == "VALIDATION_ERROR"
    assert any("function" in e for e in err["extensions"]["errors"])
    assert len(gql("{ trees { id } }")["data"]["trees"]) == before  # no orphan tree row

    res = gql(
        """mutation($input: TreeCreateInput!) { validateTree(input: $input) { valid errors } }""",
        {
            "input": {
                "name": "x",
                "nodes": [],
                "branches": [{"parent_node_id": "a", "child_node_id": "b"}],
            }
        },
    )
    assert res["data"]["validateTree"]["valid"] is False

    model = gql("""mutation { createMlModel(input: {name: "G", type: "vertex_ai",
                   parameters: {model_type: "vertex_ai", model_name: "g"}}) { id } }""")
    assert "model_type" in model["errors"][0]["message"]
    secret = gql("""mutation { createMlModel(input: {name: "G", type: "gpt",
                    parameters: {model_type: "openai", model_name: "g", api_key: "x"}}) { id } }""")
    assert "secrets" in secret["errors"][0]["message"]

    missing = gql('{ tree(id: "nope") { id } }')
    assert missing["data"]["tree"] is None


def test_import_tree(gql):
    export = {
        "data": {
            "tree": {
                "starting_node": {
                    "id": "n1",
                    "name": "Classifier",
                    "type": "gpt_node",
                    "parameters": {
                        "prompt_template": {"system": "{extraction_fields}", "human": "{text}"}
                    },
                    "mlmodel": {
                        "id": "m-x",
                        "name": "Fine-tune",
                        "type": "finetuned_streaming",
                        "parameters": {"model_type": "finetuned_streaming", "api_key": "SECRET"},
                    },
                    "fields": [
                        {
                            "name": "Document Type",
                            "customer_key": "document_type",
                            "field_type": {"name": "string"},
                        }
                    ],
                }
            }
        }
    }
    r = gql(
        """mutation($e: JSON!) { importTree(export: $e, name: "Imported") {
                tree { name starting_node { name mlmodel { name } } } remapped_models dropped_secrets } }""",
        {"e": export},
    )["data"]["importTree"]
    assert r["tree"]["starting_node"]["name"] == "Classifier"
    assert r["tree"]["starting_node"]["mlmodel"]["name"] == "GPT-4.1"
    assert r["dropped_secrets"] == [{"model": "Fine-tune", "keys": ["api_key"]}]


def test_api_key_and_explorer(gql):
    r = gql.client.post("/graphql", json={"query": "{ trees { id } }"})
    assert r.status_code == 401
    wrong = gql("{ trees { id } }", headers={"X-API-Key": "nope"})
    assert "errors" in wrong
    ok = gql("{ trees { id } }", headers={"X-API-Key": SERVICE_KEY})
    assert ok["data"]["trees"] is not None
    page = gql.client.get("/graphql", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "graphiql" in page.text.lower()


def test_meta(gql):
    meta = gql("{ meta }")["data"]["meta"]
    assert meta["supported_model_types"] == ["azure_openai", "openai", "ollama"]
