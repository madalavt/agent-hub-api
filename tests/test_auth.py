from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app.auth import tokens
from app.config import Settings, get_settings

SECRET = "test-secret"
ADMIN_PW = "admin-test-pw"


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/auth.db",
        seed_admin_password=ADMIN_PW,
        session_secret=SECRET,
        api_key="svc-key",
    )
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    from app.main import app

    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _login(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _customers(client, headers):
    return {c["name"]: c["id"] for c in client.get("/v1/customers", headers=headers).json()}


@pytest.fixture
def world(api):
    """The seeded editor (Selective) and viewer (Demo) get known passwords; headers per role."""
    admin = _login(api, "admin", ADMIN_PW)
    ids = _customers(api, admin)
    users = {u["username"]: u["id"] for u in api.get("/v1/users", headers=admin).json()}
    made = {}
    for username, customer in (("editor", "Selective"), ("viewer", "Demo")):
        r = api.put(
            f"/v1/users/{users[username]}",
            headers=admin,
            json={"password": f"{username}-password", "customer_ids": [ids[customer]]},
        )
        assert r.status_code == 200, r.text
        made[username] = _login(api, username, f"{username}-password")
    return {"admin": admin, "editor": made["editor"], "viewer": made["viewer"], "ids": ids}


def test_login_rejects_bad_password_and_unknown_user(api):
    assert (
        api.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    )
    assert api.post("/auth/login", json={"username": "ghost", "password": "x"}).status_code == 401


def test_providers_lists_enabled_sign_in_methods(api):
    assert api.get("/auth/providers").json() == [{"name": "local", "kind": "password"}]


def test_requests_without_identity_are_refused(api):
    assert api.get("/v1/customers").status_code == 401
    assert api.get("/v1/trees").status_code == 401


def test_tampered_and_expired_tokens_are_refused(api):
    admin = _login(api, "admin", ADMIN_PW)
    token = admin["Authorization"].split(" ", 1)[1]
    body, sig = token.split(".")
    forged = {"Authorization": f"Bearer {body}.{sig[:-2]}AA"}
    assert api.get("/v1/customers", headers=forged).status_code == 401
    assert tokens.verify(SECRET, tokens.issue(SECRET, "x", -10)) is None
    assert tokens.verify("other-secret", token) is None


def test_service_key_reads_everything_but_cannot_write(api):
    svc = {"X-API-Key": "svc-key"}
    assert api.get("/v1/customers", headers=svc).status_code == 200
    assert api.post("/v1/customers", headers=svc, json={"name": "Nope"}).status_code == 403


def test_me_returns_the_signed_in_user(api):
    admin = _login(api, "admin", ADMIN_PW)
    me = api.get("/v1/me", headers=admin).json()
    assert me["role"] == "admin" and me["username"] == "admin"


def test_viewer_sees_only_assigned_customers(world, api):
    viewer = world["viewer"]
    names = set(_customers(api, viewer))
    assert names == {"Demo"}
    assert api.get(f"/v1/customers/{world['ids']['Selective']}", headers=viewer).status_code == 404
    assert (
        api.get(f"/v1/customers/{world['ids']['Demo']}/projects", headers=viewer).status_code
        == 200
    )


def test_trees_are_visible_only_through_accessible_customers(world, api):
    viewer = world["viewer"]
    trees = api.get("/v1/trees", headers=viewer).json()
    assert trees, "the Demo customer's trees should be visible"
    for t in trees:
        assert all(p["customer_id"] == world["ids"]["Demo"] for p in t["projects"])
    loss_runs_tree = next(
        t["id"]
        for t in api.get("/v1/trees", headers=world["admin"]).json()
        if any(p["customer_id"] == world["ids"]["Selective"] for p in t["projects"])
    )
    assert api.get(f"/v1/trees/{loss_runs_tree}", headers=viewer).status_code == 404


def test_viewer_cannot_change_anything(world, api):
    viewer = world["viewer"]
    demo = world["ids"]["Demo"]
    assert (
        api.post(f"/v1/customers/{demo}/projects", headers=viewer, json={"name": "X"}).status_code
        == 403
    )
    tree = api.get(f"/v1/customers/{demo}/projects", headers=viewer).json()[0]["tree_id"]
    body = api.get(f"/v1/trees/{tree}", headers=viewer).json()
    assert api.put(f"/v1/trees/{tree}", headers=viewer, json=body).status_code == 403


def test_editor_changes_assigned_projects_only(world, api):
    editor = world["editor"]
    selective = world["ids"]["Selective"]
    assert (
        api.post(
            f"/v1/customers/{selective}/projects", headers=editor, json={"name": "New project"}
        ).status_code
        == 201
    )
    demo = world["ids"]["Demo"]
    assert (
        api.post(
            f"/v1/customers/{demo}/projects", headers=editor, json={"name": "Nope"}
        ).status_code
        == 404
    )
    assert api.post("/v1/customers", headers=editor, json={"name": "Acme"}).status_code == 403


def test_editor_can_only_assign_trees_already_used_by_their_customers(world, api):
    editor = world["editor"]
    selective = world["ids"]["Selective"]
    demo_tree = api.get(
        f"/v1/customers/{world['ids']['Demo']}/projects", headers=world["admin"]
    ).json()[0]["tree_id"]
    r = api.post(
        f"/v1/customers/{selective}/projects",
        headers=editor,
        json={"name": "Wrong tree", "tree_id": demo_tree},
    )
    assert r.status_code == 403


def test_only_admins_create_and_delete_customers_and_trees(world, api):
    editor = world["editor"]
    assert api.delete(f"/v1/customers/{world['ids']['Demo']}", headers=editor).status_code == 403
    assert (
        api.post(
            "/v1/trees", headers=editor, json={"name": "T", "nodes": [], "branches": []}
        ).status_code
        == 403
    )
    assert api.get("/v1/users", headers=editor).status_code == 403


def test_disabled_user_loses_access_immediately(world, api):
    admin = world["admin"]
    viewer_id = api.get("/v1/me", headers=world["viewer"]).json()["id"]
    assert (
        api.put(f"/v1/users/{viewer_id}", headers=admin, json={"active": False}).status_code == 200
    )
    assert api.get("/v1/customers", headers=world["viewer"]).status_code == 401
    assert (
        api.post(
            "/auth/login", json={"username": "viewer", "password": "viewer-password"}
        ).status_code
        == 403
    )


def test_admin_cannot_demote_or_delete_themselves(api):
    admin = _login(api, "admin", ADMIN_PW)
    me = api.get("/v1/me", headers=admin).json()
    assert (
        api.put(f"/v1/users/{me['id']}", headers=admin, json={"role": "viewer"}).status_code == 409
    )
    assert api.delete(f"/v1/users/{me['id']}", headers=admin).status_code == 409


def test_password_change_takes_effect(world, api):
    admin = world["admin"]
    viewer_id = api.get("/v1/me", headers=world["viewer"]).json()["id"]
    assert (
        api.put(
            f"/v1/users/{viewer_id}", headers=admin, json={"password": "brand-new-pw"}
        ).status_code
        == 200
    )
    assert (
        api.post(
            "/auth/login", json={"username": "viewer", "password": "viewer-password"}
        ).status_code
        == 401
    )
    _login(api, "viewer", "brand-new-pw")


def test_projects_list_is_limited_to_assigned_customers(world, api):
    viewer_rows = api.get("/v1/projects", headers=world["viewer"]).json()
    admin_rows = api.get("/v1/projects", headers=world["admin"]).json()
    demo = world["ids"]["Demo"]
    assert viewer_rows and all(p["customer_id"] == demo for p in viewer_rows)
    assert {p["customer_id"] for p in admin_rows} - {demo}


def test_node_and_display_template_follow_tree_access(world, api):
    admin, viewer = world["admin"], world["viewer"]
    demo_tree = api.get(f"/v1/customers/{world['ids']['Demo']}/projects", headers=admin).json()[0][
        "tree_id"
    ]
    tree = api.get(f"/v1/trees/{demo_tree}", headers=admin).json()
    node = next(n for n in tree["nodes"] if n["fields"])
    r = api.get(f"/v1/nodes/{node['id']}", headers=viewer)
    assert r.status_code == 200
    body = r.json()["node"]
    assert body["name"] == node["name"] and body["fields"]
    assert {n["id"] for n in body["tree"]["extraction_nodes"]} == {n["id"] for n in tree["nodes"]}
    tpl = api.get(f"/v1/nodes/{node['id']}/display-template", headers=viewer).json()
    props = tpl["node"]["display_template"]["template"]["properties"]
    first = node["fields"][0]
    assert props[first["id"]]["title"] == first["name"]
    assert props[first["id"]]["customer_key"] == first["customer_key"]
    selective_tree = api.get(
        f"/v1/customers/{world['ids']['Selective']}/projects", headers=admin
    ).json()[0]["tree_id"]
    other = next(n for n in api.get(f"/v1/trees/{selective_tree}", headers=admin).json()["nodes"])
    assert api.get(f"/v1/nodes/{other['id']}", headers=viewer).status_code == 404


def test_igpt_views_follow_customer_access(world, api):
    admin, viewer = world["admin"], world["viewer"]
    demo, selective = world["ids"]["Demo"], world["ids"]["Selective"]
    cust = api.get(f"/v1/igpt/customers/{demo}", headers=viewer).json()["customer"]
    assert cust["id"] == demo and cust["configurations"]
    assert all(
        "api_key" not in c and "api_key" not in (c["parameters"] or {})
        for c in cust["configurations"]
    )
    assert api.get(f"/v1/igpt/customers/{selective}", headers=viewer).status_code == 404
    cfg_id = cust["configurations"][0]["id"]
    cfg = api.get(f"/v1/igpt/configurations/{cfg_id}", headers=viewer).json()["configuration"]
    assert cfg["tree"]["nodes"] and cfg["tree"]["layout"]["nodes"]
    lib = api.get(f"/v1/igpt/project-field-library/{cfg_id}", headers=viewer).json()
    keys = [f["customer_key"] for f in lib["projectFieldLibrary"]]
    assert keys and len(keys) == len(set(keys))
    assert api.get("/v1/igpt/mlmodels", headers=viewer).json()["mlmodels"]
    names = {
        t["field_type"]["name"]
        for t in api.get("/v1/igpt/field-default-template", headers=admin).json()[
            "fieldDefaultTemplates"
        ]
    }
    assert {"string", "number", "table"} <= names
