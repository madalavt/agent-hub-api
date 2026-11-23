"""Regression tests for bugs identified and fixed in the decision-tree API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import conditions
from app import db as db_module
from app.config import Settings, get_settings

SECRET = "test-secret"
ADMIN_PW = "admin-test-pw"


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/reg.db",
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


def test_role_demotion_and_promotion_sync_tenant_roles(api):
    admin = _login(api, "admin", ADMIN_PW)
    customers = {c["name"]: c["id"] for c in api.get("/v1/customers", headers=admin).json()}
    cid = customers["Selective"]
    proj = api.get(f"/v1/customers/{cid}/projects", headers=admin).json()[0]

    created = api.post(
        "/v1/users",
        headers=admin,
        json={
            "username": "bob",
            "name": "Bob",
            "password": "password123",
            "role": "editor",
            "customer_ids": [cid],
        },
    ).json()
    bob = _login(api, "bob", "password123")
    assert api.patch(f"/v1/projects/{proj['id']}", headers=bob, json={"description": "ok"}).status_code == 200

    # Demote editor -> viewer: must revoke Update permission on assigned tenant immediately.
    r = api.put(f"/v1/users/{created['id']}", headers=admin, json={"role": "viewer"})
    assert r.status_code == 200
    assert api.patch(f"/v1/projects/{proj['id']}", headers=bob, json={"description": "nope"}).status_code == 403

    # Promote viewer -> editor: must restore Update permission on assigned tenant.
    r = api.put(f"/v1/users/{created['id']}", headers=admin, json={"role": "editor"})
    assert r.status_code == 200
    assert api.patch(f"/v1/projects/{proj['id']}", headers=bob, json={"description": "again"}).status_code == 200


def test_non_ascii_auth_headers_return_401_not_500(api):
    import httpx

    from app.auth import tokens

    assert tokens.verify(SECRET, "abc.é") is None
    r1 = api.get("/v1/customers", headers=httpx.Headers([(b"authorization", b"Bearer abc.\xe9")]))
    assert r1.status_code == 401
    r2 = api.get("/v1/customers", headers=httpx.Headers([(b"x-api-key", b"\xe9")]))
    assert r2.status_code == 401
    r3 = api.post(
        "/graphql",
        headers=httpx.Headers([(b"x-api-key", b"\xe9")]),
        json={"query": "{ customers { id } }"},
    )
    assert r3.status_code == 401


def test_case_insensitive_login_for_invited_user(api):
    admin = _login(api, "admin", ADMIN_PW)
    cid = api.get("/v1/customers", headers=admin).json()[0]["id"]
    inv = api.post(
        f"/v1/tenant/{cid}/users",
        headers=admin,
        json={"email": "Alice.Mixed@Example.com", "display_name": "Alice", "role_name": "ANNOTATOR"},
    )
    assert inv.status_code == 201, inv.text
    pw = inv.json()["temporary_password"]
    assert api.post("/auth/login", json={"username": "alice.mixed@example.com", "password": pw}).status_code == 200


def test_project_lookup_unicode_digit_and_overflow_return_404(api):
    admin = _login(api, "admin", ADMIN_PW)
    assert api.get("/v1/projects/²", headers=admin).status_code == 404
    assert api.get("/v1/projects/99999999999999999999", headers=admin).status_code == 404


def test_patch_project_rejects_explicit_null_name_and_parameters(api):
    admin = _login(api, "admin", ADMIN_PW)
    proj = api.get("/v1/projects", headers=admin).json()[0]
    assert api.patch(f"/v1/projects/{proj['id']}", headers=admin, json={"name": None}).status_code == 422
    assert api.patch(f"/v1/projects/{proj['id']}", headers=admin, json={"parameters": None}).status_code == 422


def test_built_in_admin_role_cannot_lose_update_permission(api):
    admin = _login(api, "admin", ADMIN_PW)
    cid = api.get("/v1/customers", headers=admin).json()[0]["id"]
    roles = api.get(f"/v1/tenant/{cid}/roles", headers=admin).json()
    admin_role = next(r for r in roles if r["role_name"] == "ROOTS_INTERNAL_ADMIN")
    # Passing same role_name is allowed; stripping Update from ROOTS_INTERNAL_ADMIN is refused.
    r = api.put(
        f"/v1/tenant/{cid}/roles/{admin_role['tenant_role_id']}",
        headers=admin,
        json={"role_name": "ROOTS_INTERNAL_ADMIN", "permissions": []},
    )
    assert r.status_code == 409


def test_tree_validate_handles_idless_nodes_and_conditions_edge_cases(api):
    admin = _login(api, "admin", ADMIN_PW)
    # Two distinct nodes with id=None sharing a customer_key on sibling branches should not false-positive
    # as same node (None == None).
    r = api.post(
        "/v1/trees/validate",
        headers=admin,
        json={
            "name": "Test",
            "nodes": [
                {
                    "name": "A",
                    "type": "gpt_node",
                    "fields": [{"name": "P", "customer_key": "placeholder"}],
                },
                {
                    "name": "B",
                    "type": "gpt_node",
                    "fields": [{"name": "P", "customer_key": "placeholder"}],
                },
            ],
            "branches": [],
        },
    )
    assert r.status_code == 200
    assert not any("placeholder" in e for e in r.json()["errors"])

    # Conditions: multiline whitespace, escaped trailing backslash, numeric currency comparison.
    assert conditions.evaluate('a == "1"\nand\tb == "2"', {"a": "1", "b": "2"}) is True
    assert conditions.evaluate('a == "foo\\\\" and b == "bar"', {"a": "foo\\", "b": "bar"}) is True
    from decimal import Decimal

    assert conditions.evaluate("amount == 100", {"amount": Decimal("100.00")}) is True
    assert conditions.evaluate("amount == 100.0", {"amount": Decimal("100.00")}) is True
