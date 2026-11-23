"""Tenant roles and permissions (v1 roots-auth), and v1 igpt shapes."""

# ruff: noqa: F811  (pytest fixtures imported from test_auth)

from __future__ import annotations

from tests.test_auth import _customers, _login, api, world  # noqa: F401  (fixtures)


def _sel(world):
    return world["ids"]["Selective"]


def test_every_tenant_has_v1_roles_and_permission(api, world):
    world["client"] = api
    admin, sel = world["admin"], _sel(world)
    roles = {r["role_name"]: r for r in world_get(world, f"/v1/tenant/{sel}/roles", admin)}
    assert set(roles) == {"ROOTS_INTERNAL_ADMIN", "ANNOTATOR"}
    assert roles["ROOTS_INTERNAL_ADMIN"]["permissions"] == ["Update"]
    assert roles["ANNOTATOR"]["permissions"] == []
    assert {"tenant_role_id", "tenant_id", "role_name", "created_at", "created_by"} <= set(
        roles["ANNOTATOR"]
    )

    perms = world_get(world, f"/v1/tenant/{sel}/permission", admin)
    assert perms["status"] == 200 and perms["statusText"] == "OK"
    assert [p["tenant_permission_name"] for p in perms["data"]] == ["Update"]
    assert perms["data"][0]["permission_description"] == "ability to update"

    rows = world_get(world, f"/v1/tenant/{sel}/role?permissions", admin)
    by_role = {r["role_name"]: r["tenant_permission_name"] for r in rows}
    assert by_role == {"ROOTS_INTERNAL_ADMIN": "Update", "ANNOTATOR": None}

    # The seeded editor became ROOTS_INTERNAL_ADMIN (keeps write access); admins are listed.
    members = world_get(world, f"/v1/tenant/{sel}/users/roles", admin)
    editor = next(m for m in members if m["username"] == "editor")
    assert editor["role_name"] == "ROOTS_INTERNAL_ADMIN" and editor["last_active_time"]
    assert any(m["is_platform_admin"] for m in members)
    assert {"tenant_user_id", "platform_user_id", "display_name", "first_name", "email"} <= set(
        editor
    )

    projects = world_get(world, f"/v1/tenant/{sel}/projects", admin)
    assert projects and projects[0]["membersCount"] >= 1
    assert projects[0]["document_project_id"] is None  # v1: not tied to a document project
    assert "queue_id" in projects[0]


def world_get(world, path, headers):
    api = world["client"]
    r = api.get(path, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def test_tenant_role_decides_write_access(api, world):
    world["client"] = api
    admin, editor, sel = world["admin"], world["editor"], _sel(world)
    project = api.get(f"/v1/customers/{sel}/projects", headers=editor).json()[0]
    patch = lambda h: api.patch(  # noqa: E731
        f"/v1/projects/{project['id']}", headers=h, json={"description": "x"}
    )
    assert patch(editor).status_code == 200
    me = api.get("/v1/me", headers=editor).json()
    assert me["tenants"][0]["role_name"] == "ROOTS_INTERNAL_ADMIN"
    assert me["tenants"][0]["permissions"] == ["Update"]

    members = api.get(f"/v1/tenant/{sel}/users/roles", headers=admin).json()
    editor_id = next(m["platform_user_id"] for m in members if m["username"] == "editor")
    r = api.put(
        f"/v1/tenant/{sel}/users/{editor_id}", headers=admin, json={"role_name": "ANNOTATOR"}
    )
    assert r.status_code == 200 and r.json()["role_name"] == "ANNOTATOR"
    assert patch(editor).status_code == 403  # ANNOTATOR: no Update
    assert (
        api.get(f"/v1/projects/{project['id']}", headers=editor).status_code == 200
    )  # still reads
    assert (
        api.post(f"/v1/tenant/{sel}/roles", headers=editor, json={"role_name": "X"}).status_code
        == 403
    )

    api.put(
        f"/v1/tenant/{sel}/users/{editor_id}",
        headers=admin,
        json={"role_name": "ROOTS_INTERNAL_ADMIN"},
    )
    assert patch(editor).status_code == 200


def test_invite_users_and_manage_roles(api, world):
    world["client"] = api
    admin, viewer, sel = world["admin"], world["viewer"], _sel(world)
    r = api.post(
        f"/v1/tenant/{sel}/users",
        headers=admin,
        json={
            "email": "new.person@example.com",
            "first_name": "New",
            "last_name": "Person",
            "role_name": "ANNOTATOR",
        },
    )
    assert r.status_code == 201, r.text
    invited = r.json()
    assert invited["role_name"] == "ANNOTATOR" and invited["display_name"] == "New Person"
    headers = _login(api, "new.person@example.com", invited["temporary_password"])
    assert api.get("/v1/me", headers=headers).json()["tenants"][0]["customer_id"] == sel
    # Inviting the same person again just updates their role (no new password).
    again = api.post(
        f"/v1/tenant/{sel}/users",
        headers=admin,
        json={"email": "NEW.PERSON@example.com", "role_name": "ROOTS_INTERNAL_ADMIN"},
    )
    assert again.status_code == 201 and "temporary_password" not in again.json()
    assert again.json()["role_name"] == "ROOTS_INTERNAL_ADMIN"

    # Custom roles
    perms = api.get(f"/v1/tenant/{sel}/permission", headers=admin).json()["data"]
    made = api.post(
        f"/v1/tenant/{sel}/roles",
        headers=admin,
        json={"role_name": "team lead", "permission_ids": [perms[0]["tenant_permission_id"]]},
    )
    assert made.status_code == 201 and made.json()["role_name"] == "TEAM_LEAD"
    assert made.json()["permissions"] == ["Update"]
    rid = made.json()["tenant_role_id"]
    assert (
        api.post(
            f"/v1/tenant/{sel}/roles", headers=admin, json={"role_name": "TEAM_LEAD"}
        ).status_code
        == 409
    )
    api.put(
        f"/v1/tenant/{sel}/users/{invited['platform_user_id']}",
        headers=admin,
        json={"tenant_role_id": rid},
    )
    assert api.delete(f"/v1/tenant/{sel}/roles/{rid}", headers=admin).status_code == 409  # in use
    builtin = next(
        r_["tenant_role_id"]
        for r_ in api.get(f"/v1/tenant/{sel}/roles", headers=admin).json()
        if r_["role_name"] == "ANNOTATOR"
    )
    assert api.delete(f"/v1/tenant/{sel}/roles/{builtin}", headers=admin).status_code == 409
    renamed = api.put(f"/v1/tenant/{sel}/roles/{rid}", headers=admin, json={"permissions": []})
    assert renamed.json()["permissions"] == [] and renamed.json()["users_count"] == 1
    assert (
        api.delete(
            f"/v1/tenant/{sel}/users/{invited['platform_user_id']}", headers=admin
        ).status_code
        == 204
    )
    assert api.delete(f"/v1/tenant/{sel}/roles/{rid}", headers=admin).status_code == 204

    # Someone else's tenant stays hidden.
    assert api.get(f"/v1/tenant/{sel}/roles", headers=viewer).status_code == 404


def test_v1_igpt_shapes_and_uppercase_ids(api, world):
    admin, sel = world["admin"], _sel(world)
    upper = api.get(f"/v1/igpt/customers/{sel.upper()}", headers=admin)
    assert upper.status_code == 200 and upper.json()["customer"]["id"] == sel
    project = upper.json()["customer"]["configurations"][0]["id"]
    tree = api.get(f"/v1/igpt/configurations/{project.upper()}", headers=admin).json()[
        "configuration"
    ]["tree"]
    edge = tree["layout"]["edges"][0]
    assert {"type", "logicType", "sourceHandle", "targetHandle", "source", "target"} <= set(edge)
    assert edge["type"] == "addWorkflowEdge"
    assert {"data", "type", "position"} <= set(tree["layout"]["nodes"][0])
    assert {"parent_node", "child_node", "standard"} <= set(tree["branches"][0])
    node = tree["nodes"][0]
    assert {"mlmodel", "fields", "display_template"} <= set(node)
    field = node["fields"][0]
    assert {"id", "name", "display_type"} <= set(field["field_type"])
    assert {"sort_order", "placeholder_text", "options"} <= set(field["field_display"])
    library = api.get(f"/v1/igpt/project-field-library/{project}", headers=admin).json()
    entry = library["projectFieldLibrary"][0]
    assert {"template", "project_id", "field_display", "child_fields"} <= set(entry)
    leaf = node["id"]
    tpl = api.get(f"/v1/nodes/{leaf}/display-template", headers=admin).json()
    prop = next(iter(tpl["node"]["display_template"]["template"]["properties"].values()))
    assert {"sort_order", "date_format", "number_of_rows"} <= set(prop["field_display"])


def test_decision_trees_belong_to_the_org_library(api, world):
    """Any member of an org opens all of its trees (not only trees a project uses); members with
    Update create, edit and delete them; other orgs never see them."""
    admin, editor, viewer = world["admin"], world["editor"], world["viewer"]
    sel = _sel(world)
    tree = {"name": "Library only", "customer_id": sel, "nodes": [], "branches": []}

    created = api.post("/v1/trees", json=tree, headers=editor)  # editor: Update on Selective
    assert created.status_code == 201, created.text
    tid = created.json()["id"]
    assert created.json()["customer_id"] == sel

    library = api.get("/v1/trees", params={"customer_id": sel}, headers=editor).json()
    assert tid in {t["id"] for t in library}
    assert api.get(f"/v1/trees/{tid}", headers=editor).status_code == 200

    # The viewer is not a member of Selective: the tree is hidden.
    assert api.get(f"/v1/trees/{tid}", headers=viewer).status_code == 404
    assert tid not in {t["id"] for t in api.get("/v1/trees", headers=viewer).json()}
    other = dict(tree, customer_id=sel, name="Not mine")
    assert api.post("/v1/trees", json=other, headers=viewer).status_code in (403, 404)

    # Saving keeps the owner (only an admin moves a tree to another org).
    body = created.json() | {"name": "Library only v2", "customer_id": None}
    assert api.put(f"/v1/trees/{tid}", json=body, headers=editor).json()["customer_id"] == sel

    copy = api.post(f"/v1/trees/{tid}/duplicate", headers=editor)
    assert copy.status_code == 201 and copy.json()["customer_id"] == sel
    for t in (tid, copy.json()["id"]):
        assert api.delete(f"/v1/trees/{t}", headers=editor).status_code == 204

    # A tree with no org stays admin-only.
    assert (
        api.post("/v1/trees", json=dict(tree, customer_id=None), headers=editor).status_code == 403
    )
    assert (
        api.post("/v1/trees", json=dict(tree, customer_id=None), headers=admin).status_code == 201
    )


def test_tenant_projects_have_their_own_queue_and_see_the_org_library(api, world):
    """v1 roots-auth projects: a project has a queue id, not a decision tree; its field library
    (v1 calls project-field-library with the page's project) covers every tree of the org."""
    admin, editor, viewer = world["admin"], world["editor"], world["viewer"]
    sel = _sel(world)
    queue = "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"
    r = api.post(
        f"/v1/tenant/{sel.upper()}/projects",
        json={"display_name": "Claims intake", "queue_id": queue},
        headers=editor,
    )
    assert r.status_code == 201, r.text
    project = r.json()
    assert project["queue_id"] == queue.replace("-", "").lower()
    assert project["tenant_id"] == sel and project["document_project_id"] is None
    pid = project["project_id"]

    listed = api.get(f"/v1/tenant/{sel}/projects", headers=editor).json()
    assert pid in {p["project_id"] for p in listed}
    assert api.get(f"/v1/tenant/{sel}/projects/{pid}", headers=editor).status_code == 200
    # Another org's member can't see or change it.
    assert api.get(f"/v1/tenant/{sel}/projects", headers=viewer).status_code == 404

    lib = api.get(f"/v1/igpt/project-field-library/{pid}", headers=editor).json()
    entries = lib["projectFieldLibrary"]
    assert entries and all(e["project_id"] == pid for e in entries)
    keys = {e["customer_key"] for e in entries}
    org_trees = api.get("/v1/trees", params={"customer_id": sel}, headers=editor).json()
    assert org_trees and len(keys) >= 1

    moved = api.put(
        f"/v1/tenant/{sel}/projects/{pid}",
        json={"display_name": "Claims", "queue_id": None},
        headers=editor,
    ).json()
    assert moved["display_name"] == "Claims" and moved["queue_id"] is None
    assert api.delete(f"/v1/tenant/{sel}/projects/{pid}", headers=admin).status_code == 204
    assert api.get(f"/v1/tenant/{sel}/projects/{pid}", headers=admin).status_code == 404
