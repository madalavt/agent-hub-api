"""GraphQL endpoint for the decision-tree API (Ariadne, schema-first like
Platform-DecisionTree-API).

    POST /graphql   queries and mutations (service X-API-Key only; users go through /v1)
    GET  /graphql   GraphiQL explorer
"""

from __future__ import annotations

import secrets
from pathlib import Path

from ariadne import load_schema_from_path, make_executable_schema
from ariadne.asgi import GraphQL
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import session_scope
from app.gql.resolvers import OBJECT_TYPES

SCHEMA_PATH = Path(__file__).with_name("schema.graphql")
schema = make_executable_schema(load_schema_from_path(str(SCHEMA_PATH)), *OBJECT_TYPES)


def _context(request: Request, data: dict) -> dict:
    return {"request": request, "session": request.state.db, "settings": get_settings()}


graphql_app = GraphQL(schema, context_value=_context, debug=False)
router = APIRouter()


@router.api_route("/graphql", methods=["GET", "POST"], include_in_schema=False)
async def graphql_endpoint(request: Request):
    settings = get_settings()
    # GraphQL resolvers do not apply the role rules, so queries and mutations are for internal
    # services only: they need the service key (fails closed when API_KEY is not set).
    if request.method == "POST":
        key = request.headers.get("x-api-key") or ""
        if not settings.api_key or not secrets.compare_digest(
            key.encode("utf-8"), settings.api_key.encode("utf-8")
        ):
            return JSONResponse(
                {
                    "errors": [
                        {
                            "message": "invalid or missing X-API-Key",
                            "extensions": {"code": "UNAUTHENTICATED"},
                        }
                    ]
                },
                status_code=401,
            )
    with session_scope() as session:  # committed after execution; mutations roll back on error
        request.state.db = session
        return await graphql_app.handle_request(request)
