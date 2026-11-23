"""Agent Hub decision-tree API. Run with: ``uvicorn app.main:app --port 8001``"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app import health
from app.config import get_settings
from app.db import init_db, lock_engine, session_scope, startup_lock
from app.gql import router as graphql_router
from app.routes import router
from app.seed import seed
from app.tenants import ensure_all_tenants
from app.tenants import router as tenants_router
from app.users import auth_router
from app.users import router as users_router

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db(settings.database_url, settings.direct_database_url)
    if settings.seed_on_start:
        with startup_lock(lock_engine()):
            seed(
                settings.seed_dir,
                settings.local_models if settings.seed_local_models else None,
                settings.seed_admin_password,
            )
    # Every tenant gets v1's roles and permission; members get a role (idempotent).
    with startup_lock(lock_engine()), session_scope() as session:
        ensure_all_tenants(session)
    if not settings.session_secret:
        logging.getLogger(__name__).warning(
            "SESSION_SECRET is not set: sessions end when this process restarts. Set it in .env."
        )
    yield


_UUID_SEGMENT = re.compile(
    rb"/([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})(?=/|$)"
)
_UUID_SEGMENT_STR = re.compile(
    r"/([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})(?=/|$)"
)


class LowercaseIdsMiddleware:
    """Ids are lowercase UUIDs; the v1 UI sends them uppercase (/igpt/customer/89949BDC-...)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            lowered_path = _UUID_SEGMENT_STR.sub(lambda m: "/" + m[1].lower(), path)
            raw = scope.get("raw_path")
            lowered_raw = (
                _UUID_SEGMENT.sub(lambda m: b"/" + m[1].lower(), raw) if raw is not None else None
            )
            if lowered_path != path or lowered_raw != raw:
                scope = dict(scope)
                scope["path"] = lowered_path
                if lowered_raw is not None:
                    scope["raw_path"] = lowered_raw
        await self.app(scope, receive, send)


app = FastAPI(
    title="Agent Hub · Decision Tree API",
    version="0.1.0",
    description="Customers, projects, decision trees (nodes, prompts, fields, branches) and ML "
    "model configuration for Agent Hub. REST under /v1; GraphQL at /graphql (GraphiQL on GET).",
    lifespan=lifespan,
)
app.add_middleware(LowercaseIdsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(tenants_router)
app.include_router(router)
app.include_router(graphql_router)


@app.get("/health/live", tags=["health"], summary="Liveness: the process is up")
def health_live() -> dict:
    return health.liveness()


@app.get(
    "/health",
    tags=["health"],
    summary="Readiness: database reachable",
    responses={503: {"description": "Database is down"}},
)
def health_ready(response: Response) -> dict:
    code, body = health.readiness()
    response.status_code = code
    return body
