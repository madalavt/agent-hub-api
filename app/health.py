"""Health checks for the decision-tree API.

``GET /health/live``  the process is up (no dependency checks).
``GET /health``       database check plus catalog counts; 503 when the database is down.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from sqlalchemy import func, select, text

from app.db import session_scope
from app.db_read import read_status
from app.models import Customer, MlModel, Project, Tree

SERVICE = "agent-hub-api"
STARTED_AT = time.monotonic()


def version() -> str:
    try:
        return metadata.version(SERVICE)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "dev"


def liveness() -> dict[str, Any]:
    return {
        "service": SERVICE,
        "status": "ok",
        "version": version(),
        "uptime_s": round(time.monotonic() - STARTED_AT, 1),
    }


def readiness() -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    check: dict[str, Any] = {"critical": True, "details": {}, "error": None}
    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
            check["details"] = {
                "dialect": session.get_bind().dialect.name,
                "customers": session.scalar(select(func.count()).select_from(Customer)),
                "projects": session.scalar(select(func.count()).select_from(Project)),
                "trees": session.scalar(select(func.count()).select_from(Tree)),
                "ml_models": session.scalar(select(func.count()).select_from(MlModel)),
            }
        check["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        check["status"] = "down"
        first = str(exc).splitlines()[0] if str(exc) else ""
        check["error"] = f"{type(exc).__name__}: {first}"[:300]
    check["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    status = "ok" if check["status"] == "ok" else "down"
    body = {
        **liveness(),
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": {"database": check},
        # Informational: reads fall back to the primary, so this never fails readiness.
        "read_replica": read_status(),
    }
    return (200 if status == "ok" else 503), body
